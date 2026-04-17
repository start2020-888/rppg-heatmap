from __future__ import annotations

"""
rPPG 实时血流灌注热力图 — 基于 POS (Plane-Orthogonal-to-Skin) 算法
==============================================================

核心目的:
  不做心率估计，而是生成逐网格的"皮下血流脉动能量"实时伪彩色热力图，
  用于医学评估（烧伤 / 皮瓣 / 微循环等皮下血流灌注状况的可视化）。

架构设计:
  整个处理管线被封装在 `RPPGProcessor` 类中，核心数学逻辑完全解耦，
  便于未来移植到 C++ / Metal / CUDA。

作者: rPPG-POS Project
日期: 2026-04-17
"""

import time
import numpy as np
import cv2
from scipy.signal import butter, sosfiltfilt

# ============================================================================
# 全局配置 — 集中管理所有可调超参数
# ============================================================================

# 视频采集参数
CAPTURE_WIDTH = 640         # 采集帧宽度（像素）
CAPTURE_HEIGHT = 480        # 采集帧高度（像素）
TARGET_FPS = 30             # 目标帧率

# 网格参数
GRID_ROWS = 40              # 网格行数 — 将画面纵向切分为 40 个块
GRID_COLS = 40              # 网格列数 — 将画面横向切分为 40 个块

# 时序缓冲参数
WINDOW_SIZE = 90            # 滑动窗口长度（帧）≈ 3 秒 @30fps

# 带通滤波参数（生理心率范围 42~180 bpm → 0.7~3.0 Hz）
BANDPASS_LOW = 0.7          # 下限截止频率 (Hz)
BANDPASS_HIGH = 3.0         # 上限截止频率 (Hz)
BUTTER_ORDER = 3            # Butterworth 滤波器阶数

# 热力图渲染参数
HEATMAP_ALPHA = 0.5         # 热力图叠加透明度 (0=全透明, 1=不透明)
COLORMAP = cv2.COLORMAP_TURBO  # 伪彩色映射方案

# 能量平滑参数（指数移动平均）
EMA_ALPHA = 0.15            # EMA 平滑系数 — 越小越平滑，越大越灵敏

# 皮肤检测参数
SKIN_THRESH = 0.3           # 网格内皮肤像素占比阈值（低于此值视为非皮肤区域）

# 进阶扫描参数
CONFIRM_PERCENTILE = 55     # 能量超过皮肤区域该百分位时，确认为"有血流"（锁定为红色）


# ============================================================================
# 核心类: RPPGProcessor
# ============================================================================

class RPPGProcessor:
    """
    rPPG 实时处理器 — 对等距网格执行向量化 POS 算法，输出血流能量图。

    处理管线:
      1. grid_sample()       — 网格化取样，提取每格 RGB 均值
      2. push_to_buffer()    — 将当前帧数据推入环形缓冲区
      3. compute_pos()       — 向量化 POS 算法核心（归一化→正交投影→Alpha→合成）
      4. compute_energy()    — 带通滤波 + 能量（标准差）计算
      5. render_heatmap()    — 伪彩色映射 + 双线性插值 + 叠加

    所有数学运算均使用 NumPy 高维广播（Vectorization），无显式 for 循环。
    """

    def __init__(
        self,
        grid_rows: int = GRID_ROWS,
        grid_cols: int = GRID_COLS,
        window_size: int = WINDOW_SIZE,
        fps: float = TARGET_FPS,
        bandpass_low: float = BANDPASS_LOW,
        bandpass_high: float = BANDPASS_HIGH,
        butter_order: int = BUTTER_ORDER,
        ema_alpha: float = EMA_ALPHA,
    ):
        """
        初始化处理器。

        Args:
            grid_rows:    网格行数
            grid_cols:    网格列数
            window_size:  环形缓冲区长度（帧数）
            fps:          视频帧率，用于计算滤波器系数
            bandpass_low: 带通滤波器下限 (Hz)
            bandpass_high:带通滤波器上限 (Hz)
            butter_order: Butterworth 滤波器阶数
            ema_alpha:    能量图 EMA 平滑系数
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.window_size = window_size
        self.fps = fps
        self.ema_alpha = ema_alpha

        # ---- 环形缓冲区 ----
        # 形状: (window_size, grid_rows, grid_cols, 3)
        # 存储每一帧中每个网格的 [R, G, B] 均值
        self.buffer = np.zeros(
            (window_size, grid_rows, grid_cols, 3), dtype=np.float32
        )
        self.buf_index = 0       # 当前写入位置（环形指针）
        self.buf_count = 0       # 已写入帧数（用于判断缓冲区是否已满）

        # ---- 预计算带通滤波器系数（Second-Order Sections 形式，更稳定）----
        nyquist = fps / 2.0
        low_norm = bandpass_low / nyquist
        high_norm = bandpass_high / nyquist
        # 钳位到 (0, 1) 范围内，防止异常参数导致崩溃
        low_norm = np.clip(low_norm, 1e-5, 1.0 - 1e-5)
        high_norm = np.clip(high_norm, low_norm + 1e-5, 1.0 - 1e-5)
        self.sos = butter(butter_order, [low_norm, high_norm], btype='band', output='sos')

        # ---- 平滑后的能量图（用于 EMA 时域平滑，避免闪烁）----
        self.energy_smooth = None  # 延迟初始化

        # ---- 皮肤检测相关 ----
        self.skin_thresh = SKIN_THRESH
        # 形态学核 — 用于闭运算去噪（预分配，避免每帧重建）
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # ---- 进阶扫描状态 ----
        self.scanning = False   # 是否正在扫描
        # 已确认的血流图: 值域 [0, 1]，0=未确认，>0=已确认的归一化能量
        self.confirmed_map = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.confirm_percentile = CONFIRM_PERCENTILE
        self.scan_frame_count = 0  # 扫描开始后的帧计数

    # -----------------------------------------------------------------------
    # 扫描控制方法
    # -----------------------------------------------------------------------
    def start_scan(self) -> None:
        """开始扫描。缓冲区保持不变，从当前帧开始累积确认。"""
        self.scanning = True
        self.scan_frame_count = 0
        print("[SCAN] ┣─ 扫描开始")

    def stop_scan(self) -> None:
        """停止扫描。已确认的区域保持不变。"""
        self.scanning = False
        confirmed_count = (self.confirmed_map > 0).sum()
        total = self.grid_rows * self.grid_cols
        print(f"[SCAN] ┣─ 扫描停止 | 已确认: {confirmed_count}/{total} 网格")

    def reset_scan(self) -> None:
        """重置所有已确认区域，回到全蓝状态。"""
        self.confirmed_map = np.zeros(
            (self.grid_rows, self.grid_cols), dtype=np.float32
        )
        self.energy_smooth = None
        self.scan_frame_count = 0
        print("[SCAN] ┣─ 已重置")

    # -----------------------------------------------------------------------
    # 第 0 步: 皮肤检测 — HSV + YCrCb 双色彩空间
    # -----------------------------------------------------------------------
    def detect_skin(self, frame_bgr: np.ndarray) -> tuple:
        """
        基于 HSV + YCrCb 双色彩空间阈值的皮肤检测。

        不依赖任何 ML/DL 库，纯色彩空间阈值 + 形态学运算。
        返回像素级掩膜和网格级掩膜。

        算法原理:
          1. HSV 空间: 皮肤色调(Hue)在 0~25 和 165~180 范围，
             饱和度(Saturation) > 20, 亮度(Value) > 50
          2. YCrCb 空间: Cr ∈ [135, 180], Cb ∈ [85, 135]
             这对应人类皮肤在色度平面上的聚类区域
          3. 两个掩膜取交集（AND），提高鲁棒性
          4. 形态学闭运算填充小孔洞

        Args:
            frame_bgr: BGR 格式帧, shape = (H, W, 3)

        Returns:
            skin_mask_pixel: 像素级二值掩膜, shape = (H, W), dtype=uint8, 值 0 或 255
            skin_mask_grid:  网格级连续掩膜, shape = (grid_rows, grid_cols), dtype=float32
                             值域 [0, 1]，表示每个网格内皮肤像素占比
        """
        # ---- HSV 色彩空间检测 ----
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # 皮肤色调范围 1: 红色系低端 (Hue 0~25)
        mask_hsv1 = cv2.inRange(hsv, (0, 20, 50), (25, 255, 255))
        # 皮肤色调范围 2: 红色系高端 (Hue 165~180, 绕过色环)
        mask_hsv2 = cv2.inRange(hsv, (165, 20, 50), (180, 255, 255))
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        # ---- YCrCb 色彩空间检测 ----
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        # Cr ∈ [135, 180], Cb ∈ [85, 135] — 经验值，覆盖多种肤色
        mask_ycrcb = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

        # ---- 双空间交集 — 提高精度 ----
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # ---- 形态学闭运算 — 填充小孔洞、去除孤立噪点 ----
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self._morph_kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self._morph_kernel)

        # ---- 计算网格级皮肤占比 ----
        h, w = skin_mask.shape
        crop_h = (h // self.grid_rows) * self.grid_rows
        crop_w = (w // self.grid_cols) * self.grid_cols
        cropped_mask = skin_mask[:crop_h, :crop_w].astype(np.float32) / 255.0
        bh = crop_h // self.grid_rows
        bw = crop_w // self.grid_cols
        # reshape + mean: 计算每个网格内皮肤像素占比
        skin_grid = cropped_mask.reshape(
            self.grid_rows, bh, self.grid_cols, bw
        ).mean(axis=(1, 3))  # (grid_rows, grid_cols), 值域 [0, 1]

        return skin_mask, skin_grid

    # -----------------------------------------------------------------------
    # 第 1 步: 网格化取样 — Grid-based Sampling
    # -----------------------------------------------------------------------
    def grid_sample(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        将输入帧划分为 grid_rows × grid_cols 个网格块，
        计算每个块内 R、G、B 三个通道的像素均值。

        利用 NumPy reshape + mean 实现向量化，无显式循环。

        Args:
            frame_bgr: BGR 格式帧, shape = (H, W, 3), dtype=uint8

        Returns:
            grid_rgb: shape = (grid_rows, grid_cols, 3), dtype=float32
                      每个元素为对应网格的 [R, G, B] 均值
        """
        h, w, _ = frame_bgr.shape

        # 裁剪到能被 grid 整除的尺寸（丢弃边缘少量像素）
        crop_h = (h // self.grid_rows) * self.grid_rows
        crop_w = (w // self.grid_cols) * self.grid_cols
        cropped = frame_bgr[:crop_h, :crop_w, :]

        # 每个网格块的像素高度 / 宽度
        bh = crop_h // self.grid_rows  # block height
        bw = crop_w // self.grid_cols  # block width

        # reshape 成 (grid_rows, bh, grid_cols, bw, 3)，然后沿 axis=(1,3) 取均值
        # 这是经典的 "block reduce" 向量化技巧
        reshaped = cropped.reshape(self.grid_rows, bh, self.grid_cols, bw, 3)
        grid_bgr = reshaped.mean(axis=(1, 3)).astype(np.float32)

        # OpenCV 读取的是 BGR，转换为 RGB 存储（与 POS 公式保持一致）
        grid_rgb = grid_bgr[:, :, ::-1]    # BGR → RGB
        return grid_rgb

    # -----------------------------------------------------------------------
    # 第 2 步: 推入环形缓冲区 — Ring Buffer
    # -----------------------------------------------------------------------
    def push_to_buffer(self, grid_rgb: np.ndarray) -> None:
        """
        将当前帧的网格 RGB 均值推入环形缓冲区。

        Args:
            grid_rgb: shape = (grid_rows, grid_cols, 3)
        """
        self.buffer[self.buf_index] = grid_rgb
        self.buf_index = (self.buf_index + 1) % self.window_size
        self.buf_count = min(self.buf_count + 1, self.window_size)

    # -----------------------------------------------------------------------
    # 第 3 步: 向量化 POS 算法 — Plane Orthogonal to Skin
    # -----------------------------------------------------------------------
    def compute_pos(self) -> np.ndarray | None:
        """
        对整个网格矩阵执行向量化 POS 算法。

        POS 论文参考:
          Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
          "Algorithmic Principles of Remote PPG."
          IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

        这里使用的是修改版 POS 公式（与用户需求一致的投影方向）:
          Xs = 3·Gn - 2·Bn
          Ys = 1.5·Rn + Gn - 1.5·Bn
          α  = σ(Xs) / σ(Ys)
          H  = Xs - α·Ys

        所有计算均通过 NumPy 高维广播完成，无显式循环。

        Returns:
            H: shape = (window_size, grid_rows, grid_cols)
               每个网格的时间序列脉搏波信号。
               如果缓冲区未满则返回 None。
        """
        if self.buf_count < self.window_size:
            return None  # 缓冲区尚未填满，无法计算

        # 将环形缓冲区按时间顺序排列
        # 当 buf_index=k 时，最老的帧在位置 k，最新的在 k-1
        ordered = np.roll(self.buffer, -self.buf_index, axis=0)
        # ordered.shape = (window_size, grid_rows, grid_cols, 3)
        # 通道索引:  [..., 0]=R,  [..., 1]=G,  [..., 2]=B

        # ------------------------------------------------------------------
        # Step 1: 时域归一化 (Temporal Normalization)
        #   μ = mean(C, axis=time)          → shape (grid_rows, grid_cols, 3)
        #   Cn = C / μ                       → shape (window_size, grid_rows, grid_cols, 3)
        #
        # 直觉: 消除肤色和光照的绝对值差异，只保留相对变化（脉动分量）
        # ------------------------------------------------------------------
        mu = ordered.mean(axis=0, keepdims=True)           # (1, R, C, 3)
        mu = np.where(mu < 1e-6, 1e-6, mu)                # 防除零
        Cn = ordered / mu                                  # (T, R, C, 3) 广播除法

        # 分离归一化后的三个通道
        Rn = Cn[:, :, :, 0]   # (T, R, C) — Red   归一化
        Gn = Cn[:, :, :, 1]   # (T, R, C) — Green 归一化
        Bn = Cn[:, :, :, 2]   # (T, R, C) — Blue  归一化

        # ------------------------------------------------------------------
        # Step 2: 正交投影 (Orthogonal Projection)
        #   Xs = 3·Gn - 2·Bn
        #   Ys = 1.5·Rn + Gn - 1.5·Bn
        #
        # 直觉: 这两个线性组合将 RGB 信号投影到一个与肤色正交的平面上，
        #        分离出脉搏波引起的色彩变化，同时抑制运动伪影。
        # ------------------------------------------------------------------
        Xs = 3.0 * Gn - 2.0 * Bn            # (T, R, C)
        Ys = 1.5 * Rn + Gn - 1.5 * Bn       # (T, R, C)

        # ------------------------------------------------------------------
        # Step 3: 动态 Alpha 校正 (Adaptive Alpha Tuning)
        #   α = σ(Xs) / σ(Ys)      对每个网格独立计算
        #
        # 直觉: Xs 和 Ys 中都含有脉搏信号和噪声，但比例不同。
        #        α 的作用是自适应地缩放 Ys，使得减法 Xs - α·Ys 能
        #        最大程度保留脉搏信号、消除公共噪声。
        # ------------------------------------------------------------------
        sigma_xs = Xs.std(axis=0)            # (R, C) — 每个网格沿时间轴的标准差
        sigma_ys = Ys.std(axis=0)            # (R, C)
        sigma_ys = np.where(sigma_ys < 1e-8, 1e-8, sigma_ys)  # 防除零
        alpha = sigma_xs / sigma_ys          # (R, C)

        # ------------------------------------------------------------------
        # Step 4: 信号合成 (Pulse Signal Synthesis)
        #   H = Xs - α · Ys
        #
        # alpha 需要广播到时间轴: (R,C) → (1,R,C) 与 (T,R,C) 运算
        # 直觉: 最终输出 H 是逐网格的血容量脉动信号的最优估计。
        # ------------------------------------------------------------------
        H = Xs - alpha[np.newaxis, :, :] * Ys  # (T, R, C)

        return H  # shape = (window_size, grid_rows, grid_cols)

    # -----------------------------------------------------------------------
    # 第 4 步: 带通滤波 + 能量计算
    # -----------------------------------------------------------------------
    def compute_energy(self, H: np.ndarray) -> np.ndarray:
        """
        对 POS 输出信号进行带通滤波 , 然后计算每个网格的信号能量（标准差）。

        带通滤波用于提取心率范围 (0.7 ~ 3.0 Hz) 的成分，滤除
        呼吸运动（< 0.7 Hz）和高频噪声（> 3.0 Hz）。

        能量 = std(H_filtered) , 代表该网格的"血流脉动强度"。

        Args:
            H: shape = (window_size, grid_rows, grid_cols)

        Returns:
            energy: shape = (grid_rows, grid_cols), dtype=float32
                    归一化到 [0, 255] 的能量值
        """
        T, R, C = H.shape

        # sosfiltfilt 要求沿 axis=0 滤波 (时间轴)
        # 将 (T, R, C) reshape 为 (T, R*C) 以便一次性处理所有网格
        H_flat = H.reshape(T, -1)

        # 零均值化 — 去除直流分量，提高滤波稳定性
        H_flat = H_flat - H_flat.mean(axis=0, keepdims=True)

        # 应用零相位带通滤波 (sosfiltfilt = forward + backward, 无相位畸变)
        H_filtered = sosfiltfilt(self.sos, H_flat, axis=0)

        # 能量 = 各网格时间序列的标准差
        energy = H_filtered.std(axis=0)    # (R*C,)
        energy = energy.reshape(R, C)      # (R, C)

        return energy

    # -----------------------------------------------------------------------
    # 第 5 步: 热力图渲染（纯热像图输出，带皮肤掩膜加权）
    # -----------------------------------------------------------------------
    def render_heatmap(
        self,
        energy: np.ndarray,
        skin_grid: np.ndarray,
        output_h: int,
        output_w: int,
    ) -> np.ndarray:
        """
        将网格能量图渲染为纯伪彩色热像图。

        关键抗伪影措施:
          1. 皮肤掩膜加权: 边缘混合网格（皮肤占比低）的能量被压制，
             避免头部移动时边界像素的"假高能量"覆盖真实脉搏信号。
          2. 百分位归一化: 用 2%~98% 百分位代替 min/max，
             抵抗运动伪影造成的异常值。
          3. 空间平滑: 对能量图做高斯模糊，使热力图更连续自然。

        输出:
          - 无信号 / 弱信号 → 蓝色（冷色）
          - 强信号 → 红色（暖色）

        Args:
            energy:    网格能量图, shape = (grid_rows, grid_cols)
            skin_grid: 网格级皮肤占比, shape = (grid_rows, grid_cols), 值域 [0,1]
            output_h:  输出图像高度（像素）
            output_w:  输出图像宽度（像素）

        Returns:
            heatmap: BGR 热像图, shape = (output_h, output_w, 3)
        """
        # ---- 皮肤掩膜加权 ----
        # 边缘网格（皮肤占比 < skin_thresh）的能量被置零
        # 中间网格按占比加权，纯皮肤网格权重 = 1.0
        skin_weight = np.where(
            skin_grid >= self.skin_thresh, skin_grid, 0.0
        )  # (R, C)
        energy_weighted = energy * skin_weight

        # ---- 空间高斯模糊 ----
        # 在网格级做 3×3 高斯平滑，消除单格噪声，使热力图更连续
        energy_weighted = cv2.GaussianBlur(
            energy_weighted.astype(np.float32), (5, 5), sigmaX=1.5
        )

        # ---- EMA 时域平滑: smoothed = α * current + (1-α) * previous ----
        if self.energy_smooth is None:
            self.energy_smooth = energy_weighted.copy()
        else:
            self.energy_smooth = (
                self.ema_alpha * energy_weighted
                + (1.0 - self.ema_alpha) * self.energy_smooth
            )

        # ---- 百分位归一化 (抗异常值) ----
        # 仅基于皮肤区域的能量值计算归一化范围
        e = self.energy_smooth
        skin_vals = e[skin_weight > 0]  # 只取皮肤区域的值

        if skin_vals.size < 10 or skin_vals.max() < 1e-8:
            # 无足够皮肤区域 → 全蓝
            norm = np.zeros_like(e, dtype=np.uint8)
        else:
            # 使用 2% ~ 98% 百分位，避免边缘伪影占据动态范围
            p_low = np.percentile(skin_vals, 2)
            p_high = np.percentile(skin_vals, 98)
            if p_high - p_low < 1e-8:
                p_high = p_low + 1e-8
            clipped = np.clip(e, p_low, p_high)
            norm = ((clipped - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)

        # 应用 COLORMAP_TURBO 伪彩色映射
        # TURBO: 0 → 深蓝,  128 → 青/绿,  255 → 红
        heatmap_small = cv2.applyColorMap(norm, COLORMAP)  # (R, C, 3) BGR

        # 双线性插值放大到输出尺寸
        heatmap_full = cv2.resize(
            heatmap_small, (output_w, output_h), interpolation=cv2.INTER_LINEAR
        )

        return heatmap_full

    # -----------------------------------------------------------------------
    # 完整管线：一帧端到端处理（支持进阶扫描模式）
    # -----------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        端到端处理单帧，返回纯热像图。

        扫描模式逻辑:
          - 未扫描: 显示当前已确认的血流图（静态）
          - 扫描中: 持续计算能量，超迎阈值的网格被永久锁定为"确认"
          - 已确认的网格显示红色，未确认的显示蓝色

        Args:
            frame_bgr: 原始 BGR 帧

        Returns:
            output: BGR 热像图
        """
        # -----------------------------------------------------------------------
        # 前处理: 镜像翻转 (让画面像照镜子一样自然)
        # -----------------------------------------------------------------------
        frame_bgr = cv2.flip(frame_bgr, 1)

        # 降分辨率
        frame_resized = cv2.resize(
            frame_bgr, (CAPTURE_WIDTH, CAPTURE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # Step 0: 皮肤检测（网格级，用于加权能量图）
        _, skin_grid = self.detect_skin(frame_resized)

        # Step 1: 网格化取样
        grid_rgb = self.grid_sample(frame_resized)

        # Step 2: 推入缓冲区（始终推入，保持缓冲区新鲜）
        self.push_to_buffer(grid_rgb)

        # Step 3: POS 计算
        H = self.compute_pos()

        if H is None:
            # 缓冲区未满 — 显示蓝色底图 + 进度条
            output = np.full(
                (CAPTURE_HEIGHT, CAPTURE_WIDTH, 3), (186, 59, 48), dtype=np.uint8
            )
            progress = self.buf_count / self.window_size
            bar_w = int(200 * progress)
            cv2.rectangle(output, (20, 20), (220, 50), (50, 50, 50), -1)
            cv2.rectangle(output, (20, 20), (20 + bar_w, 50), (0, 200, 255), -1)
            cv2.putText(
                output,
                f"Buffering... {self.buf_count}/{self.window_size}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )
            return output

        # Step 4: 能量计算
        energy = self.compute_energy(H)

        # ---- 扫描模式: 进阶确认 ----
        if self.scanning:
            self.scan_frame_count += 1

            # 皮肤加权后的能量
            skin_weight = np.where(
                skin_grid >= self.skin_thresh, skin_grid, 0.0
            )
            energy_weighted = energy * skin_weight

            # 空间平滑
            energy_weighted = cv2.GaussianBlur(
                energy_weighted.astype(np.float32), (5, 5), sigmaX=1.5
            )

            # 计算自适应确认阈值：皮肤区域能量的第 N 百分位
            skin_vals = energy_weighted[skin_weight > 0]
            if skin_vals.size >= 10:
                threshold = np.percentile(skin_vals, self.confirm_percentile)

                # 未确认的网格中，能量超迎阈值的 → 永久锁定
                new_confirm = (
                    (energy_weighted > threshold)
                    & (self.confirmed_map == 0)   # 仅更新未确认的
                    & (skin_weight > 0)            # 必须是皮肤区域
                )
                if new_confirm.any():
                    # 归一化到 [0.3, 1.0] 范围（确保确认的网格至少显示暖色）
                    e_max = skin_vals.max()
                    if e_max > 1e-8:
                        norm_energy = np.clip(
                            energy_weighted[new_confirm] / e_max, 0.3, 1.0
                        )
                    else:
                        norm_energy = 0.5
                    self.confirmed_map[new_confirm] = norm_energy

        # Step 5: 渲染热像图
        # 已确认的网格显示其锁定能量值，未确认的显示 0（蓝色）
        display_energy = (self.confirmed_map * 255.0).astype(np.uint8)

        # 如果正在扫描且还有未确认的网格，在未确认区域显示弱实时能量（让用户看到扫描进度）
        if self.scanning:
            skin_weight_now = np.where(
                skin_grid >= self.skin_thresh, skin_grid, 0.0
            )
            energy_live = energy * skin_weight_now
            energy_live = cv2.GaussianBlur(
                energy_live.astype(np.float32), (5, 5), sigmaX=1.5
            )
            live_vals = energy_live[skin_weight_now > 0]
            if live_vals.size >= 10 and live_vals.max() > 1e-8:
                live_norm = np.clip(energy_live / live_vals.max(), 0, 1.0)
                # 未确认区域显示弱实时能量 (乘以 0.3 使其比确认区域暗淡)
                unconfirmed = self.confirmed_map == 0
                display_energy[unconfirmed] = (live_norm[unconfirmed] * 0.3 * 255).astype(np.uint8)

        # 应用 COLORMAP_TURBO
        heatmap_small = cv2.applyColorMap(display_energy, COLORMAP)

        # 双线性插值放大
        output = cv2.resize(
            heatmap_small, (CAPTURE_WIDTH, CAPTURE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        return output


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    主循环: 打开 Mac 前置摄像头，实时处理并显示血流灌注热力图。

    控制键:
      空格键  — 开始 / 停止扫描
      R 键    — 重置已确认区域
      Q / ESC — 退出
    """
    print("=" * 60)
    print("  rPPG 实时血流灌注热力图  (POS Algorithm)")
    print("  空格 = 开始/停止扫描 | R = 重置 | Q/ESC = 退出")
    print("=" * 60)

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，请检查权限设置。")
        return

    # （尝试）设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # 实际帧率（摄像头可能无法满足请求值）
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    print(f"[INFO] 摄像头帧率: {actual_fps:.1f} FPS")
    print(f"[INFO] 网格分辨率: {GRID_ROWS}×{GRID_COLS}")
    print(f"[INFO] 时间窗口: {WINDOW_SIZE} 帧 ≈ {WINDOW_SIZE / actual_fps:.1f} 秒")

    # 初始化处理器
    processor = RPPGProcessor(fps=actual_fps)

    # 帧率计算用
    fps_counter = 0
    fps_timer = time.time()
    display_fps = 0.0

    window_name = "rPPG Blood Perfusion Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败，跳过...")
            continue

        # 端到端处理
        output = processor.process_frame(frame)

        # 计算帧率
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        # ---- 状态栏 (OSD) ----
        h_out, w_out = output.shape[:2]

        # 右上角 FPS
        cv2.putText(
            output, f"FPS: {display_fps:.1f}",
            (w_out - 140, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
        )

        # 左上角: 扫描状态
        if processor.scanning:
            confirmed_pct = (processor.confirmed_map > 0).sum() / (
                processor.grid_rows * processor.grid_cols
            ) * 100
            status_text = f"SCANNING... ({confirmed_pct:.0f}% confirmed)"
            status_color = (0, 200, 255)  # 橙色
            # 扫描动画: 闪烁圆点
            if int(time.time() * 3) % 2 == 0:
                cv2.circle(output, (15, 25), 6, (0, 0, 255), -1)
        else:
            confirmed_pct = (processor.confirmed_map > 0).sum() / (
                processor.grid_rows * processor.grid_cols
            ) * 100
            if confirmed_pct > 0:
                status_text = f"STOPPED ({confirmed_pct:.0f}% confirmed)"
            else:
                status_text = "READY - Press SPACE to scan"
            status_color = (200, 200, 200)  # 灰色

        cv2.putText(
            output, status_text,
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1, cv2.LINE_AA,
        )

        # 底部提示
        cv2.putText(
            output, "SPACE=Scan  R=Reset  Q=Quit",
            (w_out // 2 - 140, h_out - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
        )

        cv2.imshow(window_name, output)

        # ---- 键盘控制 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:     # Q / ESC = 退出
            break
        elif key == ord(' '):                 # 空格 = 开始/停止
            if processor.scanning:
                processor.stop_scan()
            else:
                processor.start_scan()
        elif key == ord('r') or key == ord('R'):  # R = 重置
            processor.reset_scan()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 程序已退出。")


if __name__ == "__main__":
    main()
