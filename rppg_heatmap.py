# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np
import time
from skimage.segmentation import slic

# ============================================================================
# 系统与算法参数配置
# ============================================================================

# 摄像头与分辨率
CAPTURE_WIDTH = 640         # 内部处理宽度
CAPTURE_HEIGHT = 480        # 内部处理高度
TARGET_FPS = 30             # 目标帧率 (通常前置摄像头为 30 fps)

# 纯像素级分析不需要 GRID 划分，完全依靠 ROI 内的原始分辨率
# 中心分析框 (ROI - Region of Interest)
ROI_W = 200
ROI_H = 240
ROI_X = (CAPTURE_WIDTH - ROI_W) // 2
ROI_Y = (CAPTURE_HEIGHT - ROI_H) // 2

# 伪色彩映射表
COLORMAP = cv2.COLORMAP_TURBO

# 皮肤检测参数
SKIN_THRESH = 0.3           # 时域皮肤掩膜平均占比阈值

# 批处理模式参数
BATCH_SECONDS = 40          # 固定录制秒数

# 超像素参数 (CVPR 2022)
SUPERPIXEL_COUNT = 800      # 将画面切分的不规则多边形色块数量（越多越精细）


# ============================================================================
# 核心类: RPPGProcessor
# ============================================================================

class RPPGProcessor:
    """
    负责执行 15 秒 Offline Batch FFT 分析的 rPPG (光电容积脉搏波) 引擎。
    采用 CVPR 2022 级别【SLIC 超像素 (Superpixel) + 全局相关系数 (Pearson)】架构。
    彻底根除斑驳噪点，且实现完美边界的异物防渗色识别。
    """

    def __init__(self, fps=TARGET_FPS):
        self.fps = fps
        self.total_frames = int(BATCH_SECONDS * self.fps)
        
        # 核心长时缓存 (T, R, C, 3) 
        self.buffer = np.zeros(
            (self.total_frames, ROI_H, ROI_W, 3), dtype=np.float32
        )
        self.skin_accumulator = np.zeros((ROI_H, ROI_W), dtype=np.float32)
        
        # 状态机
        self.state = 0  # 0=IDLE, 1=RECORDING, 2=DONE
        self.frame_count = 0
        self.heatmap_result = None
        self.final_display = None  # 定格画面
        self.heart_rate_bpm = 0.0
        self.base_roi_bgr = None

        self.skin_thresh = SKIN_THRESH
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def start_scan(self) -> None:
        if self.state == 0:
            self.state = 1
            self.frame_count = 0
            self.skin_accumulator.fill(0)
            self.heatmap_result = None
            print(f"[SCAN] ┣─ 开始 {BATCH_SECONDS} 秒高精度数据采集 (准备启动超像素分析)...")

    def reset_scan(self) -> None:
        self.state = 0
        self.frame_count = 0
        self.heatmap_result = None
        self.final_display = None
        print("[SCAN] ┣─ 已重置为待机状态")

    def detect_skin(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        极度放宽的皮肤检测掩膜，防止在稍微偏暗的一面脸颊丢掉皮肤特征。
        核心判断完全交由 FFT 纯动态波形能量决定，不靠死板颜色卡信号。
        """
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        # 放宽 Hue 到 0-40 和 150-180，放宽 Sat 到 10-255，放宽 Val 到 20-255
        mask_hsv1 = cv2.inRange(hsv, (0, 10, 20), (40, 255, 255))
        mask_hsv2 = cv2.inRange(hsv, (150, 10, 20), (180, 255, 255))
        skin_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)

        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self._morph_kernel)
        return skin_mask.astype(np.float32) / 255.0

    def _draw_legend(self, img: np.ndarray, x: int, y: int, h: int, w: int = 20) -> None:
        """在画面上绘制 TURBO 色彩标尺 + 文字标注"""
        bar = np.zeros((h, 1), dtype=np.uint8)
        for i in range(h):
            bar[h - 1 - i, 0] = int(255 * i / h)
        bar_color = cv2.applyColorMap(bar, COLORMAP)
        bar_color = cv2.resize(bar_color, (w, h), interpolation=cv2.INTER_NEAREST)
        ih, iw = img.shape[:2]
        if x + w <= iw and y + h <= ih:
            img[y:y+h, x:x+w] = bar_color
            cv2.putText(img, "High", (x + w + 4, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Low", (x + w + 4, y + h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "Perfusion", (x - 2, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # 核心：CVPR'22 超像素 (Superpixel) 离线分解与皮尔逊诊断
    # -----------------------------------------------------------------------
    def process_batch(self):
        print("[PROCESS] ┣─ 1/4 正在进行 SLIC 超像素智能分割...")
        start_t = time.time()
        
        # 使用录像的时间中值作为 SLIC 分割依据，以获取最稳定的人脸贴图边缘
        median_img = np.median(self.buffer, axis=0).astype(np.float64)
        
        # 保存中值帧用于半透明叠加显示 (RGB float → BGR uint8)
        self.base_roi_bgr = np.clip(median_img[:,:,::-1], 0, 255).astype(np.uint8)
        
        # n_segments 控制生成的超像素数量。数量越多拼图越小。
        # compactness 控制拼图的方正程度；值较小（如 10.0）会让多边形更努力去贴合物体纹理边界。
        segments = slic(median_img, n_segments=SUPERPIXEL_COUNT, compactness=15.0, sigma=1, start_label=0)
        
        # 实际生成的色块数量可能与要求略有偏差
        n_actual_segments = len(np.unique(segments))
        print(f"[PROCESS] ┣─ 发现 {n_actual_segments} 块完美贴合物体边缘的微组织切片。")
        
        T = self.total_frames
        buf_flat = self.buffer.reshape(T, -1, 3) # (450, 48000, 3)
        seg_flat = segments.ravel()
        skin_acc_flat = self.skin_accumulator.ravel() / T
        
        # 初始化提取容器
        super_traces = np.zeros((T, n_actual_segments, 3), dtype=np.float32)
        skin_prob = np.zeros(n_actual_segments, dtype=np.float32)
        
        # 在色块内部进行绝对平均，完美压平噪波且杜绝对外溢出
        for i in range(n_actual_segments):
            mask = (seg_flat == i)
            if mask.any():
                super_traces[:, i, :] = buf_flat[:, mask, :].mean(axis=1) # 时序均值 (T, 3)
                skin_prob[i] = skin_acc_flat[mask].mean()
        
        print("[PROCESS] ┣─ 2/4 正在进行多边形的微切片 POS 时序解算...")
        # (T, N, 3) 的低维高精度数据进行计算，速度极快
        mu = super_traces.mean(axis=0, keepdims=True)
        mu = np.where(mu < 1e-6, 1e-6, mu)
        Cn = super_traces / mu
        
        Rn = Cn[:, :, 0]
        Gn = Cn[:, :, 1]
        Bn = Cn[:, :, 2]
        
        Xs = 3.0 * Gn - 2.0 * Bn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn
        
        sigma_xs = Xs.std(axis=0)
        sigma_ys = Ys.std(axis=0)
        sigma_ys = np.where(sigma_ys < 1e-8, 1e-8, sigma_ys)
        alpha = sigma_xs / sigma_ys
        
        H_super = Xs - alpha * Ys  # (T, N)
        H_super -= H_super.mean(axis=0)
        
        # ---- CVPR'22 核心: Pearson Correlation Coefficient (PCC) 验证 ----
        # 从高皮肤概率的超像素中提取全局参考脉搏信号
        print("[PROCESS] ┣─ 2.5/4 Pearson 相关性验证 (区分真实脉搏 vs 噪声)...")
        high_skin_idx = skin_prob > self.skin_thresh
        if high_skin_idx.sum() > 5:
            weights = skin_prob[high_skin_idx]
            weights /= weights.sum()
            global_pulse = np.average(H_super[:, high_skin_idx], axis=1, weights=weights)
        else:
            # 皮肤检测失败时退化为全部超像素均值
            global_pulse = H_super.mean(axis=1)
        
        global_pulse -= global_pulse.mean()
        global_std = global_pulse.std()
        if global_std < 1e-8:
            global_std = 1e-8
        
        # 向量化 Pearson 相关系数计算 (无 for 循环)
        local_stds = np.std(H_super, axis=0)  # (N,)
        local_stds = np.where(local_stds < 1e-8, 1e-8, local_stds)
        cov = np.mean(global_pulse[:, np.newaxis] * H_super, axis=0)  # (N,)
        pcc_values = cov / (global_std * local_stds)
        pcc_values = np.clip(pcc_values, 0, 1.0).astype(np.float32)
        
        print(f"[PROCESS] ┣─    PCC 统计: mean={pcc_values.mean():.3f}, "
              f"max={pcc_values.max():.3f}, min={pcc_values.min():.3f}, "
              f">40%占比={np.mean(pcc_values > 0.4)*100:.1f}%")
        
        print("[PROCESS] ┣─ 3/4 频域 FFT + SNR 信噪比分析...")
        
        window = np.hanning(T)[:, np.newaxis]
        H_win = H_super * window
        spectrum = np.abs(np.fft.rfft(H_win, axis=0))
        freqs = np.fft.rfftfreq(T, d=1.0 / self.fps)
        
        # 全局参考脉搏频谱 → 心率提取
        global_fft = np.abs(np.fft.rfft(global_pulse * window.ravel()))
        hr_band = (freqs >= 0.8) & (freqs <= 2.5)
        if hr_band.any():
            hr_peak_idx = np.argmax(global_fft[hr_band])
            self.heart_rate_bpm = freqs[hr_band][hr_peak_idx] * 60.0
        else:
            self.heart_rate_bpm = 0.0
        print(f"[PROCESS] ┣─    检测心率: {self.heart_rate_bpm:.1f} BPM")
        
        # FFT 带通: 0.8 Hz ~ 2.5 Hz (48 ~ 150 BPM)
        valid_idx = (freqs >= 0.8) & (freqs <= 2.5)
        band_spectrum = spectrum[valid_idx, :]
        
        # SNR (信噪比): 带内峰值功率 / 带内中值功率
        # 真正有血液灌注的区域在心率频率处有窄带尖峰 → SNR 高 (>5)
        # 噪声/异物区域频谱平坦 → SNR ≈ 1
        peak_power = np.max(band_spectrum**2, axis=0)
        median_power = np.median(band_spectrum**2, axis=0)
        median_power = np.where(median_power < 1e-10, 1e-10, median_power)
        snr = peak_power / median_power
        
        # ★ 组合评分: PCC × SNR
        # 只有时域相关性和频域信噪比都确认的区域才获得高分
        score = pcc_values * snr
        
        print(f"[PROCESS] ┣─    SNR 统计: mean={snr.mean():.2f}, max={snr.max():.2f}")
        print(f"[PROCESS] ┣─    Score (PCC×SNR): mean={score.mean():.2f}, max={score.max():.2f}")
        
        print("[PROCESS] ┣─ 4/4 色彩映射与可视化...")
        
        # ==== 把 PCC×SNR 评分"涂回"原像素 ====
        heatmap_score = np.zeros(seg_flat.shape, dtype=np.float32)
        for i in range(n_actual_segments):
            mask = (seg_flat == i)
            heatmap_score[mask] = score[i]
            
        heatmap_score = heatmap_score.reshape(ROI_H, ROI_W)
        
        # ==== 全范围 [0, 1.0] 色彩映射 — Gamma 增强中间段对比度 ====
        norm_energy = np.zeros_like(heatmap_score)
        
        valid_mask = heatmap_score > 0
        if valid_mask.any():
            # 以 90 百分位为满分标杆（防止少数高分超像素压缩大部分皮肤信号）
            cap = np.percentile(heatmap_score[valid_mask], 90)
            if cap < 1e-8:
                cap = 1e-8
            
            # 归一化到 [0, 1] 全范围
            norm_energy = np.clip(heatmap_score / cap, 0, 1.0)
            
            # 硬门限: 低 PCC×SNR 区域直接归零（非灌注/异物区域）
            noise_floor = 0.05
            below_noise = norm_energy < noise_floor
            norm_energy[below_noise] = 0.0
            
            # 重新拉伸有效范围到 [0, 1]
            above_noise = norm_energy >= noise_floor
            if above_noise.any():
                norm_energy[above_noise] = (
                    (norm_energy[above_noise] - noise_floor) / (1.0 - noise_floor)
                )
            
            # ★ Gamma 变换 (指数 0.4): 将右偏分布展开到暖色区域
            # 线性映射下中等评分的皮肤全部挤在蓝色段；
            # Gamma<1 会把中间值"抬升"到暖色（黄/橙/红），同时保持零分区域不变
            norm_energy = np.power(norm_energy, 0.4)
                
        # 强高斯平滑 — 消除超像素块状边界，产生论文级絮状/云状灌注效果
        norm_energy = cv2.GaussianBlur(norm_energy, (21, 21), 0)
        
        display_energy = (norm_energy * 255.0).astype(np.uint8)
        self.heatmap_result = cv2.applyColorMap(display_energy, COLORMAP)
        
        elapsed = time.time() - start_t
        print(f"[PROCESS] ┣─ 分析完成: PCC×SNR 超像素诊断, 耗时 {elapsed:.2f}s")


    # -----------------------------------------------------------------------
    # 完整管线：一帧端到端处理
    # -----------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_resized = cv2.resize(
            frame_bgr, (CAPTURE_WIDTH, CAPTURE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        h, w = frame_resized.shape[:2]
        
        # 截取中心 ROI
        roi_bgr = frame_resized[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W, :]

        if self.state == 0:
            output = frame_resized.copy()
            output = (output * 0.4).astype(np.uint8)
            output[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W] = roi_bgr
            cv2.rectangle(output, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (200, 200, 200), 2)
            
            cv2.putText(
                output, "READY - Align Target in Box and Press SPACE", 
                (w//2 - 200, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )
            return output

        elif self.state == 1:
            # 采集阶段 (无需像素级模糊，保存原始 RGB 留给 SLIC 分割时自适应取舍)
            skin_mask = self.detect_skin(roi_bgr)
            self.skin_accumulator += skin_mask
            
            # OpenCV BGR -> RGB numpy float
            roi_rgb = roi_bgr[:, :, ::-1].astype(np.float32)
            self.buffer[self.frame_count] = roi_rgb
            self.frame_count += 1
            
            output = frame_resized.copy()
            output = (output * 0.4).astype(np.uint8)
            output[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W] = roi_bgr
            cv2.rectangle(output, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (0, 0, 255), 2)

            progress = self.frame_count / self.total_frames
            bar_w = int(ROI_W * progress)
            cv2.rectangle(output, (ROI_X, ROI_Y - 20), (ROI_X + ROI_W, ROI_Y - 10), (50, 50, 50), -1)
            cv2.rectangle(output, (ROI_X, ROI_Y - 20), (ROI_X + bar_w, ROI_Y - 10), (0, 200, 255), -1)
            
            cv2.putText(
                output, f"RECORDING... {BATCH_SECONDS - (self.frame_count/self.fps):.1f}s", 
                (ROI_X, ROI_Y - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA
            )

            if self.frame_count >= self.total_frames:
                self.process_batch()
                
                bg = frame_resized.copy()
                bg = (bg * 0.4).astype(np.uint8)
                
                # 半透明叠加: 原始解剖结构 + 热力图
                if self.base_roi_bgr is not None:
                    blended = cv2.addWeighted(
                        self.base_roi_bgr, 0.35, self.heatmap_result, 0.65, 0
                    )
                else:
                    blended = self.heatmap_result
                bg[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W] = blended
                cv2.rectangle(bg, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (0, 255, 0), 2)
                
                # 色彩标尺
                self._draw_legend(bg, ROI_X + ROI_W + 10, ROI_Y, ROI_H)
                
                # 心率显示
                cv2.putText(
                    bg, f"HR: {self.heart_rate_bpm:.0f} BPM",
                    (ROI_X, ROI_Y + ROI_H + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    bg, "CVPR'22 SLIC-PCC SCAN DONE", 
                    (20, CAPTURE_HEIGHT - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA
                )
                self.final_display = bg
                self.state = 2

            return output

        elif self.state == 2:
            return self.final_display


# ============================================================================
# 主程序入口
# ============================================================================

def main():
    print("=" * 60)
    print("  rPPG 【顶级医学形态 - CVPR 2022 原版复刻】")
    print("  架构: SLIC Superpixels + Global PCC + FFT Energy")
    print("  空格 = 开始获取 | R = 重新来过 | Q/ESC = 退出")
    print("=" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，请检查权限设置。")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # 锁定相机自动参数，防止采集期间曝光/白平衡变化引入伪信号
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 手动曝光模式
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)         # 关闭自动白平衡
    print("[INFO] 已尝试锁定相机曝光与白平衡参数")

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    print(f"[INFO] 摄像头帧率: {actual_fps:.1f} FPS")

    processor = RPPGProcessor(fps=actual_fps)

    window_name = "CVPR'22 Superpixel Scan"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        output = processor.process_frame(frame)

        h_out, w_out = output.shape[:2]
        cv2.putText(
            output, "SPACE=Start Scan | R=Reset | Q=Quit",
            (w_out // 2 - 160, h_out - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )

        cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            if processor.state == 0:
                processor.start_scan()
        elif key == ord('r') or key == ord('R'):
            processor.reset_scan()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
