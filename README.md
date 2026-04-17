# rPPG-POS: 实时血流灌注热力图

基于 **POS (Plane-Orthogonal-to-Skin)** 算法的实时远程光电容积脉搏波 (rPPG) 可视化系统。

> **核心用途**: 医疗评估级"皮下血流灌注热力图" — 不做心率估计，而是将每个局部区域的血容量脉动能量映射为伪彩色热力图实时叠加显示。

## ✨ 特性

- 🚫 **无人脸检测** — 不依赖 dlib / MediaPipe，对任意皮肤区域（手臂、面部等）生效
- ⚡ **全向量化** — 核心 POS 数学运算完全基于 NumPy 高维广播，零 `for` 循环，140+ FPS
- 🎨 **TURBO 伪彩色** — 冷色（蓝）= 无血流，暖色（红）= 脉动强烈
- 🧱 **解耦架构** — `RPPGProcessor` 类中数据缓冲 / 数学运算 / 渲染完全分离，便于 C++ 移植
- 📊 **EMA 平滑** — 消除帧间闪烁，输出稳定平滑

## 🔬 处理管线

```
摄像头帧 → 降分辨率(640×480) → 网格化(40×40) → 环形缓冲区(90帧)
    ↓
POS 算法: 归一化 → 正交投影 → 动态α → 信号合成
    ↓
带通滤波(0.7~3.0Hz) → 标准差能量 → EMA平滑 → TURBO伪彩色 → 叠加显示
```

## 🚀 快速开始

```bash
# 安装依赖
pip3 install -r requirements_heatmap.txt

# 运行（需要摄像头权限）
python3 rppg_heatmap.py
```

- 前 ~3 秒为缓冲期（显示进度条）
- 按 `q` 或 `ESC` 退出

## 📐 POS 算法公式

| 步骤 | 公式 | 说明 |
|------|------|------|
| 归一化 | Cₙ = C / μ | 消除肤色/光照绝对值差异 |
| 正交投影 | Xs = 3Gₙ - 2Bₙ | 投影到与肤色正交的平面 |
| | Ys = 1.5Rₙ + Gₙ - 1.5Bₙ | |
| 动态Alpha | α = σ(Xs) / σ(Ys) | 自适应噪声抑制 |
| 信号合成 | H = Xs - α·Ys | 血容量脉动信号估计 |

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `rppg_heatmap.py` | 核心实现：RPPGProcessor 类 + 主循环 |
| `requirements_heatmap.txt` | 依赖清单 (numpy, opencv, scipy) |
| `pos_face_seg.py` | 旧版 POS 实现（基于 dlib 人脸检测，仅供参考） |

## 📚 参考文献

Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
*"Algorithmic Principles of Remote PPG."*
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
