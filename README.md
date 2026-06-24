# 🚗 基于 YOLOv12 的实时车辆检测与分类系统

<p>
  <img src="https://img.shields.io/badge/YOLOv12-Ultralytics-orange?style=flat-square" alt="YOLOv12">
  <img src="https://img.shields.io/badge/Gradio-3.33.1-green?style=flat-square" alt="Gradio">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat-square" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License">
</p>

> 基于 YOLOv12 深度学习模型与 Gradio 构建的轻量级车辆检测与分类平台，支持图像上传、实时检测与结果可视化。

---

## 📌 项目简介

本系统基于 **YOLOv12** 目标检测模型，实现对图像中**车辆目标**的实时检测与分类，并通过 **Gradio 框架**构建简洁直观的 Web 界面，方便用户上传图片并查看检测结果。

适用于智能交通监控、车辆统计、自动驾驶辅助等应用场景。

---

## 🧠 技术栈

| 模块 | 技术选型 |
|------|---------|
| 深度学习模型 | YOLOv12（Ultralytics） |
| 后端框架 | Gradio |
| 图像处理 | OpenCV |
| 依赖管理 | PyTorch |
| 运行环境 | GPU / CPU 均可 |

---

## 📁 项目结构

```
VehicleDetection_YOLOv12/
├── app.py              # Gradio Web 应用入口
├── detect.py           # 命令行检测脚本
├── exp15.zip           # 训练好的模型权重压缩包
│   └── exp15/weights/best.pt   # YOLOv12 最佳模型权重
├── requirement.txt     # Python 依赖
└── README.md
```

---

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/fzk888/VehicleDetection_YOLOv12.git
cd VehicleDetection_YOLOv12
```

### 2. 安装依赖

```bash
pip install -r requirement.txt
```

> 如果 PyTorch 未安装，推荐使用 GPU 版本（需要 CUDA）：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. 解压模型文件

```bash
unzip exp15.zip
```

### 4. 启动 Web 服务

```bash
python app.py
```

启动后，在浏览器打开 `http://127.0.0.1:7860/`，上传车辆图片即可获得检测结果。

### 5. 命令行检测（可选）

```bash
python detect.py
# 按提示输入图片路径即可完成检测
```

---

## 🏁 项目特色

- ✅ **YOLOv12 前沿模型** — 最新 YOLO 架构，兼顾速度与精度
- ✅ **Gradio 友好界面** — 上传即检测，无需复杂配置
- ✅ **多车型分类** — 支持多种车辆类型的检测与分类
- ✅ **轻量级部署** — 支持 GPU 加速，也可在 CPU 上运行
- ✅ **开箱即用** — 提供预训练模型，无需从零训练

---

## 🔗 相关资源

- **YOLOv12**：[Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics)

---

## 📄 开源协议

MIT License
