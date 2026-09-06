```markdown
# NR-VQA 视频质量评估系统 - Demo 使用说明

## 📖 项目简介

本项目是一个基于深度学习的**无参考视频质量评估（NR-VQA）**系统，聚焦于**复杂场景下的视频时域一致性评价**，能够自动检测视频中的**闪烁（flicker）、帧冻结卡顿、时域噪声**等典型失真。

本 Demo 提供了完整的 Web 界面，方便快速演示和测试模型效果。

---

## 🏗️ 模型架构

```
单帧处理流程：
┌─────────────────────────────────────────────────────────────┐
│  输入视频帧 (224×224×3)                                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────┐   │
│  │  ResNet-50 多尺度    │    │  ViT-B/16 全局分支      │   │
│  │  卷积分支            │    │  (Transformer Encoder)  │   │
│  │  layer1~4 输出       │    │  Token 统计池化         │   │
│  │  空间统计池化 + GAP  │    │  2304 维               │   │
│  │  13568 维           │    │                         │   │
│  └─────────────────────┘    └─────────────────────────┘   │
│           ↓                            ↓                   │
│           └──────────┬─────────────────┘                   │
│                      ↓                                     │
│           拼接：15872 维单帧向量                           │
├─────────────────────────────────────────────────────────────┤
│  时间维聚合（T 帧平均） → 视频级特征 (15872 维)           │
├─────────────────────────────────────────────────────────────┤
│  回归头：Linear(15872→512) + ReLU + Linear(512→1)        │
├─────────────────────────────────────────────────────────────┤
│  输出：MOS 分数 (1-5)                                      │
└─────────────────────────────────────────────────────────────┘
```

### 特征维度说明

| 组件 | 输出维度 | 说明 |
|------|----------|------|
| ResNet-50 layer1-4 统计池化 | 3×(256+512+1024+2048) = 11520 | 每层 mean/max/std |
| ResNet-50 layer4 GAP | 2048 | 全局平均池化 |
| ViT-B/16 Token 统计池化 | 3×768 = 2304 | token 维 mean/max/std |
| **单帧特征拼接** | **15872** | 11520 + 2048 + 2304 |
| **视频级特征** | **15872** | 时间维平均 |
| **回归头输出** | **1** | MOS 分数 |

---

## 📁 目录结构

```
video-quality-assessment/
├── demo/                              # Demo 模块（本次新增）
│   ├── README.md                      # 本文档
│   ├── start_demo.py                  # Linux/Mac 启动脚本
│   ├── start_demo.bat                 # Windows 启动脚本
│   │
│   ├── backend/                       # 后端 API 服务
│   │   ├── __init__.py
│   │   ├── config.py                  # 配置文件（模型路径、端口等）
│   │   ├── inference.py               # 推理引擎（加载模型、预测）
│   │   ├── main.py                    # FastAPI 主程序
│   │   └── schemas.py                 # Pydantic 数据模型
│   │
│   ├── frontend/                      # 前端页面
│   │   ├── index.html                 # 主页面
│   │   ├── static/
│   │   │   ├── css/
│   │   │   │   └── style.css          # 样式文件
│   │   │   └── js/
│   │   │       └── app.js             # 前端逻辑
│   │   └── assets/                    # 静态资源（图片、示例视频等）
│   │
│   └── tests/                         # 测试脚本
│       ├── __init__.py
│       ├── test_load.py               # 测试模型加载
│       ├── test_inference.py          # 测试推理引擎
│       └── test_api.py                # 测试 API 接口
│
├── vqa/                               # 核心库（原有）
│   ├── config.py
│   ├── sampling.py                    # 帧抽取与预处理
│   ├── dataset.py
│   ├── models.py                      # 双分支特征提取器 + 回归头
│   ├── metrics.py
│   ├── train_utils.py
│   └── semisup.py
│
├── scripts/                           # 训练脚本（原有）
│   ├── make_flicker_dataset.py
│   ├── train_baseline.py
│   ├── train_semisup.py
│   └── predict.py
│
├── data/                              # 数据目录（不入库）
│   └── synthetic/
│       ├── videos/                    # 视频文件
│       └── labels.txt                 # 标注文件
│
├── runs/                              # 模型权重与日志（不入库）
│   └── baseline/
│       └── model_123.pt               # ⚠️ 当前为假权重（仅用于演示）
│
└── requirements.txt                   # Python 依赖
```

---

## ⚠️ 重要提示：关于模型权重

当前 Demo 使用create_fake_model.py默认加载的 `runs/baseline/model_123.pt` 是一个**假权重文件**（随机初始化），**仅用于 UI 界面测试**，其输出的 MOS 分数**没有实际参考价值**。

### 替换为真实训练权重

要使用真实训练好的模型，只需修改 **1 个文件**：

**编辑 `demo/backend/config.py`**，将 `MODEL_PATHS` 中的路径改为你的真实权重文件：

```python
# demo/backend/config.py
class DemoConfig:
    # 模型路径（按优先级顺序查找）
    MODEL_PATHS = [
        Path("runs/baseline/model_best.pt"),  # ← 改成你的真实权重文件路径
        # Path("runs/baseline/model.pt"),     # 备选
    ]
```

### 权重文件说明

| 文件名 | 说明 | 推荐使用 |
|--------|------|----------|
| `model_best.pt` | 验证 loss 最小时的完整模型 | ✅ **推荐** |
| `model.pt` | 最终 epoch 的完整模型 | 备选 |
| `extractor_best.pt` + `head_best.pt` | 分离的权重 | 需要先合并（见下方说明） |

### 如果只有分离的权重文件

如果 `runs/baseline/` 目录下只有 `extractor_best.pt` 和 `head_best.pt`，需要先合并：

```python
# merge_weights.py
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vqa.models import VQAModel

extractor_path = Path("runs/baseline/extractor_best.pt")
head_path = Path("runs/baseline/head_best.pt")

if extractor_path.exists() and head_path.exists():
    model = VQAModel(T=8)
    model.extractor.load_state_dict(torch.load(extractor_path))
    model.head.load_state_dict(torch.load(head_path))
    torch.save(model, Path("runs/baseline/model_best.pt"))
    print("✅ 已合并为完整模型: runs/baseline/model_best.pt")
```

---

## 🚀 快速开始

### 1. 环境准备

确保已安装 Python 3.8+ 并激活 conda 环境：

```bash
# 激活环境
conda activate opencv_env

# 安装 Demo 所需依赖
pip install fastapi uvicorn[standard] python-multipart aiofiles
```

### 2. 准备模型权重

> ⚠️ **注意**：如果使用假权重（默认），跳过此步骤。如果要使用真实权重，请先阅读上方【重要提示】替换权重文件。

### 3. 启动 Demo

#### Windows
```bash
demo\start_demo.bat
```

#### Linux / Mac
```bash
python demo/start_demo.py
```

#### 手动启动
```bash
python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8000
```

### 4. 访问界面

打开浏览器访问：**http://localhost:8000**

---

## 🎯 功能说明

### 前端功能

| 功能 | 说明 |
|------|------|
| **视频上传** | 支持拖拽或点击上传，支持批量处理 |
| **质量评估** | 自动计算 MOS 分数（1-5 分） |
| **结果展示** | 直观显示分数、评分条和质量等级 |
| **统计信息** | 显示平均分、最高分、最低分 |
| **模型信息** | 显示当前使用的模型和架构信息 |

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/health` | GET | 健康检查 |
| `/api/predict` | POST | 单个视频质量评估 |
| `/api/predict/batch` | POST | 批量视频评估 |
| `/api/model/info` | GET | 获取模型信息 |
| `/api/docs` | GET | API 文档（Swagger UI） |

---

## 🧪 测试

### 测试模型加载

```bash
python demo/tests/test_load.py
```

### 测试推理引擎

```bash
python demo/tests/test_inference.py
```

### 测试 API 接口

```bash
# 确保服务已启动
python demo/tests/test_api.py
```

---

## 🔧 配置说明

### 修改模型路径

编辑 `demo/backend/config.py`：

```python
class DemoConfig:
    # 模型路径（按优先级顺序查找）
    MODEL_PATHS = [
        Path("runs/baseline/model_best.pt"),  # ← 改成你的权重路径
    ]
    
    # API 配置
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # 抽帧数
    NUM_FRAMES = 8
    
    # 最大上传文件大小 (500MB)
    MAX_UPLOAD_SIZE = 500 * 1024 * 1024
```

### 修改前端端口

启动时指定端口：

```bash
python -m uvicorn demo.backend.main:app --host 0.0.0.0 --port 8080
```

---

## 📝 常见问题

### Q1: 启动时报错 `No module named uvicorn`

**解决方案**：安装缺失的依赖

```bash
pip install fastapi uvicorn[standard] python-multipart aiofiles
```

### Q2: 启动时报错 `Can't get attribute 'SerializableVQAModel'`

**解决方案**：模型文件包含自定义类，使用 `create_clean_model.py` 重新创建干净的 state_dict 模型

```bash
python create_clean_model.py
```

### Q3: 上传视频后没有反应

**解决方案**：
1. 检查浏览器控制台是否有错误信息（F12）
2. 确认模型已成功加载（查看终端输出）
3. 检查视频格式是否支持（MP4, AVI, MOV, MKV 等）

### Q4: 模型加载成功但推理报错

**解决方案**：
1. 确认 `vqa/sampling.py` 中存在 `read_video_frames` 函数
2. 检查视频文件是否可读
3. 运行测试脚本验证：`python demo/tests/test_load.py`

### Q5: runs 文件夹在 VSCode 中显示为灰色

**原因**：`runs/` 目录被 `.gitignore` 忽略，不提交到 Git 仓库

**说明**：这是正常现象，模型权重文件较大，不应提交到版本控制

### Q6: 如何确认当前使用的是假权重还是真实权重？

**方法**：
1. 查看终端启动日志，会显示加载的模型路径
2. 查看 `demo/backend/config.py` 中 `MODEL_PATHS` 的配置
3. 如果路径指向 `model_123.pt`，说明是假权重

---

## 📊 性能指标

| 指标 | 值 |
|------|-----|
| 模型参数 | ~120M |
| 单视频推理时间 | ~2-5 秒 (CPU) / ~0.5-1 秒 (GPU) |
| 抽帧数 | 8 帧 |
| 输入分辨率 | 224×224 |
| MOS 分数范围 | 1.0 - 5.0 |

---

## 🤝 协作开发

### 分支说明

- `main` / `master`：主分支，稳定版本
- `demo`：Demo 功能分支（当前）

### 提交代码

```bash
# 1. 切换到 demo 分支
git checkout demo

# 2. 添加修改
git add .

# 3. 提交
git commit -m "描述修改内容"

# 4. 推送到远程
git push origin demo
```

### 代码规范

- Python 代码遵循 PEP 8
- 前端代码遵循 ESLint 规范
- 提交信息使用中文，清晰描述改动

---

## 📄 许可证

本项目为课程设计项目，仅供学习交流使用。

---

## 👥 贡献者

- [黄彤] - Demo 开发与测试

---

## 📧 联系方式

如有问题，请联系项目负责人或在 Issue 中提出。

---

**祝使用愉快！🎉**
```