# Stanford Machine Learning (Andrew Ng) - Study Repository
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

这是一个用于记录和展示吴恩达（Andrew Ng）机器学习课程学习过程的开源项目。本仓库内容主要基于斯坦福大学（Stanford University）的经典课程 CS229。包含课程笔记、实验（Labs）、作业（Assignments）以及自主实现的算法示例。

This is an open-source repository for documenting and showcasing the learning process of Andrew Ng's Machine Learning course, primarily based on Stanford CS229. It includes course notes, labs, assignments, and self-implemented algorithm examples.

---

##  课程概况 / Course Overview (Stanford CS229)

>  **学习建议**: 本项目中的实验实现（[examples/](file:///d:/Github/MachineLearning_NG/examples)）可与 Stanford CS229 的理论内容结合学习。CS229 侧重数学推导，而本项目通过代码实践来巩固这些理论。
>  **Study Tip**: The implementations in [examples/](file:///d:/Github/MachineLearning_NG/examples) can be combined with Stanford CS229 theoretical content. CS229 focuses on mathematical derivations, while this project solidifies them through practice.

| 项目 / Item | 内容 / Details |
| :--- | :--- |
| **提供方 / Provider** | 斯坦福大学 (Stanford University) |
| **难度 / Difficulty** | 🌟🌟🌟🌟 (研究生级别 ) 
| **先决条件 / Prerequisites** | 高等数学、概率论、Python、扎实的数学基础 |
| **课时 / Duration** | 约 100 小时 (100 Hours) |

### 课程描述 / Course Description
这是吴恩达教授在斯坦福大学开设的经典机器学习课程（CS229）。它更侧重于机器学习背后的**数学理论**。如果你不满足于使用现成的工具，而是想了解算法的本质，或者渴望从事机器学习的理论研究，那么这门课程非常适合你。

###  学习资源 / Resources
- **课程网站**: [cs229.stanford.edu](http://cs229.stanford.edu/syllabus.html)
- **录像链接 (Bilibili)**: [CS229 机器学习 · 2018年 (中英字幕)](https://www.bilibili.com/video/BV1JE411w7Ub)
- **教材**: 无正式教材，但其**课堂笔记 (Lecture Notes)** 极其经典，涵盖了深度的数学推导。

---
## 补充笔记
##  项目结构 / Project Structure

>  **Note**: 该项目目前仍处于学习和总结阶段。
>  **Note**: This project is still in progress.

###  课堂笔记说明 / Study Notes Status
我的课堂笔记存放于 `notes/` 目录下，目前正在按章节进行深度总结和更新：
- **`01_Supervised_Machine_Learning.md`**: 涵盖监督学习基础（正在更新...）
- **`02_Advanced_Learning_Algorithms.md`**: 涵盖神经网络与高级算法（正在更新...）

```text
.
├──
labs_in_jupyter_notebook/                        # 课程实验与作业 / Course Labs & Assignments (Updated)
│   ├── Advanced_Learning_Algorithm/
│   │   ├── hw1/                 # 神经网络作业
│   │   └── lab1/                # 神经元与层实验
│   └── Supervised_Machine_Learning/
│       ├── hw2/                 # 线性回归作业
│       ├── hw3/                 # 逻辑回归作业
│       └── lab1/2/3/            # 监督学习基础实验
├── notes/                       # 核心概念笔记与资源 / Study Notes & Assets (Ongoing )
│   ├── assets/                  # 笔记中使用的图片资源
│   ├── 01_Supervised_Machine_Learning.md
│   └── 02_Advanced_Learning_Algorithms.md
├── examples/                    # 自主实现的代码示例 / Self-implemented Examples (Adding...)
│   ├── Linear.py                # 线性回归实现
│   ├── Logistic.py              # 逻辑回归实现
│   ├── SimpleNN.py              # 简单神经网络实现
│   └── regularized.py           # 正则化实现
└── setup/                       # 环境配置指南 / Environment Setup
    ├── requirements.txt
    └── 环境配置.md
```

##  快速开始 / Quick Start

### 环境准备 / Prerequisites

确保你已经安装了 Python 3.8 或更高版本。

1. 克隆仓库 / Clone the repository:
   ```bash
   git clone https://github.com/your-username/MachineLearning_NG.git
   cd MachineLearning_NG
   ```

2. 安装依赖 / Install dependencies:
   ```bash
   pip install -r setup/requirements.txt
   ```

### 运行实验 / Running Labs

所有的实验都以 Jupyter Notebook (`.ipynb`) 形式提供，你可以直接在本地运行：
```bash
jupyter notebook
```

##  学习内容 / What's Inside

- **监督学习 (Supervised Learning)**: 线性回归、逻辑回归、梯度下降、正则化等。
- **高级学习算法 (Advanced Learning Algorithms)**: 神经网络、正向传播、反向传播、TensorFlow 实战等。
- **实战经验 (Practical Experience)**: 特征缩放、多变量回归、分类问题处理等。

##  自主实现示例 / Custom Implementations

在 `examples/` 目录下，我通过 Numpy 手动实现了核心算法逻辑，帮助深入理解数学原理：
- `Linear.py`: 包含损失函数计算、梯度下降等核心逻辑。
- `implement_the_forward_prop_in_numpy.py`: 神经网络前向传播的底层实现。

## 📄 开源协议 / License

本项目采用 [MIT License](LICENSE) 开源。仅供学习交流使用，请勿用于商业用途。

---
*如果你觉得这个项目对你有帮助，欢迎点个 Star! ⭐*

