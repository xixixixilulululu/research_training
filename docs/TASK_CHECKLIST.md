# 任务完成情况对照表

对照《Research Training Instructions (1)》逐条核对，✅ = 已完成，⛔ = 未完成（且我没法替你做）。

## 一、通用建议（General tips）

| PDF 要求 | 状态 | 备注 |
|---|---|---|
| 判断 GPU server 类型，普通 GPU 用 `nohup`，HPC（如 DGX_Spark）用 Slurm | ✅ | 我们在 DGX Spark 上，全程用 Slurm 提交（`kean/train.slurm`、`run_task1.slurm`） |
| VPN 每 30~60 分钟会断，程序要不受影响 | ✅ | Slurm job 跑在计算节点上，跟 VPN/终端窗口无关，断了也没事 |
| 下载 BUSI 数据集并上传到服务器 | ✅ | 已上传到 `~/datasets/BUSI/` |
| 建一个 `datasets` 文件夹和一个 `projects` 文件夹分开放 | ✅ | `~/datasets/BUSI/`，`~/projects/research_training/` |

## 二、开始训练前的准备

| PDF 要求 | 状态 | 备注 |
|---|---|---|
| 用 AI chatbot 理解：.py vs .ipynb、Deep Learning、CV/分类/分割、data & label、train/val/test 划分 | ⛔ **未完成** | PDF 明确写的是"请你自己用 AI chatbot 学懂"，这是你的学习任务，我没法替你"理解" |
| 理解 CNN 相关概念：convolution、pooling、batch norm、activation function、FC layer | ⛔ **未完成** | 同上，需要你自己去问 AI 学 |
| 理解训练相关概念：forward/backward propagation、epoch、loss function、optimizer、learning rate、batch size、overfitting/underfitting | ⛔ **未完成** | 同上；不过第 1 次训练的实际结果里已经出现了一个真实的 overfitting 例子（见 README「跑过的记录」），可以拿它做例子去理解 |
| 看老师给的 CV 课程 slides | ⛔ **未完成** | 我这边看不到你有没有看，需要你自己确认 |
| 建 `research_training` 文件夹 | ✅ | `~/projects/research_training/` |
| 建 `Test.ipynb`，写一个简单 Python 程序并运行 | ✅ | 已创建并执行成功（见 `Test.ipynb`） |
| 学会写有文字说明 cell + 代码 cell 的、组织良好的 notebook | ✅（代码层面） | `Test.ipynb`、`BUSI_Classification.ipynb` 都是这个结构；但"学会"本身要靠你自己消化 |

## 三、Training Task 1：BUSI 图像分类

| PDF 要求 | 状态 | 备注 |
|---|---|---|
| 用 AI agent 帮忙写代码、注释、结果文档、分析 | ✅ | 就是我们做的这件事 |
| 用 ResNet50 做良性/恶性二分类，写成 notebook | ✅ | `BUSI_Classification.ipynb` |
| 输入图像 resize 到 224×224 | ✅ | |
| 用两个 xlsx 文件划分训练/测试集，训练集上训练、测试集上评估 | ✅ | 517 训练 / 130 测试，无重复；最新一版又从 517 条训练集里按分层抽样切出 train_sub(413)/val(104)，早停/选模型只看 val，test 130 条全程不参与训练调参，避免"偷看测试集" |
| 加载 ImageNet 预训练权重 | ✅ | `ResNet50_Weights.IMAGENET1K_V2` |
| 计算 accuracy、recall、F1 等标准分类指标 | ✅ | 结果：Accuracy 0.9077 / Precision 0.9247 / Recall 0.9451 / F1 0.9348 |
| 画每个 epoch 的 train/test loss 曲线 | ✅ | `outputs/run3_loss_curve.png`（第 1/3 次基线，train vs test）；从第 4 次开始逐 epoch 画的是 train vs **val**（`outputs/run{N}_loss_curve.png`），test 只在训练全部结束后单独评估一次（`outputs/run{N}_test_metrics.png`），避免逐 epoch 曲线里混进 test 数据 |
| **观察** loss 曲线，理解 underfitting/overfitting，据此判断合适的训练轮数 | ⛔ **未完成** | 图已经画出来了，而且已经出现了明显的过拟合信号（train loss→0.007，test loss 却在 0.3~0.5 波动），但"观察 + 理解 + 判断"这一步是 PDF 明确要求你自己做的，我不能替你下结论 |
| 测试集上逐张显示图片 + 真实标签 + 预测标签 | ✅ | notebook 第 11 节，预测错的图会标红 |
| 每个代码 cell 前面有文字说明，代码有必要注释 | ✅ | |
| **借助 AI，理解每一行代码和背后的知识** | ⛔ **未完成** | PDF 原话是"with the help of AI, understand"——这是要你自己去理解，不是我把代码跑出来就算数 |

## 总结

- **代码 / 环境 / 训练全部跑通**：数据、notebook、Slurm 提交、正式训练（第 1 次）都已完成，有实际结果。
- **真正没完成的，是 PDF 里明确写的"你自己去理解"的部分**（学概念、理解每行代码、观察并解读 loss 曲线）。这些没法由我代劳，需要你自己拿着已经跑出来的结果（比如 loss 曲线里的过拟合现象）去问 AI chatbot、结合课程 slides 弄懂。
- PDF 标题带 "(1)"，可能还有后续文档，建议跟老师确认。
