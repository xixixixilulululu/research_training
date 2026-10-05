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

## 四、Training Task 6–8：BUSI 图像分割（《Research Training Instructions (3)》）

| 要求 | 状态 | 备注 |
|---|---|---|
| 建三个子文件夹，每个一个 notebook + 一个 slurm | ✅ | `task6_unet/`、`task6_nnunet/`、`task6_deeplabv3plus/` |
| 用 train.xlsx / test.xlsx 划分，不重新随机划分 | ✅ | 517 / 130，代码里断言两边不重叠 |
| 检查 mask 命名和对应关系，多 mask 要合并 | ✅ | 文件名与原图完全相同、每张图一个 mask；仍保留 `_mask*` 的 OR 合并逻辑，并在全数据上核对过 |
| 输出肿瘤区域的二值 mask | ✅ | 每个文件夹 `predictions/*.png`，原图尺寸 |
| nnU-Net：数据格式转换（目录结构 + dataset.json）、训练、预测命令 | ✅ | notebook 第 5–11 节 |
| 测试集上算 Dice、IoU、HD95，均值 + 逐图 `per_image_metrics.csv`，三个模型顺序一致 | ✅ | Task 8 notebook 再次断言了顺序一致 |
| 预测为空时 HD95 的统一规则，写清楚并打印空预测数量 | ✅ | HD95 = 原图对角线；空预测 U-Net 5 / nnU-Net 1 / DeepLabV3+ 0 |
| notebook 结尾逐张显示 原图 / 真实 mask / 预测 mask / 叠加图 | ✅ | 130 张全部显示 |
| 模型权重和预测 mask 存在各自文件夹 | ✅ | `outputs/*.pt`；nnU-Net 在 `nnUNet_results/.../checkpoint_final.pth` |
| 每个代码单元格前有 Markdown 说明，代码有注释 | ✅ | |
| 公平比较：统一输入尺寸、预处理、种子、batch、优化器、学习率、epoch、增强；开头集中的配置单元格 | ✅ | 三个 notebook 共享超参数块逐字一致；差异（DeepLabV3+ 预训练、nnU-Net 保留默认设置）写明了原因 |
| Task 7：同一张测试图三个模型并排，挑出欠分割 / 过分割 / 漏检 / 边界不准的例子 | ✅ | `task7_visual_comparison/`，图在 `figures/` |
| 统计检验：均值 ± 标准差、配对 t 检验 + Holm、Wilcoxon、结果表、Markdown 解释 | ✅ | `task8_statistics/statistical_analysis.ipynb` |
| Task 8：项目根目录 `segmentation_comparison.md` | ✅ | 数字全部来自实际运行结果 |
| **理解** Dice / IoU / HD95、统计检验的含义，自己能讲出结果 | ⛔ **未完成** | 和前面的任务一样，"理解"这部分需要你自己拿着 Task 7 的示意图和 Task 8 的 Markdown 解释去消化 |

## 结果质量标准

这份清单核对的是"PDF 要求的事有没有做"；至于"做出来的结果算不算好"，另开了一份
`IDEAL_RESULT.md`，列了 6 条理想结果的标准（loss 曲线形状、指标口径、K 折标准差、方法论干净程度、
概率校准、数据量天花板），每条都标了现在做到了没有，供后续继续调参时参考。

## 总结

- **代码 / 环境 / 训练全部跑通**：数据、notebook、Slurm 提交、正式训练（第 1 次）都已完成，有实际结果。
- **真正没完成的，是 PDF 里明确写的"你自己去理解"的部分**（学概念、理解每行代码、观察并解读 loss 曲线）。这些没法由我代劳，需要你自己拿着已经跑出来的结果（比如 loss 曲线里的过拟合现象）去问 AI chatbot、结合课程 slides 弄懂。
- PDF 标题带 "(1)"，可能还有后续文档，建议跟老师确认。
- 《Research Training Instructions (3)》的 Task 6–8（分割 + 统计检验 + 对比文档）代码、训练和文档都已完成，见第四节。
