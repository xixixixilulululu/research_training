# research_training — BUSI 乳腺超声图像分类练习

按《Research Training Instructions (1)》的 Training Task 1 搭的项目：用 ResNet50（ImageNet 预训练）对 BUSI
超声图像做良性/恶性二分类。

## 环境
- 集群：DGX Spark，Slurm 分区 `gpu`
- Conda 环境：`~/envs/ml`（已装好 torch/torchvision/pandas/openpyxl/matplotlib/jupyter/ipykernel）
- Jupyter kernel 名称：`ml`（display name "Python (ml)"）

## 文件
- `Test.ipynb` — 环境测试笔记本，验证 notebook 能跑、PyTorch/GPU 正常。已执行过一次（CPU 环境下跑的，
  CUDA available 显示 False 是正常的——交互式终端没有分配 GPU，跑 Slurm job 时才会分到 GPU，可参考
  `../kean/README.md` 里 Job 42 的记录）。
- `BUSI_Classification.ipynb` — Training Task 1 的主 notebook：数据加载 → ResNet50 → 训练 → loss 曲线 →
  accuracy/recall/F1 → 预测结果可视化 → 保存模型。**已经通过 Job 44 正式跑完**，打开就能看到完整的
  20 epoch 训练结果（见下面"跑过的记录"）。
- `TASK_CHECKLIST.md` — 对照《Research Training Instructions (1)》逐条核对的完成情况清单。

## 数据集

已经上传到 `~/datasets/BUSI/`，结构如下：
```
~/datasets/BUSI/
├── images/        # 647 张图片：benign (1).png ... malignant (1).png ...
├── labels/        # 对应的分割 mask（本次分类任务用不到，忽略即可）
├── train.xlsx     # 517 条训练样本，列: Image, Label
└── test.xlsx      # 130 条测试样本，列: Image, Label
```
`Label` 列是数字，含义已经用文件名核实过：**0 = malignant（恶性），1 = benign（良性）**——这个映射关系
notebook 第 3.1 节会自动重新核实一遍并使用，不需要你手动改。良性 437 张 / 恶性 210 张，训练/测试两个
文件之间没有重复图片。

## 怎么跑

**方式一：交互式，在 notebook 里一格一格跑**（适合调试、确认数据格式没问题的时候）
```bash
cd ~/projects/research_training
# 在 VS Code 里直接打开 BUSI_Classification.ipynb，右上角选 kernel "Python (ml)"，逐格运行
```

**方式二：提交 Slurm job，完整跑一遍**（推荐用于正式训练——不用一直守着屏幕，VPN 断了也没事）
```bash
cd ~/projects/research_training
sbatch run_task1.slurm              # 提交
squeue -u $USER                     # 查看排队/运行状态
tail -f ~/logs/busi_task1-<JOBID>.out   # 看实时输出
```
跑完之后直接打开 `BUSI_Classification.ipynb`，里面已经保存了每一格的输出（打印的指标、loss 曲线图、
预测可视化图）。模型权重和 loss 曲线图另外也存了一份在 `outputs/` 目录下。

## 跑过的记录
- `Test.ipynb`：已在交互式终端跑通（CPU），验证 torch 2.14 / torchvision 0.29 环境正常。
- `BUSI_Classification.ipynb`：数据上传后，用一份临时的缩减版（只跑 2 epoch）在交互式终端（CPU）完整
  跑通了一遍全部流程，确认没有 bug：
  - 2 epoch 后 test accuracy 0.79，loss 从 0.61→0.38（train）/ 0.54→0.43（test），趋势正常。
  - 过程中发现并修复了两个问题：① 图片实际存放在 `images/` 子目录，不是 `BUSI/` 根目录；
    ② 数字标签方向（0/1 分别对应哪个类别）现在用文件名自动核实，而不是硬编码假设。
  - 另外发现交互式终端跑 DataLoader 多进程（`num_workers>0`）会因为 `/dev/shm` 空间不足报错，
    所以把 `NUM_WORKERS` 默认设成了 0（这个数据集不大，单进程读取足够快）。
  - 正式的 `BUSI_Classification.ipynb` 文件重新生成为**未执行**的干净版本（`NUM_EPOCHS=20`）后，
    跑 2 epoch 用的临时文件和产出已清理。
- **Job 44**（`busi_task1`，GPU 节点 `promaxgb10-4415`）：正式训练，20 epoch，跑完用时约 3~6 分钟。
  最终结果：
  | 指标 | 数值 |
  |---|---|
  | Test Accuracy | 0.9077 |
  | Precision | 0.9247 |
  | Recall | 0.9451 |
  | F1 score | 0.9348 |

  Train loss 从 0.617 → 0.007（epoch 20 时几乎降到 0），Test loss 在 0.3~0.5 之间波动、没有随
  epoch 持续下降。这是一个**过拟合（overfitting）**的典型信号，值得你自己打开
  `outputs/loss_curve.png` 看看曲线、想想为什么会这样、可以怎么改善（比如提前停止训练、加正则化/
  数据增强等）——这正是 PDF 里要求你自己"观察 loss 曲线、理解 overfitting"的部分。
