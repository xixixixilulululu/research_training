# 7 天学习计划 — 把 TASK_CHECKLIST.md 里 ⛔ 的部分学完

对应《Research Training Instructions (1)》里要求"自己用 AI chatbot 理解"的那几项（见
`TASK_CHECKLIST.md`）。每天 1~2 小时，尽量结合我们自己跑出来的代码和结果学，比抽象例子好记。

- [x] **Day 1 — 基础概念**（≈1h）
  - 学：`.py` vs `.ipynb`、什么是 Deep Learning、CV 里分类 vs 分割的区别、data 和 label、train/val/test 划分
  - 怎么学：直接问 AI chatbot，让它拿 BUSI 任务举例（图片是 data，benign/malignant 是 label）
  - 产出：合上电脑能用大白话讲清楚

- [x] **Day 2 — CNN 基础**（≈1.5h）
  - 学：convolution、pooling、batch normalization、activation function、fully connected layer
  - 结合：`BUSI_Classification.ipynb` 第 6 节（模型构建），问 AI "ResNet50 长什么样，信息怎么从图片流到 2 个类别分数"
  - 产出：能画个简单流程图 / 说清楚数据流向

- [x] **Day 3 — 训练机制**（≈1.5h）
  - 学：forward/backward propagation（不用记公式）、epoch、loss function、optimizer、learning rate、batch size
  - 结合：第 7-8 节代码，让 AI 逐行解释 `train_one_epoch` 和 `evaluate`
  - 产出：能说清楚一个 epoch 训练时数据/梯度/参数分别怎么变化

- [ ] **Day 4 — 分析自己的过拟合案例**（≈1h）
  - 学：overfitting / underfitting 定义
  - 结合：`outputs/loss_curve.png`，对着 train loss（0.617→0.007）vs test loss（0.3~0.5 波动）这组真实数字，
    自己写一段分析：为什么是过拟合、可能原因、常见对策（early stopping、数据增强、dropout、weight decay）
  - 产出：3~5 句话的分析，可以写成 notebook 里新的一个 markdown cell

- [ ] **Day 5 — 精读代码：数据部分**（≈2h）
  - 通读第 1~5 节：CONFIG、导入库、数据核实、Dataset/DataLoader、可视化样本
  - 重点问 AI：为什么用 ImageNet 均值方差归一化？为什么训练集做翻转增强、测试集不做？`__getitem__` 在干嘛？
  - 产出：能讲清楚一张图片怎么从 xlsx+png 变成送进模型的 tensor

- [ ] **Day 6 — 精读代码：模型/训练/评估部分**（≈2h）
  - 通读第 6~12 节：模型构建、损失函数/优化器、训练循环、loss 曲线、评估指标、预测可视化、保存模型
  - 重点问 AI：为什么要换掉 ResNet50 最后的 fc 层？为什么 `evaluate` 里用 `@torch.no_grad()`？
    precision/recall/F1 分别衡量什么，医学场景里为什么 recall 常常更重要？
  - 产出：能完整讲一遍从读数据到保存模型的全流程

- [ ] **Day 7 — 复盘 + slides + 提问清单**（≈1.5h）
  - 过一遍老师的 CV 课程 slides，查漏补缺
  - 整理一份"还没搞懂 / 想问老师"的问题清单
  - 把 `TASK_CHECKLIST.md` 里 ⛔ 的几项，理解后改成 ✅
  - 顺便问老师 PDF 是不是还有 (2)(3) 后续文档
