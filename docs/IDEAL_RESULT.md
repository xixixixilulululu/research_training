# 理想结果标准（Ideal Result Criteria）

这份文档回答一个问题："这个实验做到什么程度才算好？" ——写在这里，是为了给老师和以后回顾的自己一个
清楚的目标基准，不用每次都重新想"现在这个数字到底算不算行"。跟 `TASK_CHECKLIST.md`（对照老师 PDF 逐条
核对"要求做的事有没有做"）不是一回事：那份是任务完成度清单，这份是**结果质量**的评判标准。

每一条标准后面都写了"现状"，随着后续继续调参会更新，不是一次性写完就不变的。

## 1. Loss 曲线形状：train/val 应该一起下降、收敛到接近的水平

理想情况：train loss 和 val loss 一起往下降，最后收敛到接近的水平并趋于平稳，两者之间保留一个**很小**
的 gap（不是要求 gap=0——gap=0 反而说明训练/验证太像了，可能哪里有问题——而是像 run8 第 5 轮那种
gap≈0.018 的量级：train_loss=0.3606，val_loss=0.3781）。不应该是 Job44 那种 train loss 一路降到
0.007、val loss 却还在 0.35 附近来回震荡的情形——那是模型在死记训练集，而不是学到了能泛化的特征。

**现状：✅ 已达到。** run8/run9/run10 用的都是同一个 checkpoint（第 5 轮 val_f1 最高的那一轮，见
`outputs/run8_history.json`），这一轮 train_loss=0.3606、val_loss=0.3781，gap≈0.018，跟 Job44 的
"train→0.007、val 震荡不降"是完全不同的两种曲线形状。

## 2. 指标要对准真正关心的目标：medical screening 场景应该看恶性（malignant）的 recall

现在 notebook 里 `precision_score` / `recall_score` / `f1_score` 都是用 sklearn 默认的 `pos_label=1`
算的。第 3.1 节核实数字标签含义后，`LABEL_MAP` 被重新赋值为 `malignant=0, benign=1`（覆盖了第 1 节里
`benign=0, malignant=1` 的占位值），所以现在报告的 precision/recall/f1 **实际上都是"良性（benign）"
口径**的——也就是说，历史上所有跑出来的 Recall（比如 run1 的 0.9451、run10 的 0.8791）说的都是"良性
被正确识别的比例"，不是临床上真正关心的"恶性被正确识别的比例"。

医学筛查场景里，漏诊一个恶性（假阴性：真恶性被判成良性）的代价，远高于误报一个良性（假阳性：真良性
被判成恶性）——前者可能延误治疗，后者最多是多做一次复查。所以应该把"恶性"当成 positive class 来看
recall，而不是现在默认的"良性"口径。

**现状：✅ 已完成。** run11 开始，`MALIGNANT_ID` 改成用 `CLASS_NAMES.index("malignant")` 动态确认
（不再硬编码假设顺序），并用 `classification_report(final_labels, final_preds, target_names=CLASS_NAMES)`
把 malignant/benign 两个类别的 precision/recall/f1 分开算、分开存进 `outputs/run{N}_history.json`。

- **run11**（layer3，checkpoint 仍按 F1 挑，只是把评估口径换成恶性）：malignant precision/recall/f1 =
  0.75 / 0.85 / 0.80；benign 0.93 / 0.88 / 0.90。checkpoint 选中的还是第 5 轮，跟 run8/9/10 完全一样的
  权重——说明"换评估口径"本身不影响模型，只是之前没把这个数字算出来、汇报出来。
- **run14**（在 run11 基础上更进一步：CrossEntropyLoss 按 malignant/benign 实际数量反比加 class weight，
  early stopping/选 checkpoint 也从普通 F1 换成 `fbeta_score(beta=2, pos_label=malignant)`，recall
  权重是 precision 的 4 倍）：checkpoint 选中的变成第 8 轮，test 集恶性 precision/recall/f1 提升到
  0.78 / **0.90** / 0.83，良性那边也没有变差（0.95 / 0.89 / 0.92）——不是靠牺牲 precision 换 recall
  的权衡，是训练方式本身改善了对恶性类别的识别。漏诊数从 6/39 降到 4/39（见
  `outputs/run14_missed_malignant_cases.png`，剩下 4 个漏诊案例的恶性概率都在 0.25~0.37，是卡在
  门槛边缘的模糊案例，不是模型离谱判断错）。
- 额外做了一次**门槛扫描**（不重新训练，只调预测门槛）：run11 的权重把门槛从 0.5 降到 0.25，恶性
  recall 能拉到 1.0（precision 掉到 0.64，见 `outputs/run13_threshold_sweep.png`）；同样的扫描在
  run14 权重上做（`outputs/run15_threshold_sweep.png`）反而在高召回区间 precision 更差
  （recall=1.0 时 precision 只有 0.56，不如 run11 门槛调整后的 0.64）——说明 run14 训练时已经把"廉价
  的召回率提升"提前拿走了，不能简单叠加"训练时优化 + 门槛调整"两种手段期待双重收益。

## 3. K-fold 标准差要小：均值高不够，还要"切哪几份都差不多"

理想情况：5 折交叉验证不仅均值要高，标准差也要小（比如 ±0.01 量级），说明不管数据怎么切、模型表现
都很稳定，不是"运气好切到了一份好切分"。

**现状：⚠️ 部分达到，试过一种改法但没用。** layer3 配置（run10，batch_size=16）5 折验证集 accuracy
均值 0.9168，标准差 ±0.0217（5 折分别是 0.8942 / 0.9519 / 0.9320 / 0.9029 / 0.9029，最高最低差了
近 6 个百分点）；F1 均值 0.9396，标准差 ±0.0155。

试过把 `BATCH_SIZE` 从 16 改成 32、其他配置不变重新跑一遍 5 折（run12）：结果标准差没有降下来，反而
略微变差了——accuracy 均值 0.9091（降了）、标准差 ±0.0255（升了）；F1 均值 0.9352、标准差 ±0.0161
（也是略降/略升）。说明"batch size 越大越稳"这个假设在这个数据量下不成立，标准差还有改进空间，但
不是靠调 batch size 能解决的，可能得从数据增强、正则化强度，或者干脆承认这是 647 张图这个数据量下
折间波动的自然下限（见第 6 条）来想办法。

## 4. 方法论必须干净：test 集只摸一次，checkpoint 完全基于 val

理想情况：test 集只在训练/调参全部结束后**摸一次**，用来做最终汇报；训练过程中的 early stopping、
挑选最佳 checkpoint，全部只看 val 集，不允许任何形式偷看 test 集。

**现状：✅ 已达到。** run4（Job50）修复了这个问题：训练集 517 条按类别比例分层切出
train_sub(413)/val(104)，early stopping 和保存最佳权重只看 `val_f1`（`CHECKPOINT_METRIC`），test 130
条全程不参与训练/调参，只在 9.4 节跑一次做最终汇报。之后历次 K 折交叉验证（run6、run10）也只在
`train_df` 内部切，test 集依然完全不碰，进一步证实了这套方法论是干净的。（run3/Job49 是反面教材：当时
early stopping 直接看 test F1，属于偷看测试集，方法论有缺陷，已在 README「最终总结」里记录。）

## 5. 概率校准要准（次要目标）：模型说多少把握，就该有多准

理想情况：Temperature scaling 学出来的温度 T 应该能让模型输出的置信度更接近真实概率，且这个校准效果
不只在 val 集上有效，还能泛化到 test 集上。

**现状：⛔ 尝试过，但没有做到。** 9.3 节在 val 集上拟合温度 T（run10 里 T=0.9897），能降低 val loss；
但顺便在 test 集上算的校准前后 loss 几乎没变（`test_loss_uncalibrated=0.4034` →
`test_loss_calibrated=0.4036`，反而略微变差），说明这次学到的校准没能泛化到 test 集。这属于次要目标，
不影响 accuracy/precision/recall/F1（这几个指标只看 argmax，跟 T 无关），暂时搁置也不影响主线结论。

## 6. 数据量天花板：647 张图本身就限制了能达到的上限

BUSI 数据集总共只有 647 张图（训练 517 / 测试 130，良性 437 / 恶性 210），跟 ImageNet 那种百万级数据
集完全不是一个量级。理想结果不是"追求一个脱离这个数据量该有的高分"，而是**诚实地逼近 647 张图这个
数据量本身能撑到的天花板**——K 折标准差降不到接近 0、单次 test 集抽样会有明显波动，某种程度上都是
这个天花板的自然体现，不是代码或方法论的锅。判断实验做得好不好，应该看"有没有把不该有的误差消除掉"
（比如过拟合、偷看测试集），而不是硬要凑出一个大数据集才有的分数。

**现状：（背景约束，不是一个"做没做到"的检查项）** 目前 run8~run10 这套 layer3 + 干净的三路切分 + 5
折验证的结果（test 单次 accuracy 0.8692，5 折均值 0.9168±0.0217），大概率已经比较接近这批数据、这个
预训练 backbone 组合下的天花板；第 2、3、5 条列出的改进空间不会让这个天花板消失，但值得继续做，能让
汇报出来的数字更贴近这个天花板、更少受口径/波动/校准这些因素干扰。
