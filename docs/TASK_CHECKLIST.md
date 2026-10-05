# Task completion checklist

Checked item by item against "Research Training Instructions (1)" (sections 1–3) and "Research Training
Instructions (3)" (section 4). ✅ = done, ⛔ = not done (and not something I can do for you).

## 1. General tips

| PDF requirement | Status | Notes |
|---|---|---|
| Determine the GPU server type: `nohup` on a normal GPU server, Slurm on HPC (e.g. DGX_Spark) | ✅ | We are on DGX Spark and submit everything through Slurm (`kean/train.slurm`, `run_task1.slurm`) |
| The VPN drops every 30–60 minutes; programs must not be affected | ✅ | Slurm jobs run on compute nodes, independent of the VPN/terminal window |
| Download the BUSI dataset and upload it to the server | ✅ | Uploaded to `~/datasets/BUSI/` |
| Keep a separate `datasets` folder and `projects` folder | ✅ | `~/datasets/BUSI/`, `~/projects/research_training/` |

## 2. Preparation before training

| PDF requirement | Status | Notes |
|---|---|---|
| Use an AI chatbot to understand: .py vs .ipynb, deep learning, CV/classification/segmentation, data & label, train/val/test split | ⛔ **not done** | The PDF explicitly says "learn it yourself with an AI chatbot" — this is your learning task, I cannot "understand" it for you |
| Understand CNN concepts: convolution, pooling, batch norm, activation function, FC layer | ⛔ **not done** | Same as above, ask the AI and learn it yourself |
| Understand training concepts: forward/backward propagation, epoch, loss function, optimizer, learning rate, batch size, overfitting/underfitting | ⛔ **not done** | Same as above; the first training run already shows a real overfitting example (README "Runs so far") you can use |
| Read the instructor's CV course slides | ⛔ **not done** | I cannot see whether you have read them; please confirm yourself |
| Create the `research_training` folder | ✅ | `~/projects/research_training/` |
| Create `Test.ipynb`, write and run a simple Python program | ✅ | Created and executed successfully (see `Test.ipynb`) |
| Learn to write well-organised notebooks with text cells + code cells | ✅ (code side) | `Test.ipynb` and `BUSI_Classification.ipynb` follow this structure; "learning" it is up to you |

## 3. Training Task 1: BUSI image classification

| PDF requirement | Status | Notes |
|---|---|---|
| Use an AI agent to help write code, comments, result documents and analysis | ✅ | That is what we did |
| Benign/malignant binary classification with ResNet50, written as a notebook | ✅ | `BUSI_Classification.ipynb` |
| Resize input images to 224×224 | ✅ | |
| Split train/test with the two xlsx files, train on train, evaluate on test | ✅ | 517 train / 130 test, no overlap; the latest version also splits the 517 training samples (stratified) into train_sub (413) / val (104); early stopping / model selection only look at val, and the 130 test samples are never used for tuning, avoiding "peeking at the test set" |
| Load ImageNet-pretrained weights | ✅ | `ResNet50_Weights.IMAGENET1K_V2` |
| Compute standard classification metrics: accuracy, recall, F1, ... | ✅ | Results: Accuracy 0.9077 / Precision 0.9247 / Recall 0.9451 / F1 0.9348 |
| Plot train/test loss per epoch | ✅ | `outputs/run3_loss_curve.png` (runs 1/3, train vs test); from run 4 on the per-epoch curves are train vs **val** (`outputs/run{N}_loss_curve.png`) and test is evaluated once after training (`outputs/run{N}_test_metrics.png`), so no test data enters the per-epoch curves |
| **Observe** the loss curves, understand underfitting/overfitting, decide on a suitable number of epochs | ⛔ **not done** | The plots exist and show a clear overfitting signal (train loss→0.007 while test loss fluctuates between 0.3 and 0.5), but "observe + understand + decide" is explicitly your task in the PDF; I cannot draw the conclusion for you |
| Show every test image with true and predicted label | ✅ | Notebook section 11; wrong predictions are marked in red |
| Text description before every code cell, necessary comments in the code | ✅ | |
| **With the help of AI, understand every line of code and the knowledge behind it** | ⛔ **not done** | The PDF says "with the help of AI, understand" — you need to understand it yourself; running the code does not count |

## 4. Training Task 6–8: BUSI image segmentation ("Research Training Instructions (3)")

| Requirement | Status | Notes |
|---|---|---|
| Three subfolders, each with one notebook + one Slurm script | ✅ | `task6_unet/`, `task6_nnunet/`, `task6_deeplabv3plus/` |
| Use the train.xlsx / test.xlsx split, no random re-splitting | ✅ | 517 / 130; the code asserts that they do not overlap |
| Check mask naming and correspondence, merge multiple masks | ✅ | Same filename as the image, one mask per image; the `_mask*` OR-merge logic is kept and was checked on the whole dataset |
| Output binary tumour masks | ✅ | `predictions/*.png` in each folder, original image size |
| nnU-Net: data conversion (folder layout + dataset.json), training and prediction commands | ✅ | Notebook sections 5–11 |
| Dice, IoU, HD95 on the test set: means + per-image `per_image_metrics.csv`, same order for all three models | ✅ | The Task 8 notebook asserts the order again |
| Uniform HD95 rule for empty predictions, documented, number of empty predictions printed | ✅ | HD95 = original image diagonal; empty predictions U-Net 5 / nnU-Net 1 / DeepLabV3+ 0 |
| End of notebook: original / ground truth / prediction / overlay for every test image | ✅ | All 130 images shown |
| Model weights and predicted masks saved in each model's folder | ✅ | `outputs/*.pt`; nnU-Net in `nnUNet_results/.../checkpoint_final.pth` |
| Markdown description before every code cell, comments in the code | ✅ | |
| Fair comparison: same input size, preprocessing, seed, batch size, optimizer, learning rate, epochs, augmentation; one configuration cell at the top | ✅ | The shared-hyperparameter block is identical in the three notebooks; the differences (DeepLabV3+ pretraining, nnU-Net keeping its defaults) are documented with reasons |
| Task 7: three models side by side on the same test image; examples of under-segmentation / over-segmentation / missed tumour / inaccurate boundary | ✅ | `task7_visual_comparison/`, figures in `figures/` |
| Statistics: mean ± SD, paired t-test + Holm, Wilcoxon, results table, Markdown explanations | ✅ | `task8_statistics/statistical_analysis.ipynb` |
| Task 8: `segmentation_comparison.md` in the project root | ✅ | All numbers come from the actual runs |
| **Understand** Dice / IoU / HD95 and the statistical tests, and be able to explain the results yourself | ⛔ **not done** | As with the earlier tasks, the "understanding" part is yours: work through the Task 7 illustrations and the Task 8 Markdown explanations |

## Result quality criteria

This checklist tracks whether the things the PDF asks for were done; whether the results are *good* is covered
in a separate `IDEAL_RESULT.md`, which lists 6 criteria for an ideal result (loss-curve shape, metric
definition, k-fold standard deviation, methodological cleanliness, probability calibration, data-size
ceiling), each marked as met or not, for reference when tuning further.

## Summary

- **Code / environment / training all work**: data, notebook, Slurm submission and the first full training
  run are done, with real results.
- **What is genuinely not done is the part the PDF explicitly says "understand it yourself"** (learn the
  concepts, understand every line of code, observe and interpret the loss curves). I cannot do this for you;
  take the results you already have (e.g. the overfitting visible in the loss curves) to an AI chatbot and the
  course slides and work through them.
- The PDF title has "(1)", so there may be follow-up documents — worth confirming with the instructor.
- Tasks 6–8 of "Research Training Instructions (3)" (segmentation + statistical tests + comparison document)
  are done in code, training and documentation; see section 4.
