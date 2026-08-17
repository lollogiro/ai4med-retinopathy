# Diabetic Retinopathy Severity Grading: A Comparative Study of Lightweight Architectures and Ordinal Regression

**Authors:** Carolina Bonafè, Lorenzo Girotti

**Course:** AI for Medicine - University of Bologna (Prof. Stefano Diciotti), Academic Year 2025/2026

**Code:** [scratch.ipynb](scratch.ipynb) - [EfficientNet.ipynb](EfficientNet.ipynb) - [requirements.txt](requirements.txt)

## Abstract

Diabetic retinopathy (DR) severity grading from retinal fundus images is a natural target for automated screening support. This project compares a custom from-scratch CNN (MicroSENet, ~1.7M parameters) with a pretrained EfficientNetV2-RW-T (12.6M parameters), each trained with cross-entropy (CE) and CORAL ordinal regression on RetinaMNIST. Best test results: MicroSENet-CE QWK 0.7376 / AUC 0.8459, EfficientNetV2-RW-T CE partial fine-tuning accuracy 0.5200 / AUC 0.7924. CE is the more robust multi-class grader and yields the most clinically plausible saliency maps, while CORAL improves ordinal ranking in cross-validation on the pretrained backbone and gives the better-balanced binary referral filter on the from-scratch model.

## 1. Problem and Clinical Context

Diabetic retinopathy is a microvascular complication of diabetes and a leading cause of preventable blindness in working-age adults. Severity is graded on the ordinal International Clinical Diabetic Retinopathy (ICDR) scale (Wilkinson et al., 2003): 0 = no DR, 1 = mild NPDR (microaneurysms only), 2 = moderate NPDR, 3 = severe NPDR, 4 = proliferative DR (neovascularization). Grades 2-4 are **referable** and require specialist review, so automated grading can support large-scale screening, where false negatives are the costliest errors.

Two design questions are explored:

1. Can a small CNN trained from scratch compete with a much larger ImageNet-pretrained network (EfficientNetV2-RW-T; Tan & Le, 2021) under linear probing and partial fine-tuning? 
2. Does ordinal regression (CORAL; Cao et al., 2020) beat standard cross-entropy with class-balanced weights on an ordinal task? 
   
Evaluation covers 5-fold cross-validation, a held-out test set, clinically motivated screening thresholds, and HiResCAM interpretability.

## 2. Dataset: RetinaMNIST

RetinaMNIST (MedMNIST v2; Yang et al., 2023), derived from DeepDRiD (Liu et al., 2022), contains **1,600 retinal fundus images** (1,736×1,824 RGB originals, resized to 224×224) labeled with the five ICDR grades. Fixed splits: **train 1,080 / validation 120 / test 400**, the project adds 5-fold stratified cross-validation on the training split, with validation used only for early stopping and threshold tuning. The classes are heavily imbalanced (~45% grade 0, ~6% grade 4), so accuracy alone is misleading: macro F1, quadratic weighted kappa (QWK), and AUC are the primary metrics. The dataset provides no demographic metadata, so population representativeness cannot be assessed and generalization claims are correspondingly limited.

Released under **CC BY 4.0** for research and educational purposes only (**not for clinical use**). No patient-identifiable information is contained in the dataset.

## 3. Methods

- **Preprocessing (Ben Graham; Graham, 2014):** unsharp masking + circular mask applied offline and cached as PNGs, per-channel normalization computed on the training set only (recomputed per fold in CV).
- **Augmentation (train only):** random horizontal/vertical flips, random affine (rotation 90°, translation 10%), color jitter (brightness/contrast 20%). No cropping, to keep the retina in frame.
- **Leakage control:** MD5 duplicate check across splits (none found), class/importance weights computed inside each fold, thresholds tuned on validation only. RetinaMNIST provides no patient identifiers, so patient-level separation is not possible, the cross-split duplicate check is the available safeguard against exact-image leakage.
- **Training:** AdamW + cosine annealing, batch size 32, 50 epochs with early stopping on validation AUC, fixed seed and 5-fold CV. Then final training on the full training set. CE uses class-balanced weights, CORAL uses *importance weights* (Cao et al., 2020, Eq. 7).
- **Architectures:**
  - **MicroSENet (~1.7M params, from scratch):** 5 conv blocks, SE blocks, DLA-inspired feature fusion (concatenating block 2 and 5), 256-dim compression bottleneck, mild and final dropout.
  - **EfficientNetV2-RW-T (12.6M params, ImageNet-pretrained):** linear probing or partial fine-tuning (also last stage fine-tuned).
   
   Both trained with CE and CORAL heads.

## 4. Results

Test set metrics are reported for the final models trained on the full training set, while cross-validation metrics are reported as mean ± std across folds.

For the final operating point selection, thresholds in the screening tables were tuned on the validation set (mainly for Sensitivity$\ge 0.95$) and applied to the test set.

### 4.1 Cross-validation

*Table 1: 5-fold cross-validation, mean $\pm$ std.*

| Model | Config | Accuracy | QWK | AUC |
|-------|--------|----------|-----|-----|
| MicroSENet (~1.7M) | CE | 0.5093 $\pm$ 0.0356 | **0.6727 $\pm$ 0.0265** | **0.8209 $\pm$ 0.0115** |
| MicroSENet (~1.7M) | CORAL | **0.5102 $\pm$ 0.0269** | 0.6452 $\pm$ 0.0280 | 0.7475 $\pm$ 0.0160 |
| EfficientNetV2-RW-T | CE, linear probing | 0.4241 $\pm$ 0.0451 | 0.5013 $\pm$ 0.0290 | 0.7142 $\pm$ 0.0298 |
| EfficientNetV2-RW-T | CORAL, linear probing | 0.4685 $\pm$ 0.0153 | 0.6024 $\pm$ 0.0231 | 0.7291 $\pm$ 0.0139 |
| EfficientNetV2-RW-T | CE, partial fine-tuning | 0.4722 $\pm$ 0.0343 | 0.5780 $\pm$ 0.0669 | 0.7785 $\pm$ 0.0268 |
| EfficientNetV2-RW-T | CORAL, partial fine-tuning | 0.5074 $\pm$ 0.0130 | 0.6559 $\pm$ 0.0135 | 0.7634 $\pm$ 0.0111 |

### 4.2 Test set

*Table 2: Test set (n=400).*

| Model | Params | Config | Accuracy | AUC | QWK | Macro F1 |
|-------|--------|--------|----------|-----|-----|----------|
| MicroSENet | ~1.7M | CE | 0.4850 | **0.8459** | **0.7376** | 0.4323 |
| MicroSENet | ~1.7M | CORAL | 0.5125 | 0.7769 | 0.7110 | 0.3427 |
| EfficientNet | 12.6M | CE, partial fine-tuning | **0.5200** | 0.7924 | 0.6749 | **0.4460** |
| EfficientNet | 12.6M | CORAL, partial fine-tuning | 0.4800 | 0.7557 | 0.6497 | 0.2854 |

The two MicroSENet heads differ markedly on rare classes: the CE model keeps sensitivity on minority grades (**Grade 1 Recall 0.630, Grade 4 Recall 0.700**), while the CORAL model's accuracy is driven by the majority class (Grade 0 Recall 0.851) at the cost of collapsing Grade 1 (**Recall 0.022, F1 0.030**). Overall, the from-scratch MicroSENet-CE achieves the best test **QWK (0.7376)** and **AUC (0.8459)** of all models despite ~7× fewer parameters. The pretrained EfficientNet CE partial fine-tuning version wins on accuracy (0.5200) and macro F1 (0.4460).

### 4.3 Binary screening (referable = grades 2-4)

*Table 3: Binary screening on the test set. The Se $\ge$ 0.95 threshold was tuned on the validation set and then applied to the test set.*

| Model | Config | Threshold (Se ≥ 0.95) | Sensitivity | Specificity | PPV | NPV |
|-------|--------|-----------------------|-------------|-------------|-----|-----|
| MicroSENet | CE | 0.173 | **0.978** | 0.514 | 0.622 | 0.966 |
| MicroSENet | CORAL | 0.229 | 0.978 | 0.609 | 0.672 | **0.971** |
| EfficientNet | CE, partial fine-tuning | 0.108 | 0.967 | 0.259 | 0.516 | 0.905 |
| EfficientNet | CORAL, partial fine-tuning | 0.252 | 0.900 | **0.709** | **0.717** | 0.897 |

Both MicroSENet heads reach 0.978 sensitivity for referable DR, the CORAL head is the more specific filter (Se 0.978 / Sp 0.609 / PPV 0.672 vs CE 0.978 / 0.514 / 0.622). 

On EfficientNet, CE maximizes sensitivity (0.967) at the cost of low specificity (0.259), CORAL trades sensitivity for a much more balanced operating point with the best PPV (0.717).

## 5. Discussion (TODO da rivedere insieme)

**Ordinal regression is a trade-off.** CORAL improves QWK in CV on the pretrained backbone (CORAL-PFT 0.6559 vs CE-PFT 0.5780), but not on MicroSENet (0.6452 vs 0.6727), and CE-PFT beats CORAL-PFT on test QWK (0.6749 vs 0.6497). On small models CORAL collapses the rare intermediate grades: grade-1 recall 0.022 vs CE 0.630 on MicroSENet, and CORAL-PFT test macro F1 (0.2854) far below CE-PFT (0.4460).

**Lightweight from-scratch models are competitive.** MicroSENet-CE attains the best test QWK and AUC of all models despite ~7× fewer parameters than the ImageNet-pretrained network, which leads on accuracy and macro F1. Heavy regularization (SE, DLA, dropout, augmentation, no cropping) lets a small CNN hold its own, relevant for low-resource screening deployments.

**For binary screening the objective flips.** Sensitivity is paramount (never miss disease), but specificity matters to avoid flooding clinics with false referrals. The CORAL head is the better-balanced filter on MicroSENet (Se 0.978 / Sp 0.609 / PPV 0.672 vs CE 0.978 / 0.514 / 0.622), on EfficientNetV2-RW-T, CE-PFT maximizes sensitivity (0.967) at low specificity (0.259) while CORAL-PFT is balanced (0.900 / 0.709, best PPV 0.717). Thresholds were selected on the validation set via the Youden index and sensitivity/specificity constraints (Youden, 1950).

**Interpretability supports CE.** HiResCAM maps show the CE model attends to clinically plausible structures (vascular network in grades 0-2, neovascularization/hemorrhages in grades 3-4), whereas CORAL cannot visually separate grades 0 and 1, an argument for CE where explanations matter as much as accuracy.

**Limitations.** Small dataset (1,600 images), $224\times224$ downsampling, single benchmark, thresholds calibrated on 120 validation samples, the results are research findings from a public benchmark, not clinically validated.

## 6. Conclusions (TODO da rivedere insieme)

Cross-entropy with class-balanced weights is the more robust objective for multi-class severity grading (best test QWK and AUC, sensitivity on rare grades, and clinically plausible saliency maps), while CORAL helps ordinal ranking on pretrained backbones but collapses intermediate grades on small imbalanced models. A 1.7M-parameter network trained from scratch matching a 7× larger pretrained model shows that lightweight, heavily regularized CNNs are a viable direction for resource-constrained screening.

Future work: larger and external datasets, recalibrated thresholds on real screening populations, and quantitative attention evaluation.

## 7. Ethics, Privacy, and Compliance

- **Dataset license and intended use.** RetinaMNIST is released under CC BY 4.0 for research and educational purposes. **This project is a research exercise, not a clinical tool**: the models must not be used for clinical decision-making or diagnosis.
- **Privacy.** RetinaMNIST contains no patient-identifiable information, it is a de-identified derivative of DeepDRiD (Liu et al., 2022). No external or private data was collected, all processing ran on public benchmark data in Google Colab.
- **Attribution.** CC BY 4.0 requires attribution to MedMNIST (Yang et al., 2023) and DeepDRiD (Liu et al., 2022).
- **Software licenses.** All dependencies are permissively licensed: `medmnist`, `timm` (Apache-2.0), `coral-pytorch`, `torchinfo`, `grad-cam` (MIT).
- **Explainability responsibility.** Interpretability was assessed qualitatively with HiResCAM because medical deployment demands plausible explanations, the reported operating point selection and thresholds are dataset-specific and would require recalibration on real screening populations.

## 8. Reproducibility (TODO da rivedere insieme)

Run the notebooks top-to-bottom in **Google Colab** (GPU), every number in this README comes from their outputs.

1. **Install dependencies** (first cells of each notebook):

   ```python
   !pip install medmnist==3.0.2 torchinfo==1.8.0 coral_pytorch==1.4.0 grad-cam==1.5.5
   ```

   `EfficientNet.ipynb` additionally installs `timm==1.0.28`, the same versions are in [requirements.txt](requirements.txt).

2. **Run `scratch.ipynb` first**: it applies the offline Ben Graham preprocessing and caches PNGs under `PREPROCESSED_DIR = "/content/preprocessed"` (Colab path), `EfficientNet.ipynb` expects the same layout. RetinaMNIST is downloaded automatically by `medmnist` on first use.

3. **Determinism.** `SEED=42` throughout (`set_seed` per experiment, seeded DataLoader workers, deterministic algorithms).

4. **Artifacts.** No checkpoints are saved, metrics and HiResCAM figures are printed/displayed inside the notebooks.

## References

- Cao, W., Mirjalili, V., & Raschka, S. (2020). Rank consistent ordinal regression for neural networks with application to age estimation. *Pattern Recognition Letters, 140*, 325-331. https://arxiv.org/abs/1901.07884
- Graham, B. (2014). *Kaggle diabetic retinopathy detection competition* (1st-place solution; unsharp masking and circular-mask preprocessing). https://www.kaggle.com/c/diabetic-retinopathy-detection/discussion/15801
- Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Fei, T., et al. (2022). DeepDRiD: Diabetic retinopathy grading and image quality estimation challenge. *Patterns, 3*(6), 100512. https://doi.org/10.1016/j.patter.2022.100512
- Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller models and faster training. *Proceedings of the 38th International Conference on Machine Learning (ICML)*. https://arxiv.org/abs/2104.00298
- Wilkinson, C. P., Ferris, F. L., Klein, R. E., Lee, P. P., Agardh, C. D., Davis, M., et al. (2003). Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. *Ophthalmology, 110*(9), 1677-1682.
- Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., et al. (2023). MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification. *Scientific Data, 10*, 41. https://doi.org/10.1038/s41597-022-01721-8
- Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer, 3*(1), 32-35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
