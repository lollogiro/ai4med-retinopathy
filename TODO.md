# Project TODOs:

RetinaMNIST is related to [this dataset](https://github.com/deepdrdoc/DeepDRiD)

Project objectives:

* Assess performance between small and larger vision models (MobileNet vs. DINOv3) for diabetic retinopathy classification.
* Evaluate the impact of CORAL loss for ordinal classification of severity grades w.r.t. to standard Cross-Entropy loss.
* Investigate the pros and cons of fully frozen vs. partially fine-tuned models for both architectures.
* Try to introduce explainability (GradCAM++) to verify if the models are focusing on relevant features and explain errors
* Try to improve clinically relevant metrics (sensitivity for severe cases) by tuning thresholds or using temperature scaling.

### 1. Environment setup
- [ ]   **Fix random seeds**
- [ ]   **Init `requirements.txt`**
- [ ]   **Save dataset license info** (requested by the professor)

### 2. Data exploration and pre-processing
- [ ]   **Load dataset from Python package `medmnist`**
- [ ]   **Disclose data biases (if possible)**: Analyze and document demographic biases or class imbalances (e.g., high number of healthy subjects vs. rare proliferative cases) in the datasets.
- [ ]   **Check for duplicates in the dataset**: Run an MD5 hash comparison on the different splits to avoid data leakage between different sets
- [ ]   **Assess the 8 points of possible data leakage** (see slides)
- [ ]   **Apply image pre-processing:**

  * [Extract exudate area from the image as most relevant feature](https://biomedpharmajournal.org/vol10no2/diabetic-retinal-fundus-images-preprocessing-and-feature-extraction-for-early-detection-of-diabetic-retinopathy/)
  * [Different pre-processing techniques for diabetic retinopathy (the first two have been found as better)](https://link.springer.com/chapter/10.1007/978-3-030-96305-7_18)
  * [Ben Graham's pre-processing, MobileNetV3 and GradCAM++ (it has many useful references)](https://ieeexplore.ieee.org/abstract/document/11374864)
  * [Usuyama pre-processing as the best (EfficientNetB7)](https://joiv.org/index.php/joiv/article/view/857/403)
  * [EfficientNetB0 with many pre-processing techniques tried](https://link.springer.com/chapter/10.1007/978-3-031-66594-3_8)

### 3. Dataset splitting
- [ ]   **Perform patient-level separation (if any info available)**: If multiple images belong to the same patient (e.g., left and right eye), guarantee that all images from that specific patient are grouped entirely in either the training set or the test set to avoid biased evaluations (as the professor requested).
- [ ]   **Nested hold-out**: inner split to select the best model (see slides)

### 4. Model architectures and training configurations
- [ ]   **Prepare models:** Instantiate MobileNetV4(?) for lightweight inference and DINOv3-Small for dense feature extraction with a bigger model.
- [ ]   **Configure loss functions:** Set up standard Cross-Entropy for baseline runs and CORAL loss to handle the ordinal classification of severity grades. Use https://github.com/Raschka-research-group/coral-pytorch for CORAL loss implementation.
- [ ]   **Ablation study**:

    | Experiment ID | Architecture | Freezing Strategy | Loss Function | 
    | :--- | :--- | :--- | :--- |
    | **Exp_01** | MobileNetV4(?) | Fully frozen | Cross-Entropy |
    | **Exp_02** | MobileNetV4(?) | Partial fine-tuning | CORAL |
    | **Exp_03** | DINOv3-Small | Fully frozen | Cross-Entropy |
    | **Exp_04** | DINOv3-Small | Partial fine-tuning | CORAL |

### 5. Training and validation pipeline
- [ ]   **Training:** Initialize the RetinopathyMNIST dataset specifying the `size=224` parameter to load high-resolution images.
- [ ]   **Nested cross-validation:** Use a robust validation method like nested cross-validation or stratified hold-out validation.
- [ ]   **Select operating points:** Calculate ROC curves and explicitly select metric thresholds (e.g., prioritizing sensitivity for severe cases) based on clinical relevance, on validation or a separate hold-out tuning set (from the training or validation(?))
- [ ]   **Area under the ROC curve (AUC):** Compute AUC for each model and training configuration to quantify overall performance withouth thresholding.
- [ ]   **Other datasets for OOD evaluation:** 
  Professor requested only datasets from some sources (e.g., Zenodo)
  * [IDRiD Dataset](https://zenodo.org/records/17219542)
  * [FGADR Dataset](https://csyizhou.github.io/FGADR/)
  * [mBRSET, a Mobile Brazilian Retinal Dataset](https://www.physionet.org/content/mbrset/1.0/)

### 6. Explainability
- [ ]   **Apply Grad-CAM:** Generate Grad-CAM (or Grad-CAM++) heatmaps for both the fully frozen runs and the partially fine-tuned runs.
- [ ]   **Evaluate Clinical Alignment:** Manually verify if the Grad-CAM hotspots align with true pathological features (microaneurysms, exudates) rather than imaging artifacts.
- [ ]   **Model weight sanity check:** Randomize the model weights and re-run Grad-CAM to prove that the heatmaps degrade into noise and are faithfully tied to the learned parameters.

### 7. Report (as requested by the professor)
- [ ]   **Structure independence**: Ensure the Introduction, Problem Statement, Data Description, Methods, Results, and Discussion sections are distinctly separated.
- [ ]   **Map Objectives to Results:** Verify absolute consistency by ensuring every stated project objective has a corresponding result.
- [ ]   **Draft the Discussion:** Write a narrative interpretation of the results, explaining the performance differences between MobileNet and DINOv3, and the impact of CORAL loss.
- [ ]   **Write the Conclusion:** Craft a brief, one- or two-sentence take-home message that directly answers the initial clinical objective without summarizing the whole document.
- [ ]   **Document Limitations and Future Work:** State realistic constraints (like computational limits or dataset size) and define concrete, immediate next steps (not long-term research directions) for future work.