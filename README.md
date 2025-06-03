# Hybrid Local-Global Attention (HLGA) Framework for Brain Tumor Classification

### Abstract
This study introduces the Hybrid Local-Global Attention (HLGA) framework, enhanced by a Dynamic Feature Integrator (DFI), for precise brain tumor classification using T1-weighted MRI scans from a Kaggle dataset (7,022 images). The model integrates ResNet50-based local feature extraction and Vision Transformer-based global attention, achieving 99.83% accuracy (precision: 99.93%, recall: 99.86%, F1-score: 99.89%), surpassing the prior benchmark of 99.22%. Ablation studies confirm the DFI's two-stage bottleneck attention reduces misclassifications to two cases. External validation on the Crystal Clean MRI dataset demonstrates robust generalizability, supporting early and accurate brain tumor diagnosis in neuro-oncology.

### Model Architecture
As the paper is under review, the full implementation (`HLGAModel`, `DynamicFusion`) is withheld to protect intellectual property. A placeholder model is provided in `src/model.py`.

<p align="center">
  <img src="Figures/Model Architecture.png?raw=true" alt="Model Architecture" width="100%">
</p>
<p align="center"><i>Caption: Placeholder schematic of the HLGA framework. Full details will be released upon publication.</i></p>

## Usage
The pipeline is modularized into multiple scripts under `src/` for preprocessing, model definition, training, evaluation, and visualization. The entry point is `src/main.py`.

## Data Source
The dataset is sourced from:  
<p>Nickparvar, M. (2021). "Brain Tumor MRI Dataset." Kaggle. Accessed: 2025-01-11. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset</p>

### Dataset
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary Tumor.  
- **Size:** 7,022 T1-weighted MRI scans.  
- **Structure:** Organized into `train/` (5,141 images post-split), `validation/` (571 images), and `test/` (1,311 images) directories, with subfolders for each class.  
- **Access:** Download from the Kaggle source and place in `data/`, or update `train_dir` and `test_dir` in `src/preprocess.py` to your dataset path.

<p align="center">
  <img src="Figures/Brain tumor dataset.png?raw=true" alt="Dataset Sample Image" width="100%">
</p>
<p align="center"><i>Caption: Sample MRI scan from the brain tumor dataset.</i></p>

### Run the Pipeline
#### Prepare the Environment:
- Ensure Python 3.8+ and dependencies are installed (see Requirements).  
- Verify GPU availability for faster training (CUDA-supported GPU recommended).

#### Download and Set Up the Dataset:
- Download the Brain Tumor MRI Dataset from Kaggle.  
- Place `train/` and `test/` folders in `data/`, or modify `train_dir` and `test_dir` in `src/preprocess.py` to point to your dataset location.

#### Run the Pipeline:
```bash
python src/main.py
```
This executes:  
- **Preprocessing:** `src/preprocess.py` applies resizing (224x224), augmentation (rotation, flipping, color jitter, etc.), and normalization to MRI scans.  
- **Model Definition:** `src/model.py` defines a placeholder CNN model for classification.  
- **Training:** `src/train.py` trains the model with early stopping and learning rate scheduling (20 epochs, batch size 32).  
- **Evaluation:** `src/evaluate.py` computes test accuracy and classification metrics.  
- **Visualization:** `src/visualize.py` generates plots for training/validation curves and confusion matrix.

#### Expected Outputs:
- **Model Weights:** Saved as `best_model.pth` (not included in the repository).  
- **Figures:** Saved in `figures/`:  
  - Confusion matrix (`confusion_matrix.png`).  
  - Training/validation curves (`training_plot.png`).  
- **Console Output:** Test accuracy, classification report, and epoch-wise training metrics.

#### Troubleshooting:
- Ensure sufficient disk space for processed data and model weights (`/kaggle/working/`).  
- If CUDA errors occur, set `device = torch.device("cpu")` in `src/main.py`.  
- Verify dataset paths and image formats (JPG, PNG, JPEG) in `src/preprocess.py`.

#### Requirements:
Install dependencies using Python 3.8+ and the provided `requirements.txt`:  
```bash
pip install -r requirements.txt
```
**requirements.txt:**  
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.19.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
```
Manual installation:  
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

### Repository Structure
- `src/`:  
  - `preprocess.py`: Data loading, augmentation, and DataLoader setup.  
  - `model.py`: Placeholder CNN model definition.  
  - `train.py`: Training loop with early stopping and scheduling.  
  - `evaluate.py`: Test set evaluation and metrics calculation.  
  - `visualize.py`: Plotting training/validation metrics and confusion matrix.  
  - `main.py`: Main script to orchestrate the pipeline.  
- `figures/`:  
  - `Model_Architecture.png`: Placeholder model architecture diagram.  
  - `Brain_tumor_dataset.png`: Sample MRI scan from the dataset.  
  - `confusion_matrix.png`: Confusion matrix for test predictions.  
  - `training_plot.png`: Training validation accuracy and loss curves.  
- `data/`: Placeholder for the dataset (not included; see Dataset section).  
- `requirements.txt`: Python dependencies.  
- `LICENSE`: MIT License (to be updated post-publication).  
- `README.md`: This file.

### Results

#### Model Performance
Evaluated on the test set (1,311 samples), the full HLGA framework achieves:  
- **Accuracy:** 99.83%  
- **Precision:** 99.93%  
- **Recall:** 99.86%  
- **F1-Score:** 99.89%

#### Classification Report
| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Glioma       | 1.00      | 0.9967 | 0.9983   | 300     |
| Meningioma   | 0.9967    | 0.9967 | 0.9967   | 306     |
| No Tumor     | 1.00      | 1.00   | 1.00     | 405     |
| Pituitary    | 1.00      | 1.00   | 1.00     | 300     |
| **Accuracy** |           |        | **0.9983** | 1,311   |
| **Macro Avg**| 0.9993    | 0.9983 | 0.9987   | 1,311   |
| **Weighted Avg** | 0.9993 | 0.9983 | 0.9983 | 1,311   |

**Note:** The placeholder model in `src/model.py` yields lower performance. The above metrics reflect the full HLGA framework, as reported in the paper.

## Findings
- **High Performance:** The HLGA framework surpasses the prior benchmark (99.22%) by integrating local and global features via adaptive fusion.  
- **Preprocessing Benefits:** Augmentations (rotation, flipping, color jitter) and normalization enhance feature extraction, reducing errors.  
- **Generalizability:** Validation on the Crystal Clean MRI dataset confirms robust performance across diverse imaging conditions.  
- **Ablation Insights:** The DFI's two-stage bottleneck attention is critical, minimizing misclassifications for complex cases like gliomas.

## Contact
For questions or collaboration inquiries, contact **Ahmad Muhammad** at <a href="mailto:ahmadjameel7171@gmail.com">ahmadjameel7171@gmail.com</a>.

## Acknowledgments
- The Kaggle Brain Tumor MRI Dataset providers.  
- The open-source community for tools like PyTorch and scikit-learn.  
- The Crystal Clean MRI dataset providers for external validation.

## Note
The model architecture (`HLGAModel`, `DynamicFusion`) and trained weights are withheld until the paper is published to protect novel contributions. The provided scripts under `src/` include a placeholder model to demonstrate the pipeline. Post-publication, the repository will be updated with:  
- Full model implementation in `src/model.py`.  
- Detailed architecture diagram (`figures/Model_Architecture.png`).  
- Trained weights (if permitted by the dataset license).  
For reviewer access to the full code or weights, contact <a href="mailto:ahmadjameel7171@gmail.com">ahmadjameel7171@gmail.com</a>. Updates will be announced post-publication.
