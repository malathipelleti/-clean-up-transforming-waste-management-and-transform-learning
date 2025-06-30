# HematoVision: Automated Blood Cell Subtype Classification Using Deep Learning

## 📌 Project Overview

**HematoVision** is a deep learning-based diagnostic solution designed to classify different blood cell subtypes from microscopic images with high accuracy. Built using convolutional neural networks and transfer learning techniques, this project aims to support medical professionals with faster and more reliable hematological analysis.

The complete pipeline includes data preprocessing, model training using `EfficientNetB0`, performance evaluation, and model interpretability using Grad-CAM.

---

## 🚀 Key Features

- ✅ End-to-end pipeline from data loading to deployment-ready predictions
- 🧠 Utilizes **EfficientNetB0** with **transfer learning**
- 📊 Precision, Recall, F1-Score and Confusion Matrix reporting
- 🖼️ Visualization with Grad-CAM to interpret model predictions
- 📁 Organized modular code in Jupyter Notebook format

---

## 📂 Dataset

The dataset used contains labeled blood cell images across multiple subtypes, sourced from publicly available and medically approved collections. Each image is categorized into one of the major white blood cell classes:
- **Eosinophils**
- **Lymphocytes**
- **Monocytes**
- **Neutrophils**

> 📎 Dataset is assumed to be locally present in `./dataset` directory. Please update the path accordingly if used elsewhere.

---

## 🧰 Tech Stack

| Tool / Library     | Purpose                                |
|--------------------|----------------------------------------|
| Python 3.x         | Core programming language              |
| TensorFlow & Keras | Deep learning and model architecture   |
| NumPy, Pandas      | Data manipulation                      |
| Matplotlib, Seaborn| Data visualization                     |
| Scikit-learn       | Evaluation metrics                     |

---

## 🛠️ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/hematovision-bloodcell-classification.git
   cd hematovision-bloodcell-classification
2. Install Dependencies
    It's recommended to use a virtual environment.
    pip install -r requirements.txt
3. Run the Notebook
    Open and execute the notebook in a Jupyter environment:
     jupyter notebook classify-blood-cell-subtypes-all-process.ipynb

Results Summary
---------------

Validation Accuracy: ~92%

Model Architecture: EfficientNetB0 with custom dense layers

Interpretability: Grad-CAM shows key focus regions in predictions

Confusion Matrix: Shows strong classification accuracy across all classes

✨ The model is capable of real-time inference and demonstrates strong generalization on unseen data.

Model Evaluation
----------------

Precision, Recall, F1-score for each class are calculated

Confusion Matrix used to visualize prediction accuracy

Grad-CAM visualizations for post-hoc interpretability

Folder Structure
----------------
├── classify-blood-cell-subtypes-all-process.ipynb
├── dataset/
│   ├── Eosinophil/
│   ├── Lymphocyte/
│   ├── Monocyte/
│   └── Neutrophil/
├── models/
├── outputs/
└── README.md
















   
