<div align="center">

# Machine Learning–Based Network Intrusion Detection System (IDS)

**Evidence 3 – UK Global Talent Visa (Technical Contribution)**  
**Author: Ibrahim Akintunde Akinyera**

[![Python](https://img.shields.io/badge/Built%20with-Python-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![ML](https://img.shields.io/badge/Machine%20Learning-RandomForest-0B7285)]()
[![Dataset](https://img.shields.io/badge/Dataset-NSL--KDD-5C7CFA)]()
[![Status](https://img.shields.io/badge/Repository-Production--Style-brightgreen)]()

</div>

---

## 1. Project Overview

This repository contains a production-style machine learning Intrusion Detection System (IDS) designed to classify network traffic as either:

- **Normal traffic (0)**
- **Attack or malicious activity (1)**

The project is built using the **NSL-KDD dataset**, a recognised benchmark dataset for evaluating intrusion detection models.  
It demonstrates end-to-end capability in cybersecurity analytics, data engineering, and machine learning model development.

This repository supports **Evidence 3 (Technical Contribution)** for the **UK Global Talent Visa**, showcasing:

- End-to-end ML engineering  
- Cybersecurity and risk analytics expertise  
- Work with real intrusion-detection datasets  
- Clear documentation and reproducible processes  

---

## 2. Key Features

- RandomForest model trained specifically for intrusion detection  
- Fully preprocessed and encoded training and testing datasets  
- Binary classification approach for operational simplicity  
- Clear separation between data, source code, and models  
- Ready for integration with dashboards or monitoring systems  
- Aligned to Tech Nation’s standards of technical contribution  

---

## 3. System Architecture

```
Raw NSL-KDD Data
       ↓
Data Ingestion and Preprocessing
       ↓
One-Hot Encoding and Label Conversion
       ↓
Model Training (RandomForest)
       ↓
Model Evaluation and Validation
       ↓
Saved Model (intrusion_model.pkl)
       ↓
Dashboard or API Layer (optional)
```

Supporting diagrams are available in:

```
docs/figures/
  ids_architecture.png
  ids_pipeline.png
  ids_dashboard.png
```

---

## 4. Dataset: NSL-KDD (Binary Classification)

The dataset has been cleaned and converted into a binary classification problem:

- `0` = Normal traffic  
- `1` = Attack (DoS, Probe, R2L, U2R grouped together)

The following CSV files are included:

```
data/
  nsl_kdd_train_binary.csv   (Training dataset with binary labels)
  nsl_kdd_test_binary.csv    (Testing dataset with binary labels)
  intrusion_model.pkl        (Trained RandomForest IDS model)
```

Each dataset contains fully encoded features and the `binary_label` target column.

---

## 5. Model Training and Evaluation

The model is trained using `RandomForestClassifier` with the following parameters:

```
n_estimators = 200
max_depth = None
random_state = 42
n_jobs = -1
```

### Training Steps:
1. Load binary-labeled CSV datasets  
2. Separate features and target label  
3. Train the RandomForest model  
4. Evaluate performance on the official NSL-KDD test set  
5. Export the trained model  

### Example Output (Binary Classification)
- **Accuracy:** ~0.78  
- **Task:** Distinguishing normal vs attack traffic  

This performance is typical for NSL-KDD without hyperparameter tuning.

---

## 6. How to Run the Project

### Installation

```
pip install -r requirements.txt
```

### Re-Train the Model

```
python src/train_ids_pipeline.py
```

### Optional: Run Dashboard

If you have implemented a Streamlit dashboard:

```
streamlit run src/dashboard_app.py
```

---

## 7. Repository Structure

```
network-intrusion-detection-ml/
  data/
    nsl_kdd_train_binary.csv
    nsl_kdd_test_binary.csv
    intrusion_model.pkl
  src/
    train_ids_pipeline.py
    data_preprocessing.py
    dashboard_app.py
    network_intrusion_detection.py
  docs/
    figures/
      ids_architecture.png
      ids_pipeline.png
      ids_dashboard.png
  TECH_NATION_EVIDENCE.md
  requirements.txt
  LICENSE
  README.md
```

---

## 8. Tech Nation Justification

This repository demonstrates:
- Advanced capability in machine learning model development  
- Technical work in cybersecurity and network risk analysis  
- End-to-end ownership of design, implementation, and documentation  
- Production-grade structure appropriate for a technical portfolio  
- Strong alignment with UK Global Talent “Technical Contribution” criteria  

---

## 9. Contact

**Ibrahim Akintunde Akinyera**  
GitHub: https://github.com/akinyeraakintunde  
LinkedIn: https://www.linkedin.com/in/ibrahimakinyera/  

---