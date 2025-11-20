# Machine Learning–Based Network Intrusion Detection System (IDS)

### Evidence 3 – UK Global Talent Visa (Technical Contribution)  
**Author: Ibrahim Akintunde Akinyera**

This repository contains a production-style machine learning Intrusion Detection System (IDS) designed to classify network activity as either normal or malicious. It is built using the NSL-KDD dataset and demonstrates end-to-end capability across data preprocessing, feature engineering, model training, evaluation, and deployment.

The project forms part of my **Evidence 3 – Technical Contribution** for the UK Global Talent Visa and highlights my expertise in machine learning, cybersecurity analytics, and Python-based engineering.

---

## 1. Project Overview

The system analyses structured network traffic records and classifies each entry into:

- **0** – Normal traffic  
- **1** – Attack / intrusion  

The solution includes:

- Preprocessing and encoding of the NSL-KDD dataset  
- Training of a RandomForest-based intrusion detection model  
- Persisting a production-ready model (`intrusion_model.pkl`)  
- Optional interactive dashboard for live scoring  

This repository demonstrates practical technical work with real-world data and showcases end-to-end engineering ability.

---

## 2. Key Features

- Full ML training pipeline using scikit-learn  
- Cleaned and encoded binary dataset (normal vs attack)  
- RandomForest classifier trained on NSL-KDD  
- Clear separation of `data/`, `src/`, and `docs/`  
- Reproducible results through `train_ids_pipeline.py`  
- Dashboard for uploading CSV files and obtaining predictions  
- Documentation aligned with Tech Nation evidence requirements  

---

## 3. Dataset: NSL-KDD (Binary Reformatted Version)

The project uses the **NSL-KDD** dataset, one of the most widely used datasets for network intrusion detection research.

The dataset has been:

1. Loaded from raw NSL-KDD text files  
2. Cleaned, normalised, and converted into CSV  
3. Transformed into a binary classification problem:  
   - `0` = Normal  
   - `1` = Attack  
4. Fully encoded into numerical features  
5. Exported into ready-to-train CSV files:

```
data/
  nsl_kdd_train_binary.csv
  nsl_kdd_test_binary.csv
  intrusion_model.pkl
```

The target column for training and inference is:

```
binary_label
```

---

## 4. Repository Structure

```
network-intrusion-detection-ml/
  data/
    nsl_kdd_train_binary.csv
    nsl_kdd_test_binary.csv
    intrusion_model.pkl

  src/
    train_ids_pipeline.py         # End-to-end model training pipeline
    data_preprocessing.py         # Preprocessing and encoding logic
    model_training.py             # Supporting training logic
    dashboard_app.py              # Optional dashboard for predictions

  docs/
    figures/
      ids_architecture.png
      ids_pipeline.png
      ids_dashboard.png

  notebooks/
    exploration_intrusion_ids.ipynb

  README.md
  TECH_NATION_EVIDENCE.md
  requirements.txt
  LICENSE
```

This layout mirrors production-grade ML repositories and enables easy assessment, reproducibility, and deployment.

---

## 5. Machine Learning Pipeline

### a) Data Preprocessing  
Implemented in `data_preprocessing.py` and the exploratory notebook:

- Loading of NSL-KDD datasets  
- Handling missing values and inconsistencies  
- Categorical encoding using one-hot encoding  
- Conversion to binary labels  
- Export of final training and testing datasets  

### b) Model Training  
`train_ids_pipeline.py` performs:

- Feature and target separation  
- Training a RandomForest classifier with 200 estimators  
- Model evaluation on the official NSL-KDD test split  
- Exporting model file using `joblib.dump()`  

### c) Model Persistence  
The trained model is saved as:

```
data/intrusion_model.pkl
```

### d) Prediction Dashboard  
`dashboard_app.py` provides:

- CSV upload interface  
- Automatic preprocessing of uploaded data  
- Live predictions using `intrusion_model.pkl`  
- Usable for security monitoring or analyst workflows  

---

## 6. Running the Project

### Step 1: Install Dependencies
```
pip install -r requirements.txt
```

### Step 2: Train or Re-train the Model
```
python src/train_ids_pipeline.py
```

### Step 3: Launch the Dashboard (Optional)
```
streamlit run src/dashboard_app.py
```

---

## 7. Tech Nation Relevance – Evidence 3 (Technical Contribution)

This repository demonstrates:

**1. End-to-end engineering ability**  
I designed and implemented the entire machine learning pipeline independently, including preprocessing, model development, and deployment components.

**2. Cybersecurity expertise**  
The work applies ML techniques directly to intrusion detection challenges using a recognised security dataset.

**3. Production-level practices**  
The repository structure reflects real-world engineering, with clean separation of data, code, and model artefacts.

**4. Reproducibility and clarity**  
Assessors can run the training script and obtain identical results using the included datasets and model pipeline.

**5. Demonstrated impact**  
The dashboard shows how the model can be deployed for operational use in security monitoring environments.

---

## 8. Author

**Ibrahim Akintunde Akinyera**  
Machine Learning Engineer | Cybersecurity & Risk Analytics  

GitHub: https://github.com/akinyeraakintunde  
Portfolio: https://akinyeraakintunde.github.io/Ibrahim-Akinyera  
LinkedIn: https://www.linkedin.com/in/ibrahimakinyera/