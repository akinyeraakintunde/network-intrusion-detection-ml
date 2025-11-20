# Tech Nation Evidence – Evidence 3  
## Machine Learning–Based Network Intrusion Detection System (IDS)  
**Applicant:** Ibrahim Akintunde Akinyera  
**Criterion:** Optional – Proven Technical Expertise and Contribution in Digital Technology

---

## 1. Summary of Evidence

This evidence demonstrates my ability to design and implement an **end-to-end machine learning system** for **network intrusion detection**, combining:

- Data preprocessing and feature engineering  
- Supervised machine learning model development  
- Evaluation using industry-standard metrics  
- Model persistence for deployment  
- A real-time, interactive dashboard built with **Streamlit**

The work is contained in this repository and was developed **entirely by me, Ibrahim Akintunde Akinyera**.

---

## 2. Problem Description

Modern networks generate large volumes of traffic and log data. Manually detecting suspicious or malicious activity is:

- Slow  
- Inconsistent  
- Difficult to scale  

Traditional rule-based intrusion detection systems (IDS) struggle to keep up with evolving attack patterns and noisy data.

There is a clear need for **data-driven, machine learning–based intrusion detection tools** that can classify network events as normal or suspicious in a reproducible, explainable way.

---

## 3. Solution Overview

I developed a **Machine Learning–based IDS** that:

- Ingests network activity data from CSV files  
- Cleans and preprocesses features (encoding, handling missing values)  
- Trains a **Random Forest classifier** to distinguish normal vs. intrusion records  
- Evaluates the model using accuracy, precision, recall and F1-score  
- Saves the best model as a deployable artefact (`intrusion_model.pkl`)  
- Exposes the model via an interactive **Streamlit dashboard** for real-time prediction

The solution is implemented entirely in **Python**, using libraries such as:

- `pandas`, `numpy`  
- `scikit-learn`  
- `matplotlib`, `seaborn`  
- `joblib`  
- `streamlit`

---

## 4. Technical Architecture

The system is organised into clear, modular components:

- `data_preprocessing.py` – data loading, cleaning, encoding and train/test split  
- `model_training.py` – model training, evaluation, feature importance, and persistence  
- `dashboard_app.py` – Streamlit-based web UI for uploading data and viewing predictions  
- `dataset.csv` / `dataset_clean.csv` – raw and cleaned datasets  
- `intrusion_model.pkl` – exported machine learning model  

**Architecture flow:**

1. **Data Layer**  
   - Raw dataset (`dataset.csv`)  
   - Cleaned dataset (`dataset_clean.csv`)

2. **Preprocessing Layer**  
   - Handling missing values  
   - Encoding categorical variables  
   - Splitting into training and test sets  

3. **Model Layer**  
   - Random Forest classifier  
   - Model training and hyperparameters  
   - Metrics: accuracy, precision, recall, F1-score  
   - Confusion matrix and feature importance

4. **Deployment Layer**  
   - `intrusion_model.pkl` saved with `joblib`  
   - `dashboard_app.py` exposes model through a Streamlit interface  

5. **User Interaction Layer**  
   - User uploads CSV  
   - System runs predictions using the trained model  
   - Dashboard displays classification outputs and basic analytics  

---

## 5. My Role and Technical Contribution

I personally:

- Designed the overall system architecture  
- Wrote all Python modules in this repository  
- Implemented the data preprocessing and feature engineering steps  
- Selected and trained the Random Forest classifier  
- Evaluated the model with appropriate metrics and diagnostics  
- Implemented model persistence with `joblib`  
- Built the Streamlit dashboard for real-time predictions  

No external developers contributed to the core engineering work.  
This is entirely my own technical design and implementation.

---

## 6. Impact and Relevance

This project is relevant to Tech Nation’s digital technology criteria because it:

- Demonstrates **advanced applied machine learning skills**  
- Shows **cybersecurity and risk analytics** capability through intrusion detection  
- Proves I can build **end-to-end systems** (data → model → deployment → UI)  
- Uses **modern Python tooling** consistent with industry practice  
- Is structured in a way that can be extended to real-world environments

The work is directly relevant to roles such as:

- Machine Learning Engineer  
- Security Data Scientist  
- Cybersecurity Analytics Engineer  
- Risk Intelligence Engineer  

---

## 7. Supporting Materials in This Repository

This repository contains:

- **Source Code**  
  - `data_preprocessing.py` – preprocessing pipeline  
  - `model_training.py` – model training and evaluation  
  - `dashboard_app.py` – Streamlit app for predictions  
  - `Untitled.ipynb` – exploration and experimentation notebook  

- **Datasets**  
  - `dataset.csv` – raw network dataset  
  - `dataset_clean.csv` – cleaned and processed dataset  

- **Model Artefact**  
  - `intrusion_model.pkl` – saved trained model  

- **Documentation**  
  - `README.md` – high-level project overview  
  - `TECH_NATION_EVIDENCE.md` – this evidence narrative  

Optionally, I may also add:

- Confusion matrix and feature importance plots under `docs/figures/`  
- Screenshots of the Streamlit dashboard  

---

## 8. Relationship to Other Work (Supporting Evidence)

In addition to this project, I have previously built:

- A **Twitter Topic Classifier (NLP)** based on transformer models (BERT), used in my MSc Data Science dissertation.  
  That work is provided as a **supporting document**, demonstrating a longer track record in machine learning and NLP.

Together, these projects show a consistent, multi-year focus on **applied machine learning, cybersecurity, and risk analytics**.

---

## 9. Repository Link

(If public)

GitHub: `https://github.com/akinyeraakintunde/network-intrusion-detection-ml`

---

## 10. Conclusion

This evidence demonstrates my ability to:

- Design and implement a complete ML-based intrusion detection system  
- Combine data engineering, model development, and deployment in a single solution  
- Apply AI to cybersecurity and risk-related domains  
- Produce high-quality, well-structured Python code and documentation  

It provides strong support for the **Optional Criterion: Proven Technical Expertise and Contribution in Digital Technology** within the UK Global Talent Visa framework.
