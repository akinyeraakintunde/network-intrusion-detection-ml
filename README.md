<h1>Machine Learning–Based Network Intrusion Detection System (IDS)</h1>
<h3>Evidence 3 – UK Global Talent Visa (Technical Contribution)</h3>

<p>
This project is a complete, production-style machine learning system designed to detect network intrusions 
using supervised learning models. It includes data preprocessing, feature engineering, model training, 
model persistence, and a fully interactive dashboard built with <strong>Streamlit</strong>.
</p>

<p>
Developed end-to-end by <strong>Ibrahim Akintunde Akinyera</strong>, this work demonstrates advanced capability 
in machine learning, cybersecurity analytics, and Python-based system design.
</p>

<hr />

<h2>1. Project Overview</h2>

<p>
This system analyses structured network activity data and classifies each record as 
<strong>normal</strong> or <strong>potential intrusion</strong>.
</p>

<p>It includes the following components:</p>
<ul>
  <li>Data ingestion and preprocessing</li>
  <li>Feature selection and encoding</li>
  <li>Random Forest–based intrusion detection model</li>
  <li>Model training and evaluation pipeline</li>
  <li>Classification report and confusion matrix visualisation</li>
  <li>Persisted ML model for deployment</li>
  <li>Streamlit dashboard for real-time predictions from CSV uploads</li>
</ul>

<p>
This project demonstrates real-world applied ML skills relevant to cybersecurity and enterprise risk monitoring.
</p>

<hr />

<h2>2. Key Features</h2>
<ul>
  <li>End-to-end ML pipeline (preprocessing → training → evaluation → deployment)</li>
  <li>Random Forest classifier for robust performance</li>
  <li>Feature importance analysis for interpretability</li>
  <li>Interactive prediction dashboard using Streamlit</li>
  <li>Clean, modular Python codebase</li>
  <li>Realistic intrusion detection dataset</li>
  <li>Model export using <code>joblib</code> for production usage</li>
  <li>Clear ML metrics including precision, recall, and F1-score</li>
</ul>

<hr />

<h2>3. Repository Structure</h2>

<pre><code>├── data_preprocessing.py        # Cleans, encodes, and prepares the dataset
├── model_training.py            # Trains Random Forest model and evaluates performance
├── dashboard_app.py             # Streamlit dashboard for real-time predictions
├── dataset.csv                  # Raw dataset
├── dataset_clean.csv            # Cleaned, preprocessed dataset
├── intrusion_model.pkl          # Saved machine learning model (generated after training)
├── Untitled.ipynb               # Notebook with exploratory ML analysis
└── README.md
</code></pre>

<hr />

<h2>4. Machine Learning Pipeline</h2>

<h3>a) Data Preprocessing</h3>
<ul>
  <li>Handling missing values</li>
  <li>Encoding categorical features</li>
  <li>Train/test split</li>
  <li>Normalisation and cleaning</li>
</ul>

<h3>b) Model Training</h3>
<ul>
  <li>Random Forest classifier</li>
  <li>Optional hyperparameter tuning</li>
  <li>Feature importance extraction</li>
  <li>Accuracy and classification report generation</li>
</ul>

<h3>c) Model Evaluation</h3>
<ul>
  <li>Precision, recall, F1-score</li>
  <li>Confusion matrix</li>
  <li>Overall model accuracy</li>
</ul>

<h3>d) Model Persistence</h3>
<p>The best-performing model is saved using:</p>
<pre><code>joblib.dump(model, "intrusion_model.pkl")
</code></pre>

<h3>e) Deployment Dashboard</h3>
<p>The Streamlit dashboard allows:</p>
<ul>
  <li>Uploading CSV files</li>
  <li>Running predictions against the trained model</li>
  <li>Viewing results and predictions inside the UI</li>
  <li>Inspecting feature importance and basic analytics</li>
</ul>

<hr />

<h2>5. Screenshots (recommended)</h2>
<p>
You can add screenshots under <code>docs/figures/</code> to visually support this work, such as:
</p>
<ul>
  <li>ML pipeline diagram</li>
  <li>Confusion matrix</li>
  <li>Feature importance bar chart</li>
  <li>Streamlit dashboard interface</li>
</ul>

<hr />

<h2>6. Installation and Usage</h2>

<h3>Install dependencies</h3>
<pre><code>pip install -r requirements.txt
</code></pre>

<h3>Run preprocessing</h3>
<pre><code>python data_preprocessing.py
</code></pre>

<h3>Train the model</h3>
<pre><code>python model_training.py
</code></pre>

<h3>Launch the dashboard</h3>
<pre><code>streamlit run dashboard_app.py
</code></pre>

<hr />

<h2>7. Tech Nation Relevance (Evidence 3)</h2>
<p>
This project demonstrates:
</p>
<ul>
  <li>End-to-end machine learning engineering</li>
  <li>Application of AI to cybersecurity and intrusion detection</li>
  <li>Data preprocessing, feature engineering, and evaluation</li>
  <li>Deployment of a real-time prediction dashboard</li>
  <li>Independent technical leadership and original work</li>
</ul>
<p>
It provides strong evidence of expertise in digital technology and advanced applied machine learning.
</p>

<hr />

<h2>8. Author</h2>
<p><strong>Ibrahim Akintunde Akinyera</strong><br />
Machine Learning Engineer | Cybersecurity &amp; Risk Analytics<br />
GitHub: <a href="https://github.com/akinyeraakintunde">https://github.com/akinyeraakintunde</a><br />
Portfolio: <a href="https://akinyeraakintunde.github.io/Ibrahim-Akinyera">https://akinyeraakintunde.github.io/Ibrahim-Akinyera</a><br />
LinkedIn: <a href="https://www.linkedin.com/in/ibrahimakinyera">https://www.linkedin.com/in/ibrahimakinyera</a>
</p>

<hr />

<h2>9. Licence</h2>
<p>
This project is provided for assessment and demonstration purposes as part of a UK Global Talent Visa application.
</p>
