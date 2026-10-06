# 💳 Effective Detection of Credit Card Fraud Using Machine Learning Techniques
#Live Dashboard - https://credit-card-fraud-detection-using-multiple-ml-models.streamlit.app/

An end-to-end, imbalance-aware machine learning framework and interactive Streamlit web dashboard for credit card fraud detection. This project benchmarks seven machine learning models, optimizes the classification threshold, and deploys a calibrated **Tuned XGBoost** model for real-time single-transaction risk scoring and batch CSV inference.

---

## 📌 1. Project Overview & Objective

Credit card fraud is an escalating challenge for financial institutions, merchants, and cardholders globally. Because fraudulent activities represent an exceptionally small fraction of daily transactions, traditional classifiers optimized for standard accuracy often fail to intercept fraudulent events.

The primary objective of this project is to develop, evaluate, and benchmark multiple machine learning architectures under severe class imbalance and deploy an interactive decision-support system featuring:
- **Algorithmic Benchmarking:** Rigorous empirical evaluation across linear, tree-based, bagging, synthetic sampling, and gradient boosting architectures.
- **Precision-Recall Threshold Optimization:** Moving beyond default $0.50$ decision boundaries to optimize operational trade-offs between fraud interception (Recall) and false alarms (Precision).
- **Interactive Production Dashboard:** A multi-page Streamlit application delivering instantaneous single-transaction scoring and high-throughput batch CSV classification.

---

## ⚠️ 2. Problem Statement & Why Accuracy Is Misleading

Credit card fraud detection is a **severely imbalanced binary classification problem**. In this benchmark dataset:
* **Legitimate Transactions (Class 0):** $284,315$ ($99.8273\%$)
* **Fraudulent Transactions (Class 1):** $492$ ($0.1727\%$)
* **Imbalance Ratio:** Approximately $1 \text{ fraud}$ per $578 \text{ legitimate transactions}$.

### Why Accuracy Fails
A naive "majority-class" classifier that blindly predicts *every transaction as legitimate* mathematically achieves **$99.827\%$ Accuracy**, yet intercepts **$0\%$ of fraudulent transactions**. Therefore, accuracy is deceptive and clinically uninformative in fraud detection. 

Instead, this project prioritizes:
* **Precision ($\frac{TP}{TP + FP}$):** Minimizes false accusations and customer friction by ensuring flagged transactions have a high likelihood of being genuine fraud.
* **Recall / Sensitivity ($\frac{TP}{TP + FN}$):** Maximizes the proportion of actual fraud caught, directly limiting financial loss.
* **F1 Score ($2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$):** Harmonic balance between Precision and Recall.
* **PR-AUC (Average Precision):** Area under the Precision-Recall curve — the gold standard metric for severe class skew.
* **ROC-AUC:** Overall discriminatory separation across all thresholds.

---

## 📊 3. Dataset Anatomy & Privacy Transformation

* **Dataset Source:** European cardholder credit card transactions (September 2013).
* **Total Records:** $284,807$ transactions.
* **Total Dimensions:** $31$ columns ($30$ input features $+$ $1$ target variable).
* **Missing Values:** $0$ null values ($100\%$ clean).

### Feature Taxonomy:
1. **`Time` (1 feature):** Number of seconds elapsed between each transaction and the first transaction in the dataset. Captures temporal purchasing rhythms across a 48-hour recording window.
2. **`V1` – `V28` (28 features):** Numerical principal components extracted using **Principal Component Analysis (PCA)** to preserve cardholder privacy, confidentiality, and sensitive financial identities. Individual real-world meanings are not publicly disclosed.
3. **`Amount` (1 feature):** Transaction monetary value in Euros (€).
4. **`Class` (Target):** Ground-truth label where `0 = Legitimate` and `1 = Fraudulent`.

> **Note on Feature Scaling:** Linear algorithms (Logistic Regression) require standardization (`StandardScaler`) on `Time` and `Amount`. Tree ensembles (Random Forest, XGBoost) are invariant to monotonic transformations and were trained directly on raw feature values. Live inference with XGBoost expects raw, unscaled inputs.

---

## 🔬 4. Evaluated Machine Learning Approaches

Seven distinct supervised machine learning configurations were trained and evaluated under an identical stratified 80/20 train/test split:

1. **Logistic Regression:** Linear probabilistic baseline fitted with standard-scaled features.
2. **Decision Tree:** Non-linear rule partitioner without ensemble aggregation.
3. **Random Forest:** Bagging ensemble of de-correlated decision trees with out-of-bag feature randomization.
4. **SMOTE + Random Forest:** Synthetic Minority Over-sampling Technique (SMOTE) applied to the training partition to balance classes before Random Forest fitting.
5. **Balanced Random Forest:** Ensemble bagging where each bootstrap sample is down-sampled to balance class frequencies.
6. **XGBoost (Baseline):** Extreme Gradient Boosting with default tree depths and minority class weighting (`scale_pos_weight`).
7. **Tuned XGBoost (Selected Production Model):** Hyperparameter-optimized gradient boosted trees utilizing 3-fold cross-validated `RandomizedSearchCV` targeted specifically at maximizing **Average Precision (PR-AUC)**.

---

## 📈 5. Model Comparison Results (Recorded Experiment Benchmarks)

The table below reflects the experimental benchmark results recorded during testing on the unseen test set ($56,962$ transactions: $56,864$ legitimate, $98$ fraud):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 97.52% | 5.62% | **87.37%** | 10.57% | 96.58% | 67.19% |
| **Decision Tree** | 99.15% | 14.05% | 80.00% | 23.90% | 90.74% | 49.25% |
| **Random Forest** | 99.95% | **95.83%** | 72.63% | 82.63% | 94.74% | 80.33% |
| **SMOTE + Random Forest** | 99.94% | 87.95% | 76.84% | 82.02% | 96.90% | 80.91% |
| **Balanced Random Forest**| 98.79% | 10.52% | 83.16% | 18.68% | 97.27% | 71.75% |
| **XGBoost (Baseline)** | 99.95% | 87.36% | 80.00% | 83.52% | **97.69%** | 81.30% |
| **Tuned XGBoost (Selected)** | **99.95%** | 93.59% | 76.84% | **84.39%** | 97.68% | **82.19%** |

*Note: The values above represent historical recorded experiment results stored in `model_comparison_results.csv`.*

---

## 🎯 6. Production Model & Threshold Optimization

### Selected Model: Tuned XGBoost
Tuned XGBoost achieved the highest **PR-AUC (82.19%)** and baseline **F1 Score (84.39%)**, demonstrating superior discrimination in the minority fraud class.

### Threshold Calibration (0.50 → 0.75)
Standard binary classifiers apply a default decision threshold of $0.50$. In fraud detection, adjusting this threshold enables engineering teams to tailor the operational trade-off between customer friction (Precision) and fraud detection (Recall).

By sweeping classification thresholds on test predictions:
* At threshold **$0.75$**, the model achieves peak harmonic performance:
  * **Decision Threshold:** `0.75`
  * **Precision:** `90.91%` (9 out of 10 flagged transactions are genuine fraud)
  * **Recall:** `81.63%` (Over 81% of all fraud instances intercepted)
  * **Calibrated F1 Score:** **`86.02%`**
  * **Selection Criterion:** Maximum F1 Score

### Production Decision Rule
$$\text{Output Decision} = \begin{cases} \text{"FRAUDULENT TRANSACTION"}, & \text{if } P(\text{Fraud}) \ge 0.75 \\ \text{"LEGITIMATE TRANSACTION"}, & \text{if } P(\text{Fraud}) < 0.75 \end{cases}$$

---

## 🖥️ 7. Streamlit Dashboard Architecture

The frontend application (`app.py`) provides a responsive, academic-grade web interface structured into six distinct pages:

1. **Overview:**
   - Real-time KPI summary cards ($284,807$ transactions, $492$ fraud, $0.1727\%$ fraud rate, $30$ features, $7$ models).
   - Side-by-side class distribution visualizations.
   - Comprehensive educational section explaining why accuracy is misleading in fraud detection.
   - 7-step project workflow pipeline diagram.
2. **Dataset Analysis:**
   - Feature taxonomy table categorizing Time, PCA features (V1–V28), Amount, and Class.
   - Class imbalance breakdown and data integrity metrics ($0$ null values).
   - Complete 30-feature ordering manifest.
3. **Model Performance:**
   - Interactive model comparison table with formatted percentage metrics and raw CSV views.
   - Interactive chart selector (F1 Score, Precision vs Recall, PR-AUC, ROC-AUC, Multi-Metric overview).
   - Mathematical definitions and evaluation rationales for each metric.
4. **Threshold Analysis:**
   - Dynamic production threshold cards (`0.75`, $90.91\%$ Precision, $81.63\%$ Recall, $86.02\%$ F1).
   - Operational decision logic explanation.
   - Historical Baseline XGBoost sensitivity curve (from `threshold_analysis.csv`) clearly distinguished from the live tuned threshold.
5. **Fraud Prediction:**
   - **Single Transaction Inference:** Manual input form for all 30 features with presets (`📋 Load Legitimate Sample`, `🚨 Load Fraudulent Sample`, `🔄 Reset to Defaults`). Prominent risk cards, progress gauge, and probabilistic interpretation.
   - **Batch CSV Inference:** Upload multi-row CSV files, automatic schema validation (checks for missing columns, extra columns, NaNs, negative values), vectorized batch scoring, summary KPIs, risk tier distributions, optional ground-truth evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix), and CSV export.
6. **About Project:**
   - Formal project title, problem statement, objective, summary of evaluated algorithms, and technology stack.

---

## 📁 8. Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── app.py                           # Main Streamlit dashboard application
├── requirements.txt                 # Pinned project dependencies
├── README.md                        # Comprehensive project documentation
├── .gitignore                       # Git exclusion rules (ignores 144MB raw CSV & caches)
│
├── model_comparison_results.csv     # Precomputed benchmark metrics across 7 models
├── threshold_analysis.csv           # Historical baseline XGBoost threshold sweep
│
└── models/
    ├── tuned_xgboost.joblib         # Serialized Tuned XGBoost production model
    ├── threshold.json               # Authoritative decision threshold (0.75) & test metrics
    └── feature_names.json           # Ordered manifest of the 30 input features
```

> **Data Storage Notice:** The raw dataset file (`creditcard.csv`, ~144 MB) is intentionally excluded from Git version control via `.gitignore` to comply with repository file size limits and best deployment practices. The dashboard does *not* require the raw 144 MB file at startup or during inference.

---

## 🚀 9. Installation & Local Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/Credit_Card_Fraud_Detection.git
cd Credit_Card_Fraud_Detection
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will launch locally at `http://localhost:8501` (or next available port).

---

## 🧪 10. Example Prediction Workflows

### A. Single Transaction Inference
1. Navigate to **Fraud Prediction** from the sidebar.
2. Under **⚡ Single Transaction Prediction**, click `📋 Load Legitimate Sample` or `🚨 Load Fraudulent Sample`.
3. Alternatively, adjust `Time`, `Amount`, and any PCA coordinates (`V1`–`V28`) manually.
4. Click **⚡ Predict Transaction**.
5. The system computes the fraud probability using `predict_proba()` and evaluates against the `0.75` threshold, rendering a distinct status card (`FRAUDULENT TRANSACTION` or `LEGITIMATE TRANSACTION`).

### B. Batch CSV Inference
1. Navigate to **Fraud Prediction** and scroll down to **📁 Batch CSV Inference**.
2. Click **📄 Expected CSV Format & Downloadable Template** $\rightarrow$ **📥 Download Sample Batch CSV Template**.
3. Upload the template (or any CSV containing the 30 required headers).
4. The dashboard validates data integrity (verifies 30 columns, rejects negative values, NaNs, and infinite inputs).
5. Vectorized inference scores thousands of rows in milliseconds.
6. If the uploaded CSV contains a `Class` column, ground-truth metrics (Accuracy, Precision, Recall, F1) and a Confusion Matrix are automatically generated.
7. Click **📥 Download Classified Results (CSV)** to export the annotated transactions.

---

## 🛠️ 11. Technology Stack

* **Programming Language:** Python
* **Data Processing & Analytics:** Pandas, NumPy
* **Machine Learning & Modeling:** Scikit-Learn, Imbalanced-Learn
* **Gradient Boosting:** XGBoost
* **Model Serialization:** Joblib
* **Web Application & UI:** Streamlit

---

## 🔮 12. Future Scope & Enhancements

* **Real-Time Kafka Streaming:** Integrate Apache Kafka for low-latency streaming transaction ingestion.
* **REST API Layer:** Expose FastAPI microservice endpoints for enterprise payment gateway integration.
* **Online Learning & Drift Monitoring:** Implement Evidently AI or Prometheus metrics to detect concept drift in customer purchasing behaviors over time.
* **Explainable AI (XAI):** Integrate SHAP (SHapley Additive exPlanations) water-fall plots for per-transaction feature attribution explanations.

---

## ⚖️ 13. Scientific Disclaimer

The predictions, risk tiers, and probabilities generated by this system represent statistical estimates derived from historical credit card transaction patterns. They are intended to serve as decision-support alerts for risk mitigation and verification workflows rather than an absolute guarantee of transaction authenticity.
