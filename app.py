"""
Credit Card Fraud Detection - Streamlit Web Application
Final-Year Engineering Project Dashboard

Authoritative Model: models/tuned_xgboost.joblib
Authoritative Threshold: models/threshold.json (0.75)
Feature Specifications: models/feature_names.json (30 features)
"""

import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection | ML Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Global Styling (Professional Academic / Engineering Theme)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Metric Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        color: #f8fafc;
        min-height: 120px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 14px;
    }
    .kpi-title {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38bdf8;
        line-height: 1.2;
    }
    .kpi-caption {
        font-size: 0.78rem;
        color: #cbd5e1;
        margin-top: 4px;
    }

    /* Banners & Callouts */
    .info-banner {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 14px 0 20px 0;
        color: #e2e8f0;
    }
    .warning-banner {
        background: rgba(69, 26, 3, 0.4);
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 6px;
        margin: 14px 0 20px 0;
        color: #fef3c7;
    }

    /* Badges */
    .badge-success {
        background-color: #065f46;
        color: #34d399;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-live {
        background-color: #0369a1;
        color: #7dd3fc;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-ref {
        background-color: #4b5563;
        color: #e5e7eb;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Workflow Pipeline Steps */
    .workflow-step {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 14px;
        text-align: center;
        color: #f1f5f9;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .workflow-arrow {
        text-align: center;
        color: #38bdf8;
        font-size: 1.3rem;
        line-height: 42px;
    }

    /* Prediction Result Cards */
    .result-card-fraud {
        background: linear-gradient(135deg, #450a0a 0%, #1f0707 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px 28px;
        color: #fef2f2;
        margin: 20px 0;
        box-shadow: 0 10px 20px -3px rgba(239, 68, 68, 0.3);
    }
    .result-card-legit {
        background: linear-gradient(135deg, #022c22 0%, #031e17 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px 28px;
        color: #ecfdf5;
        margin: 20px 0;
        box-shadow: 0 10px 20px -3px rgba(16, 185, 129, 0.3);
    }
    .result-tag {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .result-headline {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        margin-bottom: 18px;
    }
    .result-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        background: rgba(0, 0, 0, 0.35);
        padding: 16px 20px;
        border-radius: 8px;
    }
    .result-sub-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
    }
    .result-sub-val {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cached Loaders (Zero Model Training, Safe, Fast Startup)
# -----------------------------------------------------------------------------
@st.cache_data
def load_metadata():
    """Loads authoritative feature names and threshold settings dynamically from models/."""
    feature_info = {}
    threshold_info = {}

    if os.path.exists("models/feature_names.json"):
        with open("models/feature_names.json", "r", encoding="utf-8") as f:
            feature_info = json.load(f)

    if os.path.exists("models/threshold.json"):
        with open("models/threshold.json", "r", encoding="utf-8") as f:
            threshold_info = json.load(f)

    return feature_info, threshold_info

@st.cache_resource
def load_tuned_xgboost_model():
    """
    Cached loader for the serialized Tuned XGBoost model.
    Loads once per application cache lifecycle without retraining.
    Validates model existence and expected 30-feature input dimension.
    """
    model_path = "models/tuned_xgboost.joblib"
    if not os.path.exists(model_path):
        return None, f"Model file '{model_path}' not found."
    try:
        model = joblib.load(model_path)
        reported_features = getattr(model, "n_features_in_", None)
        if reported_features != 30:
            return None, f"Model input feature count mismatch: expected 30, got {reported_features}."
        return model, None
    except Exception as e:
        return None, f"Error loading model binary: {str(e)}"

@st.cache_data
def load_model_comparison_results():
    """Loads precomputed model comparison metrics without altering the file."""
    if os.path.exists("model_comparison_results.csv"):
        return pd.read_csv("model_comparison_results.csv")
    return pd.DataFrame()

@st.cache_data
def load_threshold_analysis_results():
    """Loads historical baseline threshold evaluation table."""
    if os.path.exists("threshold_analysis.csv"):
        return pd.read_csv("threshold_analysis.csv")
    return pd.DataFrame()

# Load precomputed benchmarks & metadata
feature_info, threshold_info = load_metadata()
model_results_df = load_model_comparison_results()
threshold_df = load_threshold_analysis_results()

# Authoritative values from models/ directory
saved_threshold = float(threshold_info.get("threshold", 0.75))
saved_precision = float(threshold_info.get("precision", 0.9090909090909091))
saved_recall = float(threshold_info.get("recall", 0.8163265306122449))
saved_f1 = float(threshold_info.get("f1_score", 0.8602150537634409))
saved_features = feature_info.get("features", [])
num_features = feature_info.get("number_of_features", 30)

# Expected feature ordering validation
EXPECTED_FEATURES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9",
    "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Downloadable Sample CSV Template (Safe In-Memory Mock Data)
SAMPLE_CSV_TEMPLATE = (
    "Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount,Class\n"
    "0.0,-1.3598,-0.0728,2.5363,1.3782,-0.3383,0.4624,0.2396,0.0987,0.3638,0.0908,-0.5516,-0.6178,-0.9914,-0.3112,1.4682,-0.4704,0.2080,0.0258,0.4040,0.2514,-0.0183,0.2778,-0.1105,0.0669,0.1285,-0.1891,0.1336,-0.0211,149.62,0\n"
    "406.0,-2.3122,1.9520,-1.6099,3.9979,-0.5222,-1.4265,-2.5374,1.3917,-2.7701,-2.7723,3.2020,-2.8999,-0.5952,-4.2893,0.3897,-1.1407,-2.8301,-0.0168,0.4170,0.1269,0.5172,-0.0350,-0.4652,0.3202,0.0445,0.1778,0.2611,-0.1433,0.00,1\n"
    "1500.0,1.1584,0.1588,0.4132,0.6148,-0.3312,-0.6189,0.0125,-0.0824,0.1205,-0.1821,0.1542,0.4851,0.5312,-0.0124,-0.1084,0.2451,-0.3125,0.1124,0.0152,-0.0521,-0.1842,-0.4125,0.0841,-0.0214,0.2184,0.1125,-0.0152,0.0214,25.50,0\n"
)

# -----------------------------------------------------------------------------
# Sidebar Navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💳 Fraud Detection ML")
    st.caption("Final-Year Engineering Project")
    st.divider()

    selected_page = st.radio(
        "Dashboard Navigation",
        [
            "Overview",
            "Dataset Analysis",
            "Model Performance",
            "Threshold Analysis",
            "Fraud Prediction",
            "About Project"
        ],
        index=0
    )

    st.divider()
    st.markdown("### ⚙️ Production Artifacts")
    st.markdown("**Production Model:** `Tuned XGBoost`")
    st.markdown(f"**Live Threshold:** `{saved_threshold:.2f}`")
    st.markdown(f"**Input Dimension:** `{num_features} Features`")
    st.markdown("<span class='badge-live'>Active Pipeline</span>", unsafe_allow_html=True)

    st.divider()
    st.caption("Benchmark dataset: European Cardholders (284,807 transactions).")


# =============================================================================
# 1. OVERVIEW
# =============================================================================
if selected_page == "Overview":
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown(
        "A rigorous machine learning framework designed for detecting fraudulent credit card transactions "
        "under severe class imbalance, featuring algorithmic benchmarking and precision-recall threshold calibration."
    )

    st.markdown("""
    <div class="info-banner">
        <strong>Executive Project Summary:</strong> Credit card fraud detection is a critical financial cybersecurity challenge. 
        This system evaluates 7 machine learning architectures and integrates an optimized gradient-boosted decision tree 
        specifically calibrated to minimize financial loss while protecting genuine cardholder transactions.
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards (2 rows of 3)
    st.subheader("📌 Key Project & Dataset Metrics")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-title">Total Transactions</div>
            <div class="kpi-value">284,807</div>
            <div class="kpi-caption">Complete European cardholder benchmark</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-title">Legitimate Transactions</div>
            <div class="kpi-value">284,315</div>
            <div class="kpi-caption">Class 0 (99.8273% of total data)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-title">Fraudulent Transactions</div>
            <div class="kpi-value">492</div>
            <div class="kpi-caption">Class 1 (Ground-truth positive cases)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-title">Fraud Rate</div>
            <div class="kpi-value">0.1727%</div>
            <div class="kpi-caption">Severe class imbalance (~1 in 578)</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Input Features</div>
            <div class="kpi-value">{num_features}</div>
            <div class="kpi-caption">Time, Amount, V1–V28 (PCA transformed)</div>
        </div>
        """, unsafe_allow_html=True)

    with col6:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-title">Models Evaluated</div>
            <div class="kpi-value">7</div>
            <div class="kpi-caption">Linear, Bagging, SMOTE & Boosting</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Class Imbalance Visualization & Analysis
    st.subheader("⚖️ Class Distribution & Severe Imbalance")
    
    chart_col, exp_col = st.columns([1, 1])

    with chart_col:
        st.markdown("##### Transaction Class Breakdown")
        dist_data = pd.DataFrame({
            "Class": ["Legitimate (0)", "Fraudulent (1)"],
            "Count": [284315, 492]
        }).set_index("Class")
        st.bar_chart(dist_data, y="Count", color="#38bdf8")
        st.caption("Visual confirmation: 492 fraud instances are visually dwarfed by 284,315 legitimate transactions.")

    with exp_col:
        st.markdown("##### Distribution Summary")
        summary_table = pd.DataFrame({
            "Transaction Type": ["Legitimate (Class 0)", "Fraudulent (Class 1)", "Total"],
            "Transactions": ["284,315", "492", "284,807"],
            "Percentage": ["99.8273%", "0.1727%", "100.0%"],
            "Imbalance Ratio": ["578 : 1", "1 in 578", "-"]
        })
        st.dataframe(summary_table, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="warning-banner">
            <strong>Critical Imbalance:</strong> For every single fraudulent transaction, there are approximately 
            <strong>578 legitimate transactions</strong>. Standard classification algorithms tend to heavily bias toward 
            the majority class unless counteracted by sampling, specialized cost-weighting, and threshold tuning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Why Accuracy Is Misleading
    st.subheader("⚠️ Why Accuracy Is Misleading in Fraud Detection")
    st.write(
        "Because fraudulent transactions represent only about **0.173%** of all transactions, a naive model that simply "
        "classifies *every single transaction as legitimate (Class 0)* would achieve a deceptive **99.827% Accuracy** "
        "while failing to intercept a single fraudulent event."
    )
    
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.info("**Precision**\n\nMinimizes false accusations. Flags transactions with high confidence, avoiding customer friction.")
    with metric_cols[1]:
        st.info("**Recall**\n\nMaximizes fraud interception. Directly limits financial losses by capturing true positive frauds.")
    with metric_cols[2]:
        st.info("**F1 Score**\n\nHarmonic mean of Precision and Recall. Essential for balancing false positives and false negatives.")
    with metric_cols[3]:
        st.info("**PR-AUC**\n\nArea under Precision-Recall curve. The definitive gold-standard metric for severely skewed data.")

    st.markdown("---")

    # Concise Project Workflow
    st.subheader("🔄 End-to-End Project Workflow")
    st.markdown("Structured engineering methodology applied throughout the project:")

    step_cols = st.columns(13)
    steps = [
        "1. Dataset",
        "→",
        "2. Data Prep",
        "→",
        "3. Train/Test Split",
        "→",
        "4. Model Training",
        "→",
        "5. Evaluation",
        "→",
        "6. Threshold Tuning",
        "→",
        "7. Deployment"
    ]
    
    for i, s in enumerate(steps):
        if s == "→":
            step_cols[i].markdown("<div class='workflow-arrow'>➔</div>", unsafe_allow_html=True)
        else:
            step_cols[i].markdown(f"<div class='workflow-step'>{s}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# 2. DATASET ANALYSIS
# =============================================================================
elif selected_page == "Dataset Analysis":
    st.title("📊 Dataset Architecture & Analysis")
    st.markdown("Examination of feature properties, confidentiality PCA transformations, and class imbalance structure.")

    # High-level Shape Metrics
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Records", "284,807 Rows")
    with col_s2:
        st.metric("Total Dimensions", "31 Columns")
    with col_s3:
        st.metric("Input Features", "30 Features")
    with col_s4:
        st.metric("Missing Values", "0 Nulls (100% Clean)")

    st.markdown("---")

    # Feature Groups & Explanations
    st.subheader("🔍 Feature Taxonomy & Groupings")
    st.write(
        "Due to confidentiality constraints, sensitive personal customer information (such as card numbers, names, "
        "locations, and merchant categories) was pre-transformed using Principal Component Analysis (PCA). "
        "The 31 dataset columns are structured into four primary feature groups:"
    )

    f_col1, f_col2 = st.columns([1, 1])

    with f_col1:
        st.markdown("""
        * **Time (`1 feature`):**
          Number of seconds elapsed between each transaction and the first recorded transaction in the dataset. Captures temporal purchasing rhythms over a 48-hour recording window.
        * **V1 – V28 (`28 features`):**
          Numerical principal components obtained via PCA transformation. To preserve privacy and prevent reverse-engineering of customer identities, their individual real-world domain meanings are not disclosed.
        * **Amount (`1 feature`):**
          The transaction amount in Euros (€). Useful for monetary risk assessments and cost-sensitive fraud thresholds.
        * **Class (`Target variable`):**
          Binary ground-truth indicator:
          * `0` = Legitimate transaction (Majority class)
          * `1` = Fraudulent transaction (Minority class)
        """)

    with f_col2:
        st.markdown("##### Feature Category Table")
        cat_df = pd.DataFrame([
            {"Category": "Transaction Time", "Features": "Time", "Type": "Numerical", "Role": "Input"},
            {"Category": "PCA Features", "Features": "V1 – V28", "Type": "Numerical", "Role": "Input"},
            {"Category": "Transaction Amount", "Features": "Amount", "Type": "Numerical", "Role": "Input"},
            {"Category": "Class", "Features": "Class", "Type": "Binary (0 or 1)", "Role": "Target"}
        ])
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Visualizing the Class Imbalance
    st.subheader("📈 Class Distribution Visual")
    
    col_d1, col_d2 = st.columns([2, 1])
    with col_d1:
        class_chart_df = pd.DataFrame({
            "Transaction Class": ["Legitimate (Class 0)", "Fraudulent (Class 1)"],
            "Transactions": [284315, 492]
        }).set_index("Transaction Class")
        st.bar_chart(class_chart_df, color="#38bdf8")

    with col_d2:
        st.markdown("##### Exact Imbalance Proportions")
        st.metric("Legitimate Count", "284,315", "99.8273%")
        st.metric("Fraudulent Count", "492", "0.1727%")
        st.caption("Ratio: 1 Fraud per 578 Legitimate Transactions")

    st.markdown("---")

    # Important Dataset Characteristic Section
    st.subheader("💡 Important Dataset Characteristics & Engineering Considerations")
    st.markdown("""
    1. **Extreme Class Imbalance:** Fraud represents only **0.1727%** of the entire dataset. In standard supervised learning, loss functions driven by overall accuracy easily get overwhelmed by legitimate transactions, resulting in unacceptable false negatives.
    2. **Stratified Sampling Necessity:** The project employed an 80/20 stratified split (`stratify=y`) to ensure training (394 frauds) and test sets (98 frauds) preserved identical 0.173% minority class frequencies.
    3. **Tree-Based Invariance:** While linear classifiers (like Logistic Regression) required feature scaling on `Time` and `Amount`, tree ensemble architectures (Random Forest and XGBoost) are invariant to monotonic scaling and were trained directly on raw features.
    4. **Memory-Safe Architecture:** *Note:* The dashboard relies on verified precomputed metadata and benchmarks. The full 144 MB raw CSV is intentionally not loaded into RAM during startup to guarantee instantaneous responsiveness.
    """)

    with st.expander("🔎 View Complete Feature Ordering (30 Production Inputs)"):
        if saved_features:
            f_cols = st.columns(5)
            for idx, feature_name in enumerate(saved_features):
                f_cols[idx % 5].markdown(f"`{idx+1:02d}.` **{feature_name}**")
        else:
            st.info("Feature list loaded from metadata.")


# =============================================================================
# 3. MODEL PERFORMANCE
# =============================================================================
elif selected_page == "Model Performance":
    st.title("🏆 Comparative Model Evaluation")
    st.markdown(
        "Empirical benchmarking of seven machine learning algorithms evaluated under identical stratified 80/20 train/test splits. "
        "All metric values are presented exactly as recorded during the experimental evaluation."
    )

    st.markdown("""
    <div class="info-banner">
        <strong>Production Model Notice:</strong> The active, authoritative live prediction model deployed in this system 
        is the saved Tuned XGBoost artifact in <code>models/tuned_xgboost.joblib</code>, paired with the live decision threshold 
        in <code>models/threshold.json</code> (0.75).
    </div>
    """, unsafe_allow_html=True)

    if not model_results_df.empty:
        st.subheader("📋 Complete Model Comparison Table")

        # Create percentage formatted view alongside raw view
        formatted_df = model_results_df.copy()
        
        pct_df = formatted_df.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]:
            if col in pct_df.columns:
                pct_df[col] = pct_df[col].apply(lambda x: f"{x * 100:.2f}%")

        st.dataframe(pct_df, use_container_width=True, hide_index=True)

        with st.expander("View Exact Raw Decimal Values (From CSV)", expanded=False):
            st.dataframe(model_results_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Interactive / Selectable Visualizations
        st.subheader("📊 Interactive Metric Comparisons")

        metric_view = st.selectbox(
            "Select Metric Comparison View:",
            [
                "A. F1 Score Comparison (Balanced Precision & Recall)",
                "B. Precision vs Recall Trade-off Comparison",
                "C. PR-AUC Comparison (Area Under Precision-Recall Curve)",
                "D. ROC-AUC Comparison (Receiver Operating Characteristic)",
                "E. Comprehensive Multi-Metric Comparison"
            ],
            index=0
        )

        chart_df = model_results_df.copy().set_index("Model")

        if "A. F1 Score" in metric_view:
            st.markdown("##### F1 Score Comparison across All 7 Models")
            f1_chart = (chart_df["F1 Score"] * 100).to_frame(name="F1 Score (%)")
            st.bar_chart(f1_chart, color="#38bdf8")
            st.caption("Tuned XGBoost (84.39%) and XGBoost Baseline (83.52%) demonstrate the highest balanced F1 scores.")

        elif "B. Precision vs Recall" in metric_view:
            st.markdown("##### Precision vs Recall Comparison")
            pr_rec_chart = (chart_df[["Precision", "Recall"]] * 100).rename(
                columns={"Precision": "Precision (%)", "Recall": "Recall (%)"}
            )
            st.bar_chart(pr_rec_chart)
            st.caption("Random Forest provides the highest precision (95.83%), but Tuned XGBoost strikes superior overall balance with 93.59% precision and 76.84% recall.")

        elif "C. PR-AUC" in metric_view:
            st.markdown("##### PR-AUC (Average Precision) Comparison")
            prauc_chart = (chart_df["PR-AUC"] * 100).to_frame(name="PR-AUC (%)")
            st.bar_chart(prauc_chart, color="#10b981")
            st.caption("PR-AUC is the gold standard for severe imbalance: Tuned XGBoost achieved the project's highest score (82.19%).")

        elif "D. ROC-AUC" in metric_view:
            st.markdown("##### ROC-AUC Comparison")
            roc_chart = (chart_df["ROC-AUC"] * 100).to_frame(name="ROC-AUC (%)")
            st.bar_chart(roc_chart, color="#6366f1")
            st.caption("All ensemble models exceed 94% ROC-AUC, with XGBoost models achieving peak discriminatory capability (~97.68%).")

        else:
            st.markdown("##### Multi-Metric Overview (F1, PR-AUC, ROC-AUC, Recall, Precision)")
            all_chart = (chart_df[["F1 Score", "PR-AUC", "Recall", "Precision"]] * 100)
            st.bar_chart(all_chart)

        st.markdown("---")

        # Metric Definitions & Rationale
        st.subheader("📚 Metric Definitions & Evaluation Rationale")
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(r"""
            * **Accuracy ($\frac{TP + TN}{Total}$):**
              Measures overall correct predictions. Deceptive in fraud detection because predicting 100% legitimate transactions still yields 99.83% accuracy.
            * **Precision ($\frac{TP}{TP + FP}$):**
              The fraction of flagged transactions that are truly fraudulent. High precision prevents unnecessary card blocks and angry cardholders.
            * **Recall / Sensitivity ($\frac{TP}{TP + FN}$):**
              The fraction of actual fraudulent transactions successfully captured. High recall directly prevents financial loss.
            """)
        with m_col2:
            st.markdown(r"""
            * **F1 Score ($2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$):**
              Harmonic mean of precision and recall. Punishes extreme imbalances between false alarms and missed frauds.
            * **ROC-AUC:**
              Area under the True Positive Rate vs False Positive Rate curve. Evaluates class separation across all thresholds.
            * **PR-AUC (Average Precision):**
              Area under the Precision-Recall curve. Focuses specifically on the minority positive class and is unaffected by high true negative counts.
            """)

        # Evaluated approaches breakdown
        st.markdown("---")
        st.subheader("🔬 Evaluated Machine Learning Approaches")
        st.markdown("""
        1. **Logistic Regression:** Linear probabilistic baseline fitted on standard-scaled features.
        2. **Decision Tree:** Non-linear rule-based tree model without ensemble bagging.
        3. **Random Forest:** Bagging ensemble of de-correlated decision trees with out-of-bag randomization.
        4. **SMOTE + Random Forest:** Synthetic Minority Over-sampling Technique combined with Random Forest.
        5. **Balanced Random Forest:** Random Forest ensemble where each bootstrap sample is down-sampled to balance classes.
        6. **XGBoost (Baseline):** Extreme Gradient Boosting with default parameters and minority class weight compensation (`scale_pos_weight`).
        7. **Tuned XGBoost (Selected Production Model):** Gradient boosted trees tuned with 3-fold cross-validated `RandomizedSearchCV` optimizing Average Precision (PR-AUC).
        """)

    else:
        st.error("Error: `model_comparison_results.csv` could not be loaded.")


# =============================================================================
# 4. THRESHOLD ANALYSIS
# =============================================================================
elif selected_page == "Threshold Analysis":
    st.title("🎯 Classification Threshold Optimization")
    st.markdown(
        "Standard machine learning classifiers apply a default decision boundary of `0.50`. "
        "In credit card fraud detection, tuning this threshold enables engineering teams to tailor the operational "
        "trade-off between fraud prevention (Recall) and customer friction (Precision)."
    )

    # Production Threshold Display (Dynamic from models/threshold.json)
    st.subheader("⚡ Authoritative Live Production Threshold")
    st.markdown(
        "The live threshold displayed below is loaded dynamically from `models/threshold.json` "
        "and controls the decision logic for the serialized tuned XGBoost model."
    )

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    with t_col1:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Decision Threshold</div>
            <div class="kpi-value">{saved_threshold:.2f}</div>
            <div class="kpi-caption">{threshold_info.get('selection_criterion', 'Maximum F1 Score')}</div>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Precision @ 0.75</div>
            <div class="kpi-value">{saved_precision * 100:.2f}%</div>
            <div class="kpi-caption">9 out of 10 flags are true fraud</div>
        </div>
        """, unsafe_allow_html=True)

    with t_col3:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">Recall @ 0.75</div>
            <div class="kpi-value">{saved_recall * 100:.2f}%</div>
            <div class="kpi-caption">Interception of 81.63% of frauds</div>
        </div>
        """, unsafe_allow_html=True)

    with t_col4:
        st.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-title">F1 Score @ 0.75</div>
            <div class="kpi-value">{saved_f1 * 100:.2f}%</div>
            <div class="kpi-caption">Peak harmonic performance</div>
        </div>
        """, unsafe_allow_html=True)

    # Operational Decision Logic
    st.markdown("##### ⚖️ Production Classification Logic")
    st.markdown(f"""
    The trained Tuned XGBoost model outputs a continuous fraud probability $P(\\text{{Fraud}} \\in [0, 1])$.

    * **If $P(\\text{{Fraud}}) \\ge {saved_threshold:.2f}$:** The transaction is classified as **FRAUD (Class 1)** $\\rightarrow$ *Flag for verification / decline transaction*.
    * **If $P(\\text{{Fraud}}) < {saved_threshold:.2f}$:** The transaction is classified as **LEGITIMATE (Class 0)** $\\rightarrow$ *Approve transaction seamlessly*.
    """)

    # Precision-Recall Trade-off Explanation
    st.markdown("""
    <div class="info-banner">
        <strong>The Precision-Recall Trade-Off:</strong><br>
        • <strong>Lower Threshold (&lt; 0.50):</strong> Flags more transactions as fraud. Increases <strong>Recall</strong> (fewer missed frauds) but results in more false positives (unnecessary customer declines).<br>
        • <strong>Higher Threshold (&gt; 0.50):</strong> Flags fewer transactions as fraud with higher certainty. Increases <strong>Precision</strong> (fewer false alarms) but risks decreasing Recall.<br>
        • <strong>Optimal Calibrated Threshold (0.75):</strong> Specifically maximizes the F1 Score (86.02%) for the tuned model on test data.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Historical Baseline Sweep
    st.subheader("📜 Historical Baseline XGBoost Threshold Analysis")
    st.markdown(
        "<span class='badge-ref'>Reference / Historical Analysis</span>",
        unsafe_allow_html=True
    )
    st.markdown(
        "*Notice:* The table and chart below display the historical threshold sweep performed on the **baseline XGBoost** "
        "model (recorded in `threshold_analysis.csv`). As established in project validation, this historical reference "
        "is preserved for comparative analysis, whereas the authoritative live threshold (`0.75`) is governed by `models/threshold.json`."
    )

    if not threshold_df.empty:
        t_chart_data = (threshold_df.set_index("Threshold")[["Precision", "Recall", "F1 Score"]] * 100)
        st.line_chart(t_chart_data)
        st.caption("Historical sensitivity curve across thresholds from 0.10 to 0.90 (in 0.05 step intervals).")

        with st.expander("View Historical Baseline Threshold CSV Data", expanded=False):
            display_t_df = threshold_df.copy()
            for col in ["Precision", "Recall", "F1 Score"]:
                display_t_df[col] = display_t_df[col].apply(lambda x: f"{x * 100:.2f}%")
            display_t_df["Threshold"] = display_t_df["Threshold"].apply(lambda x: f"{x:.2f}")
            st.dataframe(display_t_df, use_container_width=True, hide_index=True)
    else:
        st.warning("`threshold_analysis.csv` not found.")


# =============================================================================
# 5. FRAUD PREDICTION (STAGE 4: SINGLE TRANSACTION & STAGE 5: BATCH CSV INFERENCE)
# =============================================================================
elif selected_page == "Fraud Prediction":
    st.title("Fraud Prediction")
    st.markdown("#### Analyze credit-card transactions using the trained Tuned XGBoost model.")

    # 1. Load Live Model dynamically and validate
    model, model_error = load_tuned_xgboost_model()

    if model_error:
        st.error(f"❌ **Model Validation Error:** {model_error}")
        st.stop()

    # Validate feature count and ordering definition
    if len(saved_features) != 30:
        st.error(f"❌ **Feature Definition Error:** Expected exactly 30 features from models/feature_names.json, but loaded {len(saved_features)}.")
        st.stop()

    st.markdown("""
    <div class="info-banner">
        <strong>Inference Engine Active:</strong> Predictions use the trained <strong>Tuned XGBoost</strong> model 
        with the authoritative calibrated decision threshold of <strong>""" + f"{saved_threshold:.2f}" + """</strong>. 
        Raw feature values are evaluated directly without artificial scaling.
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION A: SINGLE TRANSACTION PREDICTION
    # =========================================================================
    st.subheader("⚡ Single Transaction Prediction")
    st.caption("Analyze an individual credit-card transaction using manual parameter entry.")

    # Sample Preset Dictionaries for Quick Testing & Viva Demonstrations
    LEGIT_PRESET = {
        "Time": 0.0, "Amount": 149.62,
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
        "V5": -0.3383, "V6": 0.4624, "V7": 0.2396, "V8": 0.0987,
        "V9": 0.3638, "V10": 0.0908, "V11": -0.5516, "V12": -0.6178,
        "V13": -0.9914, "V14": -0.3112, "V15": 1.4682, "V16": -0.4704,
        "V17": 0.2080, "V18": 0.0258, "V19": 0.4040, "V20": 0.2514,
        "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
        "V25": 0.1285, "V26": -0.1891, "V27": 0.1336, "V28": -0.0211
    }

    FRAUD_PRESET = {
        "Time": 406.0, "Amount": 0.0,
        "V1": -2.3122, "V2": 1.9520, "V3": -1.6099, "V4": 3.9979,
        "V5": -0.5222, "V6": -1.4265, "V7": -2.5374, "V8": 1.3917,
        "V9": -2.7701, "V10": -2.7723, "V11": 3.2020, "V12": -2.8999,
        "V13": -0.5952, "V14": -4.2893, "V15": 0.3897, "V16": -1.1407,
        "V17": -2.8301, "V18": -0.0168, "V19": 0.4170, "V20": 0.1269,
        "V21": 0.5172, "V22": -0.0350, "V23": -0.4652, "V24": 0.3202,
        "V25": 0.0445, "V26": 0.1778, "V27": 0.2611, "V28": -0.1433
    }

    # Ensure widget defaults exist in session state
    for feat in saved_features:
        if f"widget_{feat}" not in st.session_state:
            st.session_state[f"widget_{feat}"] = 0.0

    # Preset Selection Bar (Quick Loaders)
    st.markdown("##### 🚀 Quick Input Presets & Reset")
    btn_col1, btn_col2, btn_col3, _ = st.columns([1.2, 1.2, 1.2, 2.4])

    if btn_col1.button("📋 Load Legitimate Sample", use_container_width=True):
        for feat, val in LEGIT_PRESET.items():
            st.session_state[f"widget_{feat}"] = float(val)
        st.session_state["form_action"] = "Loaded legitimate sample preset."
        st.rerun()

    if btn_col2.button("🚨 Load Fraudulent Sample", use_container_width=True):
        for feat, val in FRAUD_PRESET.items():
            st.session_state[f"widget_{feat}"] = float(val)
        st.session_state["form_action"] = "Loaded fraudulent sample preset."
        st.rerun()

    if btn_col3.button("🔄 Reset to Defaults (Zero)", use_container_width=True):
        for feat in saved_features:
            st.session_state[f"widget_{feat}"] = 0.0
        st.session_state["form_action"] = "Form reset to default zero values."
        st.rerun()

    if "form_action" in st.session_state and st.session_state["form_action"]:
        st.caption(f"ℹ️ {st.session_state['form_action']}")

    # Form Definition for Single Transaction
    with st.form("single_transaction_prediction_form"):
        st.markdown("### 📝 Transaction Parameter Input")
        st.caption("Provide numerical transaction parameters. Features V1–V28 are PCA components; Time and Amount are in seconds and Euros.")

        # Top Section: Time and Amount
        c_time, c_amount = st.columns(2)
        with c_time:
            time_val = st.number_input(
                "Transaction Time (seconds)",
                step=1.0,
                format="%.2f",
                help="Seconds elapsed since the first transaction in the dataset. Must be non-negative.",
                key="widget_Time"
            )
        with c_amount:
            amount_val = st.number_input(
                "Transaction Amount (€)",
                step=1.0,
                format="%.2f",
                help="Transaction monetary value in Euros. Must be non-negative.",
                key="widget_Amount"
            )

        st.markdown("#### Anonymized PCA Features (V1 – V28)")
        st.caption("Numerical principal components preserving cardholder confidentiality. Enter real-valued decimal coordinates.")

        # Inputs for V1 through V28 in 4 columns of 7
        v_inputs = {}
        v_features = [f"V{i}" for i in range(1, 29)]
        grid_cols = st.columns(4)

        for idx, v_name in enumerate(v_features):
            col_target = grid_cols[idx % 4]
            with col_target:
                v_inputs[v_name] = st.number_input(
                    v_name,
                    step=0.01,
                    format="%.4f",
                    key=f"widget_{v_name}"
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_prediction = st.form_submit_button(
            "⚡ Predict Transaction",
            type="primary",
            use_container_width=True
        )

    # Prediction Execution Logic
    if submit_prediction:
        # 1. Validation Checks
        validation_errors = []

        if amount_val < 0.0:
            validation_errors.append(f"Transaction Amount cannot be negative (entered: {amount_val}).")

        if time_val < 0.0:
            validation_errors.append(f"Transaction Time cannot be negative (entered: {time_val}).")

        raw_input_dict = {"Time": time_val, "Amount": amount_val}
        raw_input_dict.update(v_inputs)

        for k, v in raw_input_dict.items():
            if not np.isfinite(v):
                validation_errors.append(f"Feature '{k}' contains non-finite value ({v}). All inputs must be finite numbers.")

        if validation_errors:
            for err in validation_errors:
                st.error(f"⚠️ Validation Error: {err}")
            st.stop()

        # 2. Strict Feature Ordering Guarantee
        try:
            input_dataframe = pd.DataFrame([raw_input_dict])[saved_features]
        except KeyError as ke:
            st.error(f"❌ Feature Construction Error: Missing expected feature key: {ke}")
            st.stop()

        if len(input_dataframe.columns) != 30:
            st.error(f"❌ Input validation failed: Expected exactly 30 features, but constructed {len(input_dataframe.columns)}.")
            st.stop()

        if input_dataframe.columns.tolist() != saved_features:
            st.error(f"❌ Feature order mismatch! Input columns do not match expected manifest.")
            st.stop()

        # 3. Model Inference via predict_proba()
        with st.spinner("Executing Tuned XGBoost inference..."):
            probabilities = model.predict_proba(input_dataframe)
            fraud_probability = float(probabilities[0][1])

        # 4. Decision Rule based on Authoritative Threshold
        is_fraudulent = fraud_probability >= saved_threshold
        pred_label = "FRAUDULENT TRANSACTION" if is_fraudulent else "LEGITIMATE TRANSACTION"

        # 5. Results UI
        st.markdown("---")
        st.subheader("Transaction Analysis")

        fraud_prob_pct = fraud_probability * 100.0
        threshold_pct = saved_threshold * 100.0

        if is_fraudulent:
            st.markdown(f"""
            <div class="result-card-fraud">
                <div class="result-tag">⚠️ High Risk Detection</div>
                <div class="result-headline">FRAUDULENT TRANSACTION</div>
                <div class="result-grid">
                    <div>
                        <div class="result-sub-label">Fraud Probability</div>
                        <div class="result-sub-val" style="color: #f87171;">{fraud_prob_pct:.2f}%</div>
                    </div>
                    <div>
                        <div class="result-sub-label">Decision Threshold</div>
                        <div class="result-sub-val">{threshold_pct:.2f}%</div>
                    </div>
                    <div>
                        <div class="result-sub-label">Model Used</div>
                        <div class="result-sub-val">Tuned XGBoost</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"🚨 **Final Decision: FRAUDULENT TRANSACTION** (Calculated Fraud Probability: {fraud_prob_pct:.4f}%)")
        else:
            st.markdown(f"""
            <div class="result-card-legit">
                <div class="result-tag">✅ Security Verification Passed</div>
                <div class="result-headline">LEGITIMATE TRANSACTION</div>
                <div class="result-grid">
                    <div>
                        <div class="result-sub-label">Fraud Probability</div>
                        <div class="result-sub-val" style="color: #34d399;">{fraud_prob_pct:.2f}%</div>
                    </div>
                    <div>
                        <div class="result-sub-label">Decision Threshold</div>
                        <div class="result-sub-val">{threshold_pct:.2f}%</div>
                    </div>
                    <div>
                        <div class="result-sub-label">Model Used</div>
                        <div class="result-sub-val">Tuned XGBoost</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"✅ **Final Decision: LEGITIMATE TRANSACTION** (Calculated Fraud Probability: {fraud_prob_pct:.4f}%)")

        # Visual confidence placement relative to threshold
        st.markdown("##### 📊 Decision Boundary Placement")
        st.progress(min(max(fraud_probability, 0.0), 1.0))
        st.caption(f"Estimated Probability: **{fraud_prob_pct:.2f}%** | Production Threshold Boundary: **{threshold_pct:.2f}%**")

        # Probability Interpretation
        st.markdown("""
        <div class="info-banner">
            <strong>Probability Interpretation:</strong><br>
            The model estimates the probability that this transaction belongs to the fraud class. 
            The production decision threshold is <strong>""" + f"{saved_threshold:.2f}" + """</strong>. 
            Probabilities at or above <strong>""" + f"{saved_threshold:.2f}" + """</strong> are classified as fraudulent.<br><br>
            <em>Note:</em> This value reflects statistical likelihood computed from learned patterns in training data 
            and serves as decision support for risk mitigation rather than absolute certainty.
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # SECTION B: BATCH CSV INFERENCE (STAGE 5)
    # =========================================================================
    st.markdown("---")
    st.header("📁 Batch CSV Inference")
    st.markdown("Upload a CSV containing multiple transactions and classify them using the same Tuned XGBoost model and production threshold.")

    # Expandable format specification and template download
    with st.expander("📄 Expected CSV Format & Downloadable Template", expanded=False):
        st.markdown("""
        The uploaded CSV file must contain the exact **30 model input features**:
        * `Time`
        * `V1` through `V28`
        * `Amount`
        
        *Note on the `Class` column:*
        * The CSV does **not** need to contain a `Class` column.
        * If `Class` is present, it is **never** passed as an input to the model.
        * If `Class` is present, it will be used exclusively for optional ground-truth evaluation (Accuracy, Precision, Recall, F1, and Confusion Matrix).
        """)
        st.download_button(
            label="📥 Download Sample Batch CSV Template",
            data=SAMPLE_CSV_TEMPLATE,
            file_name="sample_transactions_template.csv",
            mime="text/csv",
            use_container_width=True
        )

    # File Uploader
    uploaded_batch_file = st.file_uploader(
        "Upload CSV for Batch Inference",
        type=["csv"],
        key="batch_csv_uploader",
        help="Upload a CSV file containing transactions with the required 30 features."
    )

    if uploaded_batch_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_batch_file)
        except Exception as read_err:
            st.error(f"❌ Failed to parse uploaded CSV file: {read_err}")
            st.stop()

        if batch_df.empty:
            st.error("❌ Invalid CSV: The uploaded CSV file contains no data rows.")
            st.stop()

        # PART 3: File Validation
        missing_features = [col for col in saved_features if col not in batch_df.columns]
        extra_features = [col for col in batch_df.columns if col not in saved_features and col != "Class"]

        if missing_features:
            st.error(f"❌ Invalid CSV: Required feature columns are missing.\n\nMissing {len(missing_features)} column(s): **{', '.join(missing_features)}**.")
            st.caption("Please review the expected feature specification above or download the sample template.")
            st.stop()

        if extra_features:
            st.info(f"ℹ️ Note: Detected {len(extra_features)} extra column(s) not utilized by the model: `{', '.join(extra_features)}`. These will be safely ignored during model inference.")

        # Large file notice
        if len(batch_df) > 50000:
            st.warning(f"⚠️ Large batch detected ({len(batch_df):,} rows). Processing will execute vectorized inference.")

        # PART 4: Data Validation
        validation_failed = False
        batch_val_errors = []

        # Convert 30 feature columns to numeric, coercing invalid values to NaN
        coerced_features_df = batch_df[saved_features].apply(pd.to_numeric, errors="coerce")

        # Check for NaN / non-numeric / missing
        nan_mask = coerced_features_df.isna()
        if nan_mask.any().any():
            nan_rows = int(nan_mask.any(axis=1).sum())
            nan_cols = coerced_features_df.columns[nan_mask.any(axis=0)].tolist()
            batch_val_errors.append(f"Missing, NaN, or non-numeric values detected across **{nan_rows} row(s)** in column(s): `{', '.join(nan_cols)}`.")

        # Check for Infinite values
        inf_mask = np.isinf(coerced_features_df.values)
        if inf_mask.any():
            inf_rows = int(np.isinf(coerced_features_df).any(axis=1).sum())
            batch_val_errors.append(f"Infinite values (`inf` / `-inf`) detected across **{inf_rows} row(s)**. Inputs must be finite real numbers.")

        # Check for Negative Time
        if "Time" in coerced_features_df.columns:
            neg_time_count = int((coerced_features_df["Time"] < 0.0).sum())
            if neg_time_count > 0:
                batch_val_errors.append(f"Negative values detected in `Time` across **{neg_time_count} row(s)**. Transaction Time must be non-negative (>= 0).")

        # Check for Negative Amount
        if "Amount" in coerced_features_df.columns:
            neg_amount_count = int((coerced_features_df["Amount"] < 0.0).sum())
            if neg_amount_count > 0:
                batch_val_errors.append(f"Negative values detected in `Amount` across **{neg_amount_count} row(s)**. Transaction Amount must be non-negative (>= 0).")

        if batch_val_errors:
            st.error("❌ **Data Validation Summary (Inference Aborted):**")
            for b_err in batch_val_errors:
                st.error(f"⚠️ {b_err}")
            st.caption("Please correct the invalid entries in your dataset and re-upload the file.")
            st.stop()

        # PART 5: Batch Prediction
        X_batch = coerced_features_df[saved_features]
        if X_batch.columns.tolist() != saved_features or len(X_batch.columns) != 30:
            st.error("❌ Batch feature ordering validation failed.")
            st.stop()

        with st.spinner(f"Executing vectorized inference on {len(X_batch):,} transactions with Tuned XGBoost..."):
            batch_probabilities = model.predict_proba(X_batch)[:, 1]
            batch_predictions = np.where(batch_probabilities >= saved_threshold, "FRAUDULENT", "LEGITIMATE")
            is_fraud_bool = batch_probabilities >= saved_threshold

        total_tx = len(batch_df)
        pred_fraud_count = int(is_fraud_bool.sum())
        pred_legit_count = int((~is_fraud_bool).sum())
        pred_fraud_rate = (pred_fraud_count / total_tx) * 100.0

        # Check if ground-truth Class exists
        has_ground_truth = False
        valid_class_series = None
        if "Class" in batch_df.columns:
            class_num = pd.to_numeric(batch_df["Class"], errors="coerce")
            if not class_num.isna().any() and set(class_num.unique()).issubset({0, 1}):
                has_ground_truth = True
                valid_class_series = class_num.astype(int)

        # PART 7: Batch Summary KPI Cards
        st.markdown("### 📊 Batch Summary")
        kpi_b1, kpi_b2, kpi_b3, kpi_b4 = st.columns(4)

        with kpi_b1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Total Analyzed</div>
                <div class="kpi-value">{total_tx:,}</div>
                <div class="kpi-caption">Transactions in batch</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_b2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Predicted Fraudulent</div>
                <div class="kpi-value" style="color: #f87171;">{pred_fraud_count:,}</div>
                <div class="kpi-caption">Flagged (P ≥ {saved_threshold:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_b3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Predicted Legitimate</div>
                <div class="kpi-value" style="color: #34d399;">{pred_legit_count:,}</div>
                <div class="kpi-caption">Approved (P &lt; {saved_threshold:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        with kpi_b4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Predicted Fraud Rate</div>
                <div class="kpi-value">{pred_fraud_rate:.2f}%</div>
                <div class="kpi-caption">Of total batch records</div>
            </div>
            """, unsafe_allow_html=True)

        # Optional Ground Truth Summary
        if has_ground_truth and valid_class_series is not None:
            actual_fraud_count = int((valid_class_series == 1).sum())
            actual_legit_count = int((valid_class_series == 0).sum())
            actual_fraud_rate = (actual_fraud_count / total_tx) * 100.0

            st.markdown("##### 🎯 Ground-Truth Distribution (From Uploaded 'Class' Column)")
            gt_col1, gt_col2, gt_col3 = st.columns(3)
            gt_col1.metric("Actual Fraudulent", f"{actual_fraud_count:,}")
            gt_col2.metric("Actual Legitimate", f"{actual_legit_count:,}")
            gt_col3.metric("Actual Fraud Rate", f"{actual_fraud_rate:.2f}%")

            # PART 8: Optional Evaluation When Class Exists
            st.markdown("#### 🔬 Ground-Truth Evaluation Metrics & Confusion Matrix")
            y_pred_binary = (batch_probabilities >= saved_threshold).astype(int)

            batch_acc = accuracy_score(valid_class_series, y_pred_binary)
            batch_prec = precision_score(valid_class_series, y_pred_binary, zero_division=0)
            batch_rec = recall_score(valid_class_series, y_pred_binary, zero_division=0)
            batch_f1 = f1_score(valid_class_series, y_pred_binary, zero_division=0)

            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            met_col1.metric("Batch Accuracy", f"{batch_acc * 100:.2f}%")
            met_col2.metric("Batch Precision", f"{batch_prec * 100:.2f}%")
            met_col3.metric("Batch Recall", f"{batch_rec * 100:.2f}%")
            met_col4.metric("Batch F1 Score", f"{batch_f1 * 100:.2f}%")

            cm_array = confusion_matrix(valid_class_series, y_pred_binary, labels=[0, 1])
            cm_table = pd.DataFrame(
                cm_array,
                index=["Actual Legitimate (Class 0)", "Actual Fraudulent (Class 1)"],
                columns=["Predicted Legitimate", "Predicted Fraudulent"]
            )
            st.markdown("##### Confusion Matrix")
            st.dataframe(cm_table, use_container_width=True)
        else:
            st.info("ℹ️ No ground-truth `Class` column was provided (or non-binary), so model evaluation metrics cannot be calculated for this uploaded batch.")

        # PART 9: Visualizations
        st.markdown("---")
        st.subheader("📈 Batch Visualizations")
        chart_col1, chart_col2 = st.columns([1, 1])

        with chart_col1:
            st.markdown("##### Prediction Counts")
            pred_count_df = pd.DataFrame({
                "Classification": ["Legitimate", "Fraudulent"],
                "Transactions": [pred_legit_count, pred_fraud_count]
            }).set_index("Classification")
            st.bar_chart(pred_count_df, y="Transactions", color="#38bdf8")

        with chart_col2:
            st.markdown("##### Fraud Probability Distribution (Risk Tiers)")
            low_risk = int((batch_probabilities < 0.10).sum())
            mod_risk = int(((batch_probabilities >= 0.10) & (batch_probabilities < 0.50)).sum())
            elev_risk = int(((batch_probabilities >= 0.50) & (batch_probabilities < saved_threshold)).sum())
            high_risk = int((batch_probabilities >= saved_threshold).sum())

            tier_df = pd.DataFrame({
                "Risk Tier": ["Low (<10%)", "Moderate (10-50%)", "Elevated (50-75%)", f"Fraud (≥{saved_threshold*100:.0f}%)"],
                "Count": [low_risk, mod_risk, elev_risk, high_risk]
            }).set_index("Risk Tier")
            st.bar_chart(tier_df, y="Count", color="#f87171")

        # PART 6: Results Table
        st.markdown("---")
        st.subheader("📋 Transaction Classification Results")

        results_export_df = batch_df.copy()
        results_export_df.insert(0, "Row_Number", range(1, len(results_export_df) + 1))
        results_export_df["Fraud_Probability"] = batch_probabilities
        results_export_df["Predicted_Class"] = batch_predictions

        # If actual class present, rename or ensure clear label
        if "Class" in results_export_df.columns:
            results_export_df["Actual_Class"] = results_export_df["Class"].apply(
                lambda x: "FRAUDULENT" if x == 1 or x == "1" else "LEGITIMATE" if x == 0 or x == "0" else str(x)
            )

        # Formatted display table
        display_results_df = results_export_df.copy()
        display_results_df["Fraud_Probability"] = display_results_df["Fraud_Probability"].apply(lambda p: f"{p * 100:.2f}%")

        columns_priority = ["Row_Number", "Predicted_Class", "Fraud_Probability"]
        if "Actual_Class" in display_results_df.columns:
            columns_priority.append("Actual_Class")
        other_cols = [c for c in display_results_df.columns if c not in columns_priority]
        ordered_display_cols = columns_priority + other_cols

        st.dataframe(display_results_df[ordered_display_cols], use_container_width=True)

        # PART 10: Download Results
        st.markdown("---")
        st.subheader("💾 Export Batch Classification Results")

        csv_download_buffer = io.StringIO()
        results_export_df.to_csv(csv_download_buffer, index=False)
        csv_data_bytes = csv_download_buffer.getvalue().encode("utf-8")

        st.download_button(
            label="📥 Download Classified Results (CSV)",
            data=csv_data_bytes,
            file_name="fraud_detection_batch_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 9. Expandable Model Information
    st.markdown("---")
    with st.expander("ℹ️ Model Architecture & Threshold Specifications", expanded=False):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            * **Model:** `Tuned XGBoost`
            * **Model Artifact:** `models/tuned_xgboost.joblib`
            * **Number of Input Features:** `30` (Time, V1–V28, Amount)
            * **Feature Scaling:** *None* (Raw features expected directly)
            * **Inference Method:** Vectorized `model.predict_proba(df)[:, 1]`
            """)
        with col_m2:
            st.markdown(f"""
            * **Production Decision Threshold:** `{saved_threshold:.2f}` (Loaded from `models/threshold.json`)
            * **Threshold Selection Criterion:** `Maximum F1 Score`
            * **Saved Threshold Metrics (Tuned Model Test Evaluation):**
              * **Precision:** `{saved_precision * 100:.2f}%`
              * **Recall:** `{saved_recall * 100:.2f}%`
              * **F1 Score:** `{saved_f1 * 100:.2f}%`
            """)
        st.caption("Note: The threshold-selection metrics above were obtained during test-set calibration of the final tuned XGBoost model and are distinct from baseline comparisons recorded in earlier project stages.")


# =============================================================================
# 6. ABOUT PROJECT
# =============================================================================
elif selected_page == "About Project":
    st.title("📖 About This Project")
    st.markdown("### Final-Year Engineering Project in Machine Learning & Financial Cybersecurity")

    st.markdown("""
    #### Project Title
    **Effective Detection of Credit Card Fraud Using Logistic Regression, Decision Tree and Machine Learning Techniques**

    #### Problem Statement
    Credit card fraud detection is a **highly imbalanced binary classification problem** where fraudulent transactions 
    are rare compared with legitimate transactions (only 0.1727% fraud rate). Detecting such infrequent occurrences 
    requires models that can distinguish subtle patterns without generating an unmanageable volume of false alarms.

    #### Project Objective
    Develop, benchmark, and evaluate multiple machine learning approaches on real-world transaction data, selecting 
    an effective fraud-detection model using appropriate imbalance-aware evaluation metrics (Precision, Recall, F1 Score, and PR-AUC).

    #### Evaluated Approaches
    This study comprehensively implemented and compared seven machine learning configurations:
    1. **Logistic Regression** (Linear baseline with standardized features)
    2. **Decision Tree** (Non-linear rule partitioner)
    3. **Random Forest** (Bootstrap aggregation ensemble)
    4. **SMOTE + Random Forest** (Synthetic minority over-sampling combined with ensemble trees)
    5. **Balanced Random Forest** (Ensemble with balanced bootstrap sub-sampling)
    6. **XGBoost (Baseline)** (Gradient boosting with positive scale weighting)
    7. **Tuned XGBoost (Selected Final Model)** (Hyperparameter-optimized gradient boosting with PR-AUC target)

    #### Final Saved Prediction Model
    The final saved prediction model is **Tuned XGBoost**, persisted in `models/tuned_xgboost.joblib`. 
    Paired with an optimized decision threshold of **0.75** (from `models/threshold.json`), it delivers a balanced 
    **86.02% F1 Score**, **90.91% Precision**, and **81.63% Recall** on unseen test transactions.

    #### Technology Stack
    * **Core Programming Language:** Python
    * **Data Manipulation & Analysis:** Pandas, NumPy
    * **Machine Learning & Preprocessing:** Scikit-learn, Imbalanced-Learn
    * **Gradient Boosting Framework:** XGBoost
    * **Model Persistence & Serialization:** Joblib
    * **Frontend Dashboard & Visualization:** Streamlit
    """)

    st.markdown("---")
    st.caption("Credit Card Fraud Detection ML Project • Engineering Viva & Demonstration System")
