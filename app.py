# ============================================================
# TELCO CUSTOMER CHURN ANALYTICS SYSTEM 
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. FILE SETTINGS
# ============================================================

DATA_FILE = "Telco_Customer-Churn.csv"
MODEL_FILE = "logistic_regression_model.pkl"

# ============================================================
# 3. SIDEBAR SETTINGS
# ============================================================

st.sidebar.markdown("## 📊 Telco Churn System")
st.sidebar.caption("Customer Analytics & Prediction Platform")
st.sidebar.divider()

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🔎 Pattern Mining",
        "🔮 Churn Prediction",
        "📈 Model Evaluation",
        "💡 Business Insights"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("# Semester XI")
st.sidebar.markdown("## DATA AND KNOWLEDGE MINING (IS-212) PROJECT")
st.sidebar.markdown("### By")
st.sidebar.caption("Bhone Min Khant")
st.sidebar.caption(" YKPT - 22522 (SE)")


# ============================================================
# 4. THEME
# ============================================================

if dark_mode:
    BG = "#0e1117"
    CARD = "#161b22"
    CARD_2 = "#1c2128"
    TEXT = "#f0f6fc"
    MUTED = "#9da7b3"
    BORDER = "#30363d"
    GRID = "#30363d"
    INPUT_BG = "#21262d"
else:
    BG = "#f5f7fb"
    CARD = "#ffffff"
    CARD_2 = "#f8fafc"
    TEXT = "#172033"
    MUTED = "#667085"
    BORDER = "#e5e7eb"
    GRID = "#d9dee7"
    INPUT_BG = "#ffffff"

# Presentation palette
BLUE = "#1976D2"
ORANGE = "#FF9800"
GREEN = "#2E9D62"
RED = "#E53935"
PURPLE = "#6A5ACD"
LIGHT_BLUE = "#DCEEFF"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    /* Main content spacing */
    .block-container {{
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1500px;
    }}

    /* Remove Streamlit's extra top whitespace on inner pages */
    [data-testid="stMainBlockContainer"] {{
        padding-top: 0.5rem !important;
    }}

    .main .block-container {{
        padding-top: 0.5rem !important;
    }}

    /* Keep the Streamlit header compact */
    [data-testid="stHeader"] {{
        height: 2.5rem !important;
        background: transparent !important;
    }}

    [data-testid="stSidebar"] {{
        background: {CARD};
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT} !important;
    }}

    /* Inner page wrapper */
    .page-content {{
        margin-top: 0 !important;
        padding-top: 0 !important;
    }}

    /* Main headings */
    .hero-title {{
        font-size: 44px;
        font-weight: 850;
        line-height: 1.05;
        color: {TEXT};
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        font-size: 17px;
        color: {MUTED};
        margin-bottom: 20px;
    }}

    .section-title {{
        font-size: 24px;
        font-weight: 750;
        color: {TEXT};
        margin-top: 12px;
        margin-bottom: 8px;
    }}

    /* Hero visual */
    .hero-card {{
        background: linear-gradient(135deg, #0D47A1 0%, #1976D2 55%, #42A5F5 100%);
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 22px;
        min-height: 190px;
        overflow: hidden;
        position: relative;
        box-shadow: 0 12px 30px rgba(0,0,0,.16);
    }}

    .hero-card h2 {{
        color: white;
        font-size: 29px;
        margin: 0 0 8px 0;
    }}

    .hero-card p {{
        color: #dce5ff;
        font-size: 16px;
        max-width: 720px;
        margin: 0;
    }}

    .hero-badge {{
        display: inline-block;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(255,255,255,.14);
        color: white;
        font-size: 13px;
        margin-bottom: 12px;
    }}

    .hero-orb {{
        position: absolute;
        width: 170px;
        height: 170px;
        border-radius: 50%;
        right: 50px;
        top: 12px;
        background: rgba(255,255,255,.10);
    }}

    .hero-orb2 {{
        position: absolute;
        width: 95px;
        height: 95px;
        border-radius: 50%;
        right: 145px;
        bottom: -28px;
        background: rgba(255,255,255,.09);
    }}

    /* KPI cards */
    .kpi-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 17px;
        padding: 20px 21px;
        min-height: 125px;
        box-shadow: 0 5px 18px rgba(0,0,0,.06);
    }}

    .kpi-label {{
        color: {MUTED};
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 7px;
    }}

    .kpi-value {{
        color: {TEXT};
        font-size: 29px;
        font-weight: 850;
    }}

    .kpi-note {{
        color: {MUTED};
        font-size: 12px;
        margin-top: 5px;
    }}

    /* Insight cards */
    .insight-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 17px;
        padding: 20px;
        min-height: 155px;
        margin-bottom: 12px;
    }}

    .insight-card h4 {{
        color: {TEXT};
        margin: 0 0 8px 0;
        font-size: 17px;
    }}

    .insight-card p {{
        color: {MUTED};
        margin: 0;
        line-height: 1.55;
        font-size: 14px;
    }}

    /* Prediction result */
    .prediction-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,.07);
    }}

    .risk-high {{
        color: #ff5c5c;
        font-size: 30px;
        font-weight: 850;
    }}

    .risk-medium {{
        color: #f59e0b;
        font-size: 30px;
        font-weight: 850;
    }}

    .risk-low {{
        color: #22c55e;
        font-size: 30px;
        font-weight: 850;
    }}

    .workflow {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 18px;
        padding: 22px;
        text-align: center;
    }}

    .workflow-step {{
        display: inline-block;
        background: {CARD_2};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 13px 15px;
        margin: 5px;
        color: {TEXT};
        font-weight: 700;
    }}

    /* Streamlit inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {{
        background: {INPUT_BG};
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        padding: 10px 18px;
    }}

    /* Hide excessive Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 5. LOAD DATA + MODEL
# ============================================================

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    ).fillna(0)
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


try:
    df = load_data()
    model = load_model()
except FileNotFoundError as e:
    st.error(
        "Required file not found. Please keep these files in the "
        "same folder as app.py:"
    )
    st.code(f"{DATA_FILE}\n{MODEL_FILE}")
    st.caption(str(e))
    st.stop()

# ============================================================
# 6. CLUSTERING MODEL FOR PREDICTION
# ============================================================
# The final churn model was trained with "Cluster" as an input feature.
# Therefore, every new customer must first be assigned to the same
# K-Means cluster before the churn model makes a prediction.

CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

cluster_scaler = StandardScaler()
cluster_scaled = cluster_scaler.fit_transform(df[CLUSTER_FEATURES])

# The notebook selected K=6 using the highest silhouette score.
final_kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)
df["Cluster"] = final_kmeans.fit_predict(cluster_scaled)

# Business-friendly names for the six K-Means customer segments.
# Numeric Cluster is kept for the ML model; these names are for display.
CLUSTER_NAMES = {
    0: "Low-Cost Loyal Customers",
    1: "High-Value Loyal Customers",
    2: "Senior High-Risk Customers",
    3: "New Low-Cost Customers",
    4: "Senior High-Value Customers",
    5: "New High-Cost Customers"
}

df["Cluster_Name"] = df["Cluster"].map(CLUSTER_NAMES)


def assign_customer_cluster(tenure, monthly_charges, total_charges, senior_citizen):
    """Assign one new customer to the same K-Means cluster used in training."""
    customer_cluster_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "SeniorCitizen": [senior_citizen]
    })

    customer_scaled = cluster_scaler.transform(customer_cluster_data)
    return int(final_kmeans.predict(customer_scaled)[0])


# ============================================================
# 7. HELPER FUNCTIONS
# ============================================================

def churn_rate_by(column):
    return (
        df.groupby(column)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )


def make_bar_chart(series, title, xlabel, ylabel=""):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.barh(
        series.index.astype(str),
        series.values,
        color=BLUE
    )
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25, color=GRID)
    ax.set_facecolor(CARD)
    fig.patch.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    for bar, value in zip(bars, series.values):
        ax.text(
            value + max(series.values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
            color=TEXT
        )
    plt.tight_layout()
    return fig


def make_tenure_chart():
    temp = df.copy()
    bins = [0, 6, 12, 24, 48, 72]
    labels = ["0–6", "7–12", "13–24", "25–48", "49–72"]
    temp["TenureGroup"] = pd.cut(
        temp["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )
    values = (
        temp.groupby("TenureGroup", observed=False)["Churn"]
        .apply(lambda x: (x == "Yes").mean() * 100)
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(values.index.astype(str), values.values, color=BLUE)
    ax.set_title(
        "Churn Rate Across Customer Tenure",
        fontsize=14,
        fontweight="bold",
        pad=12
    )
    ax.set_xlabel("Customer Tenure (months)")
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(axis="y", alpha=0.25, color=GRID)
    ax.set_facecolor(CARD)
    fig.patch.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    for x, value in enumerate(values.values):
        ax.text(x, value + 1, f"{value:.1f}%", ha="center", color=TEXT)
    plt.tight_layout()
    return fig


def kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 7. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    total_customers = len(df)
    churn_count = int((df["Churn"] == "Yes").sum())
    churn_rate = churn_count / total_customers * 100
    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-orb"></div>
            <div class="hero-orb2"></div>
            <div class="hero-badge"> Bhone Min Khant (YKPT - 22522) </div>
            <h2>Understand churn / Identify risk /  Take action </h2>
            <p>
                An interactive analytics system for exploring customer churn
                patterns and estimating individual churn risk using Logistic Regression
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-title">📊 Telco Customer Churn Analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Customer insights, churn patterns and business risk overview</div>',
        unsafe_allow_html=True
    )

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("👥 TOTAL CUSTOMERS", f"{total_customers:,}", "Customers in dataset")
    with c2:
        kpi("🔴 CHURN RATE", f"{churn_rate:.2f}%", f"{churn_count:,} churned customers")
    with c3:
        kpi("💰 AVG MONTHLY CHARGES", f"${avg_monthly:.2f}", "Average per customer")
    with c4:
        kpi("📅 AVG TENURE", f"{avg_tenure:.1f} months", "Average customer tenure")

    st.write("")

    tab1, tab2, tab3 = st.tabs(
        ["📌 Overview", "📊 Churn Analysis", "📋 Customer Data"]
    )

    with tab1:
        st.markdown('<div class="section-title">🎯 Overall Churn Overview</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            churn_counts = df["Churn"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.pie(
                churn_counts.values,
                labels=churn_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                explode=[0, 0.05],
                shadow=False,
                colors=[BLUE, ORANGE]
            )
            ax.set_title(
                "Customer Churn Distribution",
                fontsize=14,
                fontweight="bold",
                color=TEXT
            )
            fig.patch.set_facecolor(CARD)
            ax.set_facecolor(CARD)
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.markdown(
                """
                <div class="insight-card">
                    <h4>🔎 What does the dashboard show?</h4>
                    <p>
                    The dataset contains customers with different contract types,
                    tenure levels, service configurations and payment methods.
                    The analysis focuses on how these characteristics are
                    associated with churn.
                    </p>
                </div>
                <div class="insight-card">
                    <h4>📌 Analyst mindset</h4>
                    <p>
                    A higher churn rate for a customer group indicates an observed
                    association. It does not by itself prove that the characteristic
                    causes churn.
                    </p>
                </div>
                <div class="insight-card">
                    <h4>🤖 Predictive layer</h4>
                    <p>
                    The final baseline model is Logistic Regression, selected because
                    it achieved the strongest overall baseline performance among the
                    three tested classification models.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">🔄 Analytics Workflow</div>',
                    unsafe_allow_html=True)
        st.markdown(
            """
            <div class="workflow">
                <span class="workflow-step">📥 Customer Data</span>
                ➜
                <span class="workflow-step">🧹 Preprocessing</span>
                ➜
                <span class="workflow-step">📊 Descriptive Mining</span>
                ➜
                <span class="workflow-step">🔗 Association Mining</span>
                ➜
                <span class="workflow-step">🤖 Logistic Regression</span>
                ➜
                <span class="workflow-step">🎯 Churn Risk</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with tab2:
        st.markdown('<div class="section-title">📊 Churn Analysis</div>',
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            contract = churn_rate_by("Contract")
            fig = make_bar_chart(
                contract,
                "Churn Rate by Contract Type",
                "Churn Rate (%)",
                "Contract"
            )
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            fig = make_tenure_chart()
            st.pyplot(fig)
            plt.close(fig)

        c1, c2 = st.columns(2)

        with c1:
            internet = churn_rate_by("InternetService")
            fig = make_bar_chart(
                internet,
                "Churn Rate by Internet Service",
                "Churn Rate (%)",
                "Internet Service"
            )
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            support = churn_rate_by("TechSupport")
            fig = make_bar_chart(
                support,
                "Churn Rate by Tech Support",
                "Churn Rate (%)",
                "Tech Support"
            )
            st.pyplot(fig)
            plt.close(fig)

        c1, c2 = st.columns(2)

        with c1:
            payment = churn_rate_by("PaymentMethod")
            fig = make_bar_chart(
                payment,
                "Churn Rate by Payment Method",
                "Churn Rate (%)",
                "Payment Method"
            )
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            senior = churn_rate_by("SeniorCitizen")
            senior.index = ["Senior" if int(x) == 1 else "Non-Senior"
                            for x in senior.index]
            fig = make_bar_chart(
                senior,
                "Churn Rate by Senior Citizen Status",
                "Churn Rate (%)",
                "Customer Group"
            )
            st.pyplot(fig)
            plt.close(fig)

        c1, c2 = st.columns(2)

        with c1:
            partner = churn_rate_by("Partner")
            fig = make_bar_chart(
                partner,
                "Churn Rate by Partner Status",
                "Churn Rate (%)",
                "Partner"
            )
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            dependents = churn_rate_by("Dependents")
            fig = make_bar_chart(
                dependents,
                "Churn Rate by Dependents",
                "Churn Rate (%)",
                "Dependents"
            )
            st.pyplot(fig)
            plt.close(fig)

        st.markdown('<div class="section-title">💰 Monthly Charges</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(13, 4.8))
        for status in ["No", "Yes"]:
            values = df.loc[df["Churn"] == status, "MonthlyCharges"]
            hist_color = BLUE if status == "No" else ORANGE
            ax.hist(values, bins=30, alpha=0.55, label=f"Churn = {status}", color=hist_color)
        ax.set_title(
            "Monthly Charges Distribution by Churn Status",
            fontsize=14,
            fontweight="bold",
            color=TEXT
        )
        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Number of Customers")
        ax.legend()
        ax.grid(axis="y", alpha=0.2, color=GRID)
        ax.set_facecolor(CARD)
        fig.patch.set_facecolor(CARD)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.markdown('<div class="section-title">📋 Customer Data Preview</div>',
                    unsafe_allow_html=True)

        search = st.text_input(
            "🔎 Search customer ID",
            placeholder="Type a customer ID..."
        )

        display_df = df.copy()
        if search:
            display_df = display_df[
                display_df["customerID"].astype(str)
                .str.contains(search, case=False, na=False)
            ]

        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            height=500
        )

# ============================================================
# 8. PATTERN MINING
# ============================================================

elif page == "🔎 Pattern Mining":

    st.markdown(
        '<div class="hero-title">🔎 Pattern Mining</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Association rules and customer segmentation</div>',
        unsafe_allow_html=True
    )

    tab_assoc, tab_cluster = st.tabs(["🔗 Association Rules", "👥 Customer Clusters"])

    with tab_assoc:
        st.markdown('<div class="section-title">🔗 Churn-Associated Patterns</div>', unsafe_allow_html=True)
        st.info(
            "Association mining identifies characteristics that frequently occur together with Churn = Yes. "
            "These are associations, not proof of causation and not future churn probabilities."
        )

        association_data = pd.DataFrame({
            "Pattern": [
                "Fiber optic + Month-to-month + TechSupport = No",
                "Fiber optic + Month-to-month + Payment = Electronic check",
                "Fiber optic + Month-to-month",
                "Month-to-month + TechSupport = No + Payment = Electronic check",
                "Fiber optic + TechSupport = No + SeniorCitizen = Yes"
            ],
            "Support": [0.1017, 0.1062, 0.1650, 0.1247, 0.0663],
            "Confidence": [0.6292, 0.5807, 0.5461, 0.5842, 0.6243],
            "Lift": [2.3709, 2.1884, 2.0598, 2.2013, 2.3527]
        })

        display_rules = association_data.copy()
        display_rules["Support"] = display_rules["Support"].map(lambda x: f"{x:.2%}")
        display_rules["Confidence"] = display_rules["Confidence"].map(lambda x: f"{x:.2%}")
        display_rules["Lift"] = display_rules["Lift"].map(lambda x: f"{x:.2f}")
        st.dataframe(display_rules, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">📊 Lift Visualization</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(11, 4.8))
        short_labels = ["Fiber + M2M + No Tech", "Fiber + M2M + E-check", "Fiber + M2M", "M2M + No Tech + E-check", "Fiber + No Tech + Senior"]
        ax.barh(short_labels[::-1], association_data["Lift"].values[::-1], color=GREEN)
        ax.axvline(1, linestyle="--", linewidth=1.2)
        ax.set_xlabel("Lift")
        ax.set_title("Top Churn-Associated Association Rules", fontsize=14, fontweight="bold", color=TEXT)
        ax.grid(axis="x", alpha=0.2, color=GRID)
        ax.set_facecolor(CARD)
        fig.patch.set_facecolor(CARD)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(
            '<div class="insight-card"><h4>🧠 Analyst interpretation</h4>'
            '<p>Month-to-month contracts, fiber optic service and no Tech Support repeatedly appear in churn-associated combinations. The rules describe observed co-occurrence in the dataset and should not be interpreted as causal relationships.</p></div>',
            unsafe_allow_html=True
        )

    with tab_cluster:
        st.markdown('<div class="section-title">👥 Customer Segmentation with K-Means</div>', unsafe_allow_html=True)
        st.info(
            "K-Means groups customers using tenure, MonthlyCharges, TotalCharges and SeniorCitizen. "
            "The selected K is based on the highest Silhouette Score among K = 2 to 6."
        )

        cluster_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
        cluster_data = df[cluster_features].copy()
        cluster_scaled = StandardScaler().fit_transform(cluster_data)
        k_values = list(range(2, 7))
        sil_scores = []
        for k in k_values:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(cluster_scaled)
            sil_scores.append(silhouette_score(cluster_scaled, labels))

        best_k = k_values[int(np.argmax(sil_scores))]
        best_silhouette = max(sil_scores)

        c1, c2 = st.columns(2)
        with c1:
            kpi("🎯 SELECTED K", str(best_k), "Highest silhouette score")
        with c2:
            kpi("📐 SILHOUETTE SCORE", f"{best_silhouette:.4f}", "Cluster separation / cohesion")

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(k_values, sil_scores, marker="o", linewidth=2, color=BLUE)
            ax.axvline(best_k, linestyle="--", color=ORANGE, linewidth=1.5)
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score by K", fontsize=14, fontweight="bold", color=TEXT)
            ax.grid(alpha=0.2, color=GRID)
            ax.set_facecolor(CARD)
            fig.patch.set_facecolor(CARD)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
            for spine in ax.spines.values(): spine.set_color(BORDER)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with c2:
            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = final_kmeans.fit_predict(cluster_scaled)
            temp = df.copy()
            temp["Cluster_UI"] = cluster_labels
            temp["ChurnFlag_UI"] = (temp["Churn"] == "Yes").astype(int)
            summary = temp.groupby("Cluster_UI").agg(
                Customers=("customerID", "count"),
                Avg_Tenure=("tenure", "mean"),
                Avg_MonthlyCharges=("MonthlyCharges", "mean"),
                Churn_Rate=("ChurnFlag_UI", "mean")
            ).reset_index()
            summary["Churn_Rate"] *= 100
            summary["Cluster_Name"] = summary["Cluster_UI"].map(CLUSTER_NAMES)
            ax = None
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(summary["Cluster_Name"], summary["Churn_Rate"], color=ORANGE)
            ax.set_xlabel("Customer Segment")
            ax.tick_params(axis="x", labelrotation=35)
            ax.set_ylabel("Observed Churn Rate (%)")
            ax.set_title("Observed Churn Rate by Cluster", fontsize=14, fontweight="bold", color=TEXT)
            ax.grid(axis="y", alpha=0.2, color=GRID)
            ax.set_facecolor(CARD); fig.patch.set_facecolor(CARD); ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
            for spine in ax.spines.values(): spine.set_color(BORDER)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        summary_display = summary.rename(columns={
            "Cluster_UI": "Cluster ID",
            "Cluster_Name": "Customer Segment",
            "Avg_Tenure": "Avg Tenure",
            "Avg_MonthlyCharges": "Avg Monthly Charges",
            "Churn_Rate": "Churn Rate"
        }).round(2)
        summary_display["Churn Rate"] = summary_display["Churn Rate"].map(lambda x: f"{x:.2f}%")
        st.dataframe(summary_display, use_container_width=True, hide_index=True)

        st.caption(
            "The clustering result is descriptive segmentation. Differences in observed churn rates across clusters do not establish that cluster characteristics cause churn."
        )

# ============================================================
# 8. CHURN PREDICTION
# ============================================================

elif page == "🔮 Churn Prediction":

    st.markdown(
        '<div class="hero-title">🔮 Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Enter customer information to estimate churn risk</div>',
        unsafe_allow_html=True
    )

    st.info(
        "The model predicts churn using the same feature structure used during training."
    )

    st.markdown('<div class="section-title">👤 Customer Information</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with c2:
        tenure = st.number_input(
            "Tenure (months)", min_value=0, max_value=72, value=12
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with c3:
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    st.markdown('<div class="section-title">📡 Service & Billing Information</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with c2:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )

    with c3:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    c1, c2 = st.columns(2)

    with c1:
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.01
        )

    with c2:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(monthly_charges * max(tenure, 1)),
            step=0.01
        )

    if tenure <= 6:
        tenure_group = "0-6"
    elif tenure <= 12:
        tenure_group = "7-12"
    elif tenure <= 24:
        tenure_group = "13-24"
    elif tenure <= 48:
        tenure_group = "25-48"
    else:
        tenure_group = "49-72"

    st.caption(f"Automatically derived tenure group: **{tenure_group} months**")

    st.divider()

    predict_button = st.button(
        "🔮 Predict Churn Risk",
        use_container_width=True,
        type="primary"
    )

    if predict_button:

        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "tenure_group": [tenure_group],
            "Cluster": [assign_customer_cluster(
                tenure,
                monthly_charges,
                total_charges,
                senior_citizen
            )]
        })

        customer_cluster = int(input_data["Cluster"].iloc[0])
        customer_cluster_name = CLUSTER_NAMES.get(
            customer_cluster, f"Cluster {customer_cluster}"
        )

        st.info(
            f"Customer segmentation: **{customer_cluster_name}** "
            f"(derived automatically using K-Means)"
        )

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        probability_percent = probability * 100

        if probability >= 0.70:
            risk_level = "HIGH RISK"
            risk_class = "risk-high"
            icon = "🔴"
        elif probability >= 0.40:
            risk_level = "MEDIUM RISK"
            risk_class = "risk-medium"
            icon = "🟠"
        else:
            risk_level = "LOW RISK"
            risk_class = "risk-low"
            icon = "🟢"

        st.markdown('<div class="section-title">📊 Prediction Result</div>',
                    unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            result = "YES" if prediction == 1 else "NO"
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="color:{MUTED};font-size:13px;font-weight:700;">
                        CHURN PREDICTION
                    </div>
                    <div style="font-size:31px;font-weight:850;color:{TEXT};">
                        {"🔴" if prediction == 1 else "🟢"} {result}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c2:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="color:{MUTED};font-size:13px;font-weight:700;">
                        CHURN PROBABILITY
                    </div>
                    <div style="font-size:31px;font-weight:850;color:{TEXT};">
                        {probability_percent:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with c3:
            st.markdown(
                f"""
                <div class="prediction-card">
                    <div style="color:{MUTED};font-size:13px;font-weight:700;">
                        RISK LEVEL
                    </div>
                    <div class="{risk_class}">
                        {icon} {risk_level}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(float(probability))

        st.markdown('<div class="section-title">💡 Business Interpretation</div>',
                    unsafe_allow_html=True)

        if prediction == 1:
            st.warning(
                "This customer is predicted to have a higher likelihood of churn. "
                "The company may consider targeted retention actions."
            )
        else:
            st.success(
                "This customer is predicted to be more likely to remain with the company. "
                "However, monitoring is still recommended."
            )

        st.caption(
            "Important: model-estimated risk is not a causal explanation. "
            "Observed associations do not establish that a feature causes churn."
        )


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

elif page == "📈 Model Evaluation":

    st.markdown(
        '<div class="hero-title">📈 Model Evaluation</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Baseline classification performance comparison</div>',
        unsafe_allow_html=True
    )

    # Actual baseline results from the project experiments
    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],
        "Accuracy": [0.8013, 0.7303, 0.7779],
        "Precision": [0.6610, 0.4924, 0.6027],
        "Recall": [0.5160, 0.5187, 0.4786],
        "F1 Score": [0.5796, 0.5052, 0.5335],
        "ROC-AUC": [0.8449, 0.6625, 0.8184]
    })

    best = results.iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("🏆 SELECTED MODEL", "Logistic Regression",
            "Strongest baseline overall")
    with c2:
        kpi("🎯 ROC-AUC", "84.49%",
            "Best among tested baselines")
    with c3:
        kpi("📊 F1 SCORE", "57.96%",
            "Best among tested baselines")

    st.markdown('<div class="section-title">📊 Performance Comparison</div>',
                unsafe_allow_html=True)

    styled = results.copy()
    for col in styled.columns[1:]:
        styled[col] = styled[col].map(lambda x: f"{x:.2%}")

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">📉 Metric Visualization</div>',
                unsafe_allow_html=True)

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    x = np.arange(len(results))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5.5))

    offsets = [-2, -1, 0, 1, 2]
    metric_colors = [BLUE, ORANGE, GREEN, RED, PURPLE]
    for metric, offset, bar_color in zip(metric_names, offsets, metric_colors):
        ax.bar(
            x + offset * width,
            results[metric],
            width,
            label=metric,
            color=bar_color
        )

    ax.set_xticks(x)
    ax.set_xticklabels(results["Model"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title(
        "Classification Model Performance Comparison",
        fontsize=14,
        fontweight="bold",
        color=TEXT
    )
    ax.grid(axis="y", alpha=0.2, color=GRID)
    ax.legend(ncol=5)
    ax.set_facecolor(CARD)
    fig.patch.set_facecolor(CARD)
    ax.tick_params(colors=TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(BORDER)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown('<div class="section-title">🎯 Confusion Matrix Summary</div>',
                unsafe_allow_html=True)

    cm = {
        "Logistic Regression": [[936, 99], [181, 193]],
        "Decision Tree": [[835, 200], [180, 194]],
        "Random Forest": [[917, 118], [195, 179]]
    }

    c1, c2, c3 = st.columns(3)
    for container, (name, matrix) in zip([c1, c2, c3], cm.items()):
        with container:
            st.markdown(f"**{name}**")
            st.dataframe(
                pd.DataFrame(
                    matrix,
                    index=["Actual No", "Actual Yes"],
                    columns=["Predicted No", "Predicted Yes"]
                ),
                use_container_width=True
            )

    st.info(
        "Logistic Regression achieved the strongest baseline performance across "
        "Accuracy, Precision, F1-score and ROC-AUC. Recall was very similar to "
        "Decision Tree, so Logistic Regression was selected as the current final model."
    )

    st.markdown('<div class="section-title">⚖️ Class Imbalance Experiment</div>', unsafe_allow_html=True)
    st.caption("The dataset contains more No-Churn customers than Churn customers. The experiments compare the baseline with class weighting and SMOTE using the same test set.")

    imbalance = pd.DataFrame({
        "Method": ["Baseline", "Class Weight = Balanced", "SMOTE"],
        "Accuracy": [0.8041, 0.7388, 0.7410],
        "Precision": [0.6531, 0.5051, 0.5077],
        "Recall": [0.5588, 0.8021, 0.7941],
        "F1 Score": [0.6023, 0.6198, 0.6194],
        "ROC-AUC": [0.8426, 0.8449, 0.8428]
    })

    shown = imbalance.copy()
    for col in shown.columns[1:]:
        shown[col] = shown[col].map(lambda x: f"{x:.2%}")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    x2 = np.arange(len(imbalance))
    width2 = 0.16
    metrics2 = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    metric_colors2 = [BLUE, ORANGE, GREEN, RED, PURPLE]
    for j, (metric, col) in enumerate(zip(metrics2, metric_colors2)):
        ax.bar(x2 + (j-2)*width2, imbalance[metric], width2, label=metric, color=col)
    ax.set_xticks(x2); ax.set_xticklabels(imbalance["Method"])
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Baseline vs Class Weight vs SMOTE", fontsize=14, fontweight="bold", color=TEXT)
    ax.legend(ncol=5); ax.grid(axis="y", alpha=0.2, color=GRID)
    ax.set_facecolor(CARD); fig.patch.set_facecolor(CARD); ax.tick_params(colors=TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values(): spine.set_color(BORDER)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="insight-card"><h4>⚖️ Class Weight</h4><p>Giving more weight to the minority churn class increased Recall to 80.21%, but Accuracy and Precision decreased compared with the baseline.</p></div>',
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            '<div class="insight-card"><h4>🧪 SMOTE</h4><p>SMOTE generated synthetic minority-class samples only in the training data. Recall reached 79.41% and F1-score 61.94%, with lower Accuracy than the baseline.</p></div>',
            unsafe_allow_html=True
        )

# ============================================================
# 10. BUSINESS INSIGHTS
# ============================================================

elif page == "💡 Business Insights":

    st.markdown(
        '<div class="hero-title">💡 Business Insights & Recommendations</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="hero-subtitle">Turning observed patterns into practical retention actions</div>',
        unsafe_allow_html=True
    )

    insights = [
        (
            "📄 Month-to-Month Contracts",
            "Month-to-month customers show a substantially higher observed churn rate.",
            "Consider targeted incentives or suitable contract options for customers who may benefit from longer-term plans."
        ),
        (
            "📅 New Customers",
            "Customers with 0–6 months of tenure show the highest observed churn rate.",
            "Introduce early onboarding, engagement and retention activities during the first months."
        ),
        (
            "🌐 Fiber Optic Customers",
            "Fiber optic customers show a higher observed churn rate than DSL and no-internet groups.",
            "Investigate service experience, pricing and support factors before deciding on a retention strategy."
        ),
        (
            "🛠️ Tech Support = No",
            "Customers without Tech Support show a higher observed churn rate.",
            "Test targeted support offers or customer education, while recognizing this is an association rather than proof of causation."
        ),
        (
            "💳 Electronic Check",
            "Electronic-check customers show a higher observed churn rate among payment methods.",
            "Investigate payment experience and consider offering convenient alternative payment options."
        ),
        (
            "👴 Senior Citizens",
            "Senior customers show a higher observed churn rate than non-senior customers.",
            "Consider tailored communication, support and service guidance."
        )
    ]

    for i in range(0, len(insights), 2):
        c1, c2 = st.columns(2)
        for container, item in zip([c1, c2], insights[i:i+2]):
            title, finding, action = item
            with container:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <h4>{title}</h4>
                        <p><b>Finding:</b> {finding}</p>
                        <p style="margin-top:8px;"><b>Action:</b> {action}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown('<div class="section-title">🧠 Analyst Note</div>',
                unsafe_allow_html=True)

    st.warning(
        "These findings describe observed associations in the dataset. "
        "They should not be interpreted as causal relationships without further "
        "analysis or controlled evidence."
    )

    st.markdown('<div class="section-title">🚀 Recommended Business Workflow</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="workflow">
            <span class="workflow-step">1️⃣ Identify Risk</span>
            ➜
            <span class="workflow-step">2️⃣ Understand Customer</span>
            ➜
            <span class="workflow-step">3️⃣ Target Retention</span>
            ➜
            <span class="workflow-step">4️⃣ Monitor Outcome</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("")
    st.caption(
        "Project scope: descriptive mining + association mining + predictive mining + system implementation."
    )

