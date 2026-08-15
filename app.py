import json

import altair as alt
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Breast Cancer Risk Screening", layout="wide", page_icon="🩺")

st.markdown(
    """
    <style>
    header[data-testid="stHeader"] { display: none; }
    footer { visibility: hidden; }
    html, body { overflow-x: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 0 !important; }

    .navbar {
        position: relative;
        left: 50%;
        width: 100vw;
        margin-left: -50vw;
        margin-top: -1.5rem;
        margin-bottom: 2.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 3rem;
        background: #ffffff;
        border-bottom: 1px solid #e8e8e8;
    }
    .navbar-left { display: flex; align-items: center; gap: 0.7rem; }
    .navbar-logo {
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
        width: 2.3rem;
        height: 2.3rem;
        border-radius: 8px;
        background: #0E7C86;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .navbar-brand { font-size: 1.15rem; font-weight: 700; color: #262730; }
    .navbar-brand span { color: #0E7C86; }
    .navbar-badge {
        background: #E6F4F3;
        color: #0E7C86;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        white-space: nowrap;
    }

    /* ---------- Elegant global polish ---------- */
    .stApp { background: #E4F1F8; }
    h1, h2, h3 { letter-spacing: -0.01em; }

    [data-testid="stForm"],
    [data-testid="stExpander"],
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]:first-child span.card-mark) {
        border: 1px solid #E6EAEC !important;
        border-radius: 16px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 4px 14px rgba(16,24,40,0.045) !important;
        background: #ffffff !important;
    }
    [data-testid="stForm"],
    div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"]:first-child span.card-mark) {
        padding: 1.8rem 2rem !important;
    }
    [data-testid="stExpander"] summary { padding: 1rem 1.5rem !important; }
    [data-testid="stExpanderDetails"] { padding: 0.25rem 1.5rem 1.5rem 1.5rem !important; }

    [data-testid="stFormSubmitButton"] button {
        border-radius: 10px !important;
        border: none !important;
        background: #0E7C86 !important;
        transition: background 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease !important;
        padding: 0.65rem 1rem !important;
    }
    [data-testid="stFormSubmitButton"] button p { color: #ffffff !important; font-weight: 600 !important; }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #0A5F67 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(14,124,134,0.28) !important;
    }

    [data-testid="stNumberInputContainer"] {
        border-radius: 8px !important;
        border: 1px solid #E0E4E7 !important;
        background: #FAFBFC !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    [data-testid="stNumberInputContainer"]:focus-within {
        border-color: #0E7C86 !important;
        box-shadow: 0 0 0 3px rgba(14,124,134,0.12) !important;
    }

    .notice-card {
        background: #FFF9EF;
        border: 1px solid #F0DDB8;
        border-left: 4px solid #C9862A;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        color: #4a4438;
        line-height: 1.65;
    }
    .notice-title {
        font-weight: 700;
        color: #9A5B00;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.02rem;
    }
    .notice-card ul { margin: 0.6rem 0 0 0; padding-left: 1.2rem; }
    .notice-card li { margin-bottom: 0.4rem; }
    .notice-card strong { color: #7a4600; }

    .app-footer {
        position: relative;
        left: 50%;
        width: 100vw;
        margin-left: -50vw;
        margin-top: 3rem;
        padding: 1.6rem 3rem;
        border-top: 1px solid #E6EAEC;
        text-align: center;
        color: #8A9199;
        font-size: 0.85rem;
        background: #ffffff;
    }
    .app-footer strong { color: #3d3d3d; }
    </style>

    <div class="navbar">
        <div class="navbar-left">
            <div class="navbar-logo">BC</div>
            <div class="navbar-brand">BreastScan<span>AI</span></div>
        </div>
        <div class="navbar-badge">&#128994; Model Online &mdash; Stacking Ensemble</div>
    </div>
    """,
    unsafe_allow_html=True,
)


def size_to_category(radius, small_max, medium_max):
    if radius <= small_max:
        return "Small"
    if radius <= medium_max:
        return "Medium"
    return "Large"


@st.cache_resource
def load_artifacts():
    stacking_model = joblib.load("bc_model_stacking.pkl")
    xgb_pipe = joblib.load("bc_model_xgb_for_shap.pkl")
    with open("bc_feature_metadata.json") as f:
        metadata = json.load(f)
    return stacking_model, xgb_pipe, metadata


stacking_model, xgb_pipe, metadata = load_artifacts()
num = metadata["numeric_features"]
thresholds = metadata["thresholds"]
STACK_THRESH = thresholds["stacking"]

# size_category thresholds, computed from training data tertiles (Section 7 of the notebook)
SMALL_MAX, MEDIUM_MAX = 12.51, 15.09

FEATURE_LAYOUT = {
    "Mean values": [
        ("mean radius", "Mean Radius", 0.1),
        ("mean texture", "Mean Texture", 0.1),
        ("mean perimeter", "Mean Perimeter", 1.0),
        ("mean area", "Mean Area", 10.0),
        ("mean smoothness", "Mean Smoothness", 0.001),
        ("mean compactness", "Mean Compactness", 0.001),
        ("mean concavity", "Mean Concavity", 0.001),
        ("mean concave points", "Mean Concave Points", 0.001),
        ("mean symmetry", "Mean Symmetry", 0.001),
        ("mean fractal dimension", "Mean Fractal Dimension", 0.001),
    ],
    "Measurement error (standard error)": [
        ("radius error", "Radius Error", 0.01),
        ("texture error", "Texture Error", 0.01),
        ("perimeter error", "Perimeter Error", 0.1),
        ("area error", "Area Error", 1.0),
        ("smoothness error", "Smoothness Error", 0.0001),
        ("compactness error", "Compactness Error", 0.0001),
        ("concavity error", "Concavity Error", 0.0001),
        ("concave points error", "Concave Points Error", 0.0001),
        ("symmetry error", "Symmetry Error", 0.0001),
        ("fractal dimension error", "Fractal Dimension Error", 0.0001),
    ],
    "Worst (largest) values": [
        ("worst radius", "Worst Radius", 0.1),
        ("worst texture", "Worst Texture", 0.1),
        ("worst perimeter", "Worst Perimeter", 1.0),
        ("worst area", "Worst Area", 10.0),
        ("worst smoothness", "Worst Smoothness", 0.001),
        ("worst compactness", "Worst Compactness", 0.001),
        ("worst concavity", "Worst Concavity", 0.001),
        ("worst concave points", "Worst Concave Points", 0.001),
        ("worst symmetry", "Worst Symmetry", 0.001),
        ("worst fractal dimension", "Worst Fractal Dimension", 0.001),
    ],
}


def bounds_for(feature):
    """Derive slider/input bounds from the training data's own min/max, padded 15%,
    so this never goes stale if the underlying training data changes."""
    lo, hi, med = num[feature]["min"], num[feature]["max"], num[feature]["median"]
    pad = max((hi - lo) * 0.15, 1e-6)
    return max(0.0, lo - pad), hi + pad, med


st.title("Breast Cancer Risk Screening")
st.caption(
    "CIA-3 — ML for Social Good | Mission Health | "
    "Predicts the risk that a fine-needle aspirate (FNA) biopsy sample is malignant, to support faster, "
    "more consistent early screening. This is a screening aid, not a diagnosis — a pathologist must "
    "confirm any result. See the Ethics section below before acting on any output."
)

with st.expander("Model Performance", expanded=False):
    st.caption(
        "All four models were scored on the same held-out test set, with the same simulated measurement "
        "noise applied (notebook Section 10.5) — a fair, apples-to-apples comparison."
    )

    comparison_df = pd.DataFrame(
        {
            "ROC-AUC": [0.9880, 0.9929, 0.9896, 0.9917],
            "Accuracy": [0.9474, 0.9693, 0.9649, 0.9693],
            "F1": [0.9286, 0.9586, 0.9524, 0.9586],
            "Precision": [0.9398, 0.9643, 0.9639, 0.9643],
            "Recall": [0.9176, 0.9529, 0.9412, 0.9529],
        },
        index=["Baseline (LogReg)", "Bagging (Random Forest)", "Boosting (XGBoost)", "Stacking Ensemble"],
    )
    st.dataframe(comparison_df.style.format("{:.2%}"), use_container_width=True)

    chart_df = comparison_df.reset_index(names="Model").melt(id_vars="Model", var_name="Metric", value_name="Score")
    metric_order = ["Accuracy", "F1", "Precision", "ROC-AUC", "Recall"]
    base = alt.Chart(chart_df).encode(
        x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Metric:N", sort=metric_order),
        y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0.85, 1.0]), axis=alt.Axis(format="%")),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(domain=metric_order, range=["#0E7C86", "#4FA8B0", "#8FC3C8", "#C9E4E6", "#1C4E52"]),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".2%")],
    )
    # Dot plot, not bars: with a truncated (non-zero) axis — needed here since every score
    # sits in the 90s% — a bar's rectangle would draw from a baseline pushed off-canvas.
    chart = base.mark_circle(size=110).properties(width=600, height=320)
    st.altair_chart(chart, use_container_width=False)
    st.caption(
        "Random Forest and Stacking are essentially tied for best overall (96.9% accuracy each); "
        "all three ensembles beat the single-model LogReg baseline on every metric."
    )

    st.markdown("**Confusion matrices — same held-out test set**")
    conf_data = {
        "Baseline (LogReg)": (138, 5, 7, 78),
        "Bagging (RF)": (140, 3, 4, 81),
        "Boosting (XGB)": (140, 3, 5, 80),
        "Stacking": (140, 3, 4, 81),
    }
    cm_cols = st.columns(4)
    for col, (name, (tn, fp, fn, tp)) in zip(cm_cols, conf_data.items()):
        with col:
            cm = [[tn, fp], [fn, tp]]
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(cm, cmap="Blues", vmin=0, vmax=140)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center",
                            color="white" if cm[i][j] > 70 else "black", fontsize=13, fontweight="bold")
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
            ax.set_title(name, fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def render_field(feat, label, step, values):
    lo, hi, med = bounds_for(feat)
    decimals = max(0, -len(f"{step:g}".split(".")[-1])) if "." not in f"{step:g}" else len(f"{step:g}".split(".")[-1])
    fmt = f"%.{decimals}f" if decimals > 0 else "%.0f"
    values[feat] = st.number_input(label, float(lo), float(hi), float(round(med, decimals)), step=step, format=fmt)


with st.form("sample_form"):
    st.subheader("Cell-Nuclei Measurements (FNA biopsy)")
    st.caption(
        "Only the 10 mean values below are required. Error and worst-value measurements are optional — "
        "the model uses realistic median defaults for any left uninspected, so accuracy is unaffected. "
        "If your biopsy report includes them, fill them in for a more precise, patient-specific prediction."
    )
    values = {}

    st.markdown("**Mean values**")
    mean_cols = st.columns(2)
    for i, (feat, label, step) in enumerate(FEATURE_LAYOUT["Mean values"]):
        with mean_cols[i % 2]:
            render_field(feat, label, step, values)

    with st.expander("Optional: measurement error & worst values (improves precision if available)"):
        adv_cols = st.columns(2)
        for col, group_name in zip(adv_cols, ["Measurement error (standard error)", "Worst (largest) values"]):
            with col:
                st.markdown(f"**{group_name}**")
                for feat, label, step in FEATURE_LAYOUT[group_name]:
                    render_field(feat, label, step, values)

    submitted = st.form_submit_button("Predict Malignancy Risk", use_container_width=True)

if submitted:
    row = dict(values)
    row["radius_growth_ratio"] = row["worst radius"] / row["mean radius"] if row["mean radius"] else 0.0
    row["size_category"] = size_to_category(row["mean radius"], SMALL_MAX, MEDIUM_MAX)

    input_df = pd.DataFrame([row])

    proba = stacking_model.predict_proba(input_df)[0, 1]
    pred = int(proba >= STACK_THRESH)
    risk_pct = proba * 100
    # Confidence = how far the model's own probability sits from the 50/50 coin-flip point,
    # scaled to 0-100, for the class it actually predicted.
    confidence_pct = max(proba, 1 - proba) * 100

    st.divider()
    with st.container(border=True):
        st.markdown('<span class="card-mark"></span>', unsafe_allow_html=True)
        st.subheader("Prediction")
        verdict_col, prob_col, conf_col = st.columns([3, 2, 2])
        with verdict_col:
            if pred == 1:
                st.markdown("#### 🔴 HIGH RISK")
                st.markdown("likely **malignant**")
            else:
                st.markdown("#### 🟢 LOWER RISK")
                st.markdown("likely **benign**")
        with prob_col:
            st.metric("Malignancy Probability", f"{risk_pct:.1f}%")
        with conf_col:
            st.metric("Confidence", f"{confidence_pct:.0f} / 100")
        st.progress(int(round(confidence_pct)), text="Model confidence in this prediction")

    with st.container(border=True):
        st.markdown('<span class="card-mark"></span>', unsafe_allow_html=True)
        st.subheader("Why this prediction? (SHAP explanation)")
        st.caption(
            "Explained using the tuned XGBoost model (one of the three models combined inside the final "
            "Stacking prediction above) — TreeExplainer gives fast, exact SHAP values for tree-based models."
        )
        xgb_clf = xgb_pipe.named_steps["clf"]
        xgb_pre = xgb_pipe.named_steps["preprocessor"]
        input_transformed = xgb_pre.transform(input_df)
        feature_names = xgb_pre.get_feature_names_out()

        explainer = shap.TreeExplainer(xgb_clf)
        explanation = explainer(pd.DataFrame(input_transformed, columns=feature_names))

        fig, ax = plt.subplots(figsize=(9, 5))
        shap.plots.waterfall(explanation[0], max_display=10, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with st.container(border=True):
        st.markdown('<span class="card-mark"></span>', unsafe_allow_html=True)
        st.subheader("Ethics & Human Oversight")
        st.markdown(
            """
            <div class="notice-card">
                <div class="notice-title">Screening aid, not an automatic diagnosis</div>
                No patient should be diagnosed or treated based solely on this model's output.
                <ul>
                    <li><strong>False negatives are the costlier error</strong>: a missed malignant tumor
                        (scored low-risk) delays treatment; a false positive causes anxiety and an extra
                        biopsy, but not direct harm.</li>
                    <li><strong>No demographic fairness audit is possible</strong>: this dataset has no
                        age/ethnicity fields, so subgroup bias cannot be checked here — a real limitation,
                        not a strength.</li>
                    <li><strong>Small, single-institution, 1995 dataset</strong> (569 samples), with
                        simulated rather than genuinely collected measurement noise — must be re-validated
                        on real current imaging equipment and populations before any real clinical use.</li>
                    <li>A pathologist must confirm every result via standard biopsy review.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="app-footer">
        Built by <strong>Angel Blessy</strong> under the guidance of <strong>Dr. Helen K Joy</strong>
        &mdash; CHRIST (Deemed to be University), Department of Computer Science
    </div>
    """,
    unsafe_allow_html=True,
)
