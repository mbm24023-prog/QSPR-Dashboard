from rdkit import Chem
from rdkit.Chem import Draw
import math
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

plt.rcParams.update({
    "axes.edgecolor": "#9ca3af",
    "axes.labelcolor": "#111827",
    "text.color": "#111827",
    "xtick.color": "#6b7280",
    "ytick.color": "#6b7280",
})

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# --------- polynomial helpers ---------
def fit_poly(x, y, degree):
    X = np.vstack([x ** d for d in range(1, degree + 1)]).T
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model, r2


def predict_poly(model, x, degree):
    X = np.vstack([x ** d for d in range(1, degree + 1)]).T
    return model.predict(X)


def equation_str(model, degree, index_name, prop_name):
    coefs = model.coef_
    intercept = model.intercept_
    terms = [f"{intercept:.3f}"]
    for d, c in enumerate(coefs, start=1):
        terms.append(f"{c:+.3f}·{index_name}^{d}")
    return f"{prop_name} = " + " ".join(terms)


# --------- topological indices and entropy ---------
INDEX_INFO = {
    "M1": ("First Zagreb Index", "Sum of squared vertex degrees; measures overall connectivity."),
    "M2": ("Second Zagreb Index", "Sum of products of degrees of adjacent vertices."),
    "Randic": ("Randić Index", "Branching index based on inverse square root of degree products."),
    "ABC": ("Atom–Bond Connectivity Index", "Captures complexity with emphasis on highly connected atoms."),
    "AZI": ("Augmented Zagreb Index", "Refined Zagreb-type index for connectivity effects."),
    "Harmonic": ("Harmonic Index (edge-based)", "Uses harmonic mean of degrees over edges."),
    "GA": ("Geometric–Arithmetic Index", "Compares geometric vs arithmetic mean of neighbor degrees."),
    "SumConn": ("Sum-Connectivity Index", "Uses inverse of degree sums over edges."),
    "Forgotten": ("F-index (Forgotten Index)", "Sum of cubes of degrees; emphasizes high-degree atoms."),
    "SSD": ("Symmetric Division Degree Index", "Measures similarity of degrees on each bond."),
    "H_deg": ("Degree-based Entropy (vertex)", "Shannon entropy of vertex degree distribution."),
    "H_edge": ("Edge-based Entropy", "Shannon entropy of edge degree-product distribution."),
}


def get_degrees(mol):
    return [atom.GetDegree() for atom in mol.GetAtoms()]


def edges_with_degrees(mol):
    pairs = []
    for bond in mol.GetBonds():
        du = bond.GetBeginAtom().GetDegree()
        dv = bond.GetEndAtom().GetDegree()
        pairs.append((du, dv))
    return pairs


def M1_index(mol):
    return sum(d ** 2 for d in get_degrees(mol))


def M2_index(mol):
    return sum(du * dv for du, dv in edges_with_degrees(mol))


def Randic_index(mol):
    return sum(1.0 / math.sqrt(du * dv) for du, dv in edges_with_degrees(mol))


def ABC_index(mol):
    total = 0.0
    for du, dv in edges_with_degrees(mol):
        num = du + dv - 2
        den = du * dv
        total += math.sqrt(num / den)
    return total


def AZI_index(mol):
    total = 0.0
    for du, dv in edges_with_degrees(mol):
        num = du * dv
        den = du + dv - 2
        total += (num / den) ** 3
    return total


def harmonic_index(mol):
    return sum(2.0 / (du + dv) for du, dv in edges_with_degrees(mol))


def GA_index(mol):
    total = 0.0
    for du, dv in edges_with_degrees(mol):
        num = 2.0 * math.sqrt(du * dv)
        den = du + dv
        total += num / den
    return total


def sum_connectivity_index(mol):
    return sum(1.0 / math.sqrt(du + dv) for du, dv in edges_with_degrees(mol))


def forgotten_index(mol):
    return sum(d ** 3 for d in get_degrees(mol))


def SSD_index(mol):
    return sum((du - dv) ** 2 for du, dv in edges_with_degrees(mol))


def degree_entropy(mol, base=2):
    degs = get_degrees(mol)
    n = len(degs)
    if n == 0:
        return 0.0
    counts = Counter(degs)
    H = 0.0
    log_fn = math.log2 if base == 2 else math.log
    for cnt in counts.values():
        p = cnt / n
        H -= p * log_fn(p)
    return H


def edge_degree_entropy(mol, base=2):
    pairs = edges_with_degrees(mol)
    if not pairs:
        return 0.0
    s_vals = [du + dv for du, dv in pairs]
    m = len(s_vals)
    counts = Counter(s_vals)
    H = 0.0
    log_fn = math.log2 if base == 2 else math.log
    for cnt in counts.values():
        p = cnt / m
        H -= p * log_fn(p)
    return H

def compute_all_indices_from_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    img = Draw.MolToImage(mol, size=(200, 200))

    return {
        "Molecule": img,   # <-- ADD THIS
        "SMILES": smi,
        "M1": M1_index(mol),
        "M2": M2_index(mol),
        "Randic": Randic_index(mol),
        "ABC": ABC_index(mol),
        "AZI": AZI_index(mol),
        "Harmonic": harmonic_index(mol),
        "GA": GA_index(mol),
        "SumConn": sum_connectivity_index(mol),
        "Forgotten": forgotten_index(mol),
        "SSD": SSD_index(mol),
        "H_deg": degree_entropy(mol),
        "H_edge": edge_degree_entropy(mol),
    }
# =================== UI ===================

st.set_page_config(
    page_title="QSPR Topological & Entropy Descriptor Dashboard",
    page_icon="🧪",
    layout="wide",
)
st.markdown("""
<style>

/* -------- GLOBAL -------- */
html, body, .stApp {
    background-color: #f8fafc;
    color: #111827;
    font-family: 'Inter', sans-serif;
}

/* -------- TITLE -------- */
h1 {
    font-weight: 700;
    color: #1e293b;
}

/* -------- SECTION TITLE -------- */
.section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #1e3a8a;
    margin-bottom: 0.5rem;
}

/* -------- CARD -------- */
.qspr-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}

/* -------- BUTTON -------- */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 500;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* -------- DATAFRAME -------- */
div[data-testid="stDataFrame"] table {
    border-radius: 10px;
    overflow: hidden;
}

div[data-testid="stDataFrame"] thead tr {
    background-color: #2563eb;
    color: white;
}

div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background-color: #f1f5f9;
}

/* -------- TABS -------- */
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2563eb;
    border-bottom: 2px solid #2563eb;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center; color:#1e3a8a;'>
⏣ QSPR Descriptor & Modeling Dashboard ⏣
</h1>

<p style='text-align:center; color:gray;'>
Degree & Entropy based Topological Indices • QSPR Modeling
</p>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["Descriptors", "QSPR"])


# --------- TAB 1: descriptors ---------
with tab1:
    st.markdown('<div class="qspr-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Topological & Entropy Indices</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Enter SMILES to compute degree-based and entropy-based descriptors.</p>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 1])

    with left:
        st.markdown("""
        **Instructions:**
        - Enter one or more SMILES strings (one per line)
        - Click **Calculate indices**
        - Results will include molecular structures and computed descriptors

        """)
        smi_input = st.text_area("SMILES input", height=100)
        compute_btn = st.button("Calculate indices")

        if compute_btn:
            smiles_list = [s.strip() for s in smi_input.splitlines() if s.strip()]
            if not smiles_list:
                st.warning("Please enter at least one SMILES string.")
            else:
                results = []
                for smi in smiles_list:
                    try:
                        idx_values = compute_all_indices_from_smiles(smi)
                        idx_values["SMILES"] = smi
                        results.append(idx_values)
                    except ValueError as e:
                        st.error(f"Error for SMILES '{smi}': {e}")
                if results:
                    df_res = pd.DataFrame(results)

                    # show images + per‑molecule table (optional, as you had)
                    for i, row in df_res.iterrows():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(row["Molecule"], caption=row["SMILES"])
                        with col2:
                            row_df = pd.DataFrame(row).T.drop(columns=["Molecule"])
                            st.dataframe(row_df, use_container_width=True)
                        st.markdown("---")

                    # combined table without images
                    st.subheader("All indices")
                    st.dataframe(df_res.drop(columns=["Molecule"]), use_container_width=True)

                    # CSV download for all SMILES
                    csv_smiles = df_res.drop(columns=["Molecule"]).to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=" Download (CSV)",
                        data=csv_smiles,
                        file_name="smiles_indices.csv",
                        mime="text/csv",
                        icon=":material/save_alt:"
                    )
    with right:
        st.markdown("**Available indices**")
        for key, (name, desc) in INDEX_INFO.items():
            st.write(f"- **{key}**: {name}")

    st.markdown('</div>', unsafe_allow_html=True)
    


# --------- TAB 2: QSPR ---------

with tab2:
    st.markdown('<div class="qspr-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">QSPR Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="muted">Upload a dataset with indices and properties, then fit linear and polynomial models.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload CSV with indices + properties (for example: qspr_full.csv)",
        type=["csv"],
    )

    if uploaded is not None:
        data = pd.read_csv(uploaded)

        st.subheader("Dataset preview")
        st.dataframe(data.head(), use_container_width=True)
        st.subheader(" Correlation Heatmap")

        corr = data.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)

        st.pyplot(fig)
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="Download Heatmap (PNG)",
            data=buf,
            file_name="correlation_heatmap.png",
            mime="image/png",
            icon=":material/save_alt:"
        )

        # ---- detect index columns and property columns ----
        possible_index_cols = [
            "M1", "M2", "Randic", "ABC", "AZI",
            "Harmonic", "GA", "SumConn",
            "Forgotten", "SSD", "H_deg", "H_edge",
        ]

        index_cols = [c for c in possible_index_cols if c in data.columns]

        numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
        property_candidates = [c for c in numeric_cols if c not in index_cols]

        if not property_candidates:
            st.error("No numeric property columns found (only indices present).")
        else:
            property_col = st.selectbox(
                "Select property (y-axis)",
                property_candidates,
                key="qspr_property",
            )

            # -------- linear QSPR analysis --------
            def fit_linear_local(x, y):
                coeffs = np.polyfit(x, y, 1)
                p = np.poly1d(coeffs)
                y_pred = p(x)
                r2 = r2_score(y, y_pred)
                return coeffs, r2

            if st.button("Run QSPR analysis"):
                y = data[property_col].values
                results = []
                for idx in index_cols:
                    x = data[idx].values
                    coeffs, r2 = fit_linear_local(x, y)
                    a, b = coeffs[1], coeffs[0]
                    eq = f"{property_col} = {a:.3f} + {b:.3f}·{idx}"
                    results.append({"Index": idx, "R2_linear": r2, "Equation": eq})
                res_df = pd.DataFrame(results).sort_values("R2_linear", ascending=False)
                st.subheader("Linear models per index")
                st.dataframe(res_df, use_container_width=True)
                csv = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download (CSV)",
                    data=csv,
                    file_name="qspr_results.csv",
                    mime="text/csv",
                    icon=":material/save_alt:"
                )

                # -------- CORRELATION WITH PROPERTY --------
                st.subheader("Correlation with Property")

                # keep only numeric columns
                numeric_data = data.select_dtypes(include=[np.number])

                # ensure property is numeric
                numeric_data[property_col] = pd.to_numeric(numeric_data[property_col], errors='coerce')

                # compute correlation
                corr_target = numeric_data.corr()[property_col].drop(property_col)

                # sort values
                corr_target = corr_target.sort_values(ascending=False)

                # OPTIONAL: show top 3
                top3 = corr_target.head(3)
                st.success(f"Top descriptors: {', '.join(top3.index)}")

                fig, ax = plt.subplots(figsize=(8, 6))
                corr_bars = corr_target.sort_values().plot(kind="barh", ax=ax)

                # add exact R values on each bar
                for i, (idx, r_value) in enumerate(corr_target.sort_values().items()):
                    ax.text(r_value + 0.01, i, f"{r_value:.3f}", va="center", fontweight="bold")

                ax.set_xlabel("Correlation with property (R)")
                ax.set_ylabel("Descriptors")
                ax.set_title("Topological Descriptor Influence")
                ax.grid(axis="x", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)
                st.download_button(
                    label="Download Plot (PNG)",
                    data=buf,
                    file_name="correlation_bars.png",
                    mime="image/png",
                    icon=":material/save_alt:"
                )


                best_model = res_df.iloc[0]

                col1, col2 = st.columns(2)

                col1.metric("Best Index", best_model["Index"])
                col2.metric("Best R²", f"{best_model['R2_linear']:.3f}")

                st.markdown("### Machine Learning Model (Random Forest)")

                # X = input features (all indices)
                X = data[index_cols]

                # y = target property
                y = data[property_col]

                # split data (train/test)
                X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                )

                # create model
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
                scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
                # train model
                rf_model.fit(X_train, y_train)

                # predict
                y_pred = rf_model.predict(X_test)

                # evaluate
                r2_rf = r2_score(y_test, y_pred)

                # show result
                st.metric("Random Forest R² (Test Data)", f"{r2_rf:.3f}")
                if r2_rf < best_model["R2_linear"]:
                    st.info("Linear model performs better → indicates a strong linear relationship between descriptor and property.")
                else:
                    st.success("Random Forest captures additional non-linear patterns.")

            # -------- scatter plot with linear fit --------
            st.subheader("Scatter plot with linear fit")
            idx_for_plot = st.selectbox(
                "Select index (x-axis)",
                index_cols,
                key="qspr_index_scatter",
            )

            if st.button("Show scatter plot"):
                x = data[idx_for_plot].values
                y = data[property_col].values

                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
                ax.scatter(x, y, color="#2563eb", edgecolor="white", linewidth=0.6, label="Data")
                m, c = np.polyfit(x, y, 1)
                ax.plot(x, m * x + c, color="#0ea5e9", linewidth=2.5, label="Linear fit")
                ax.set_xlabel(idx_for_plot)
                ax.set_ylabel(property_col)
                ax.set_title(f"{property_col} vs {idx_for_plot}", fontsize=12, weight='bold')
                ax.grid(alpha=0.25)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend()
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)

                st.download_button(
                    label="Download (PNG)",
                    data=buf,
                    file_name=f"{property_col}_vs_{idx_for_plot}.png",
                    mime="image/png",
                    icon=":material/save_alt:"
                )

            # -------- polynomial fit --------
            st.subheader("Polynomial fit (linear / quadratic / cubic)")
            idx_poly = st.selectbox(
                "Select index for polynomial fit (x-axis)",
                index_cols,
                key="qspr_index_poly",
            )
            degree_choice = st.selectbox(
                "Select polynomial degree",
                ["Linear (1)", "Quadratic (2)", "Cubic (3)"],
                key="qspr_degree_poly",
            )
            degree_map = {"Linear (1)": 1, "Quadratic (2)": 2, "Cubic (3)": 3}
            degree = degree_map[degree_choice]

            if st.button("Run polynomial fit"):
                x = data[idx_poly].values
                y = data[property_col].values

                model, r2 = fit_poly(x, y, degree)
                eq_text = equation_str(model, degree, idx_poly, property_col)

                st.markdown(f"**Equation:** {eq_text}")
                st.markdown(f"**R² = {r2:.3f}**")

                x_plot = np.linspace(x.min(), x.max(), 200)
                y_plot = predict_poly(model, x_plot, degree)

                fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
                ax.scatter(x, y, color="#2563eb", edgecolor="white", linewidth=0.6)
                ax.plot(x_plot, y_plot, color="#10b981", linewidth=2.5)
                ax.set_xlabel(idx_poly)
                ax.set_ylabel(property_col)
                ax.set_title(f"{property_col} vs {idx_poly}", fontsize=12, weight='bold')
                ax.grid(alpha=0.25)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.legend()
                st.pyplot(fig)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                buf.seek(0)

                st.download_button(
                    label="Download (PNG)",
                    data=buf,
                    file_name=f"{property_col}_vs_{idx_for_plot}.png",
                    mime="image/png",
                    icon=":material/save_alt:"
                )

    st.markdown('</div>', unsafe_allow_html=True)
    
