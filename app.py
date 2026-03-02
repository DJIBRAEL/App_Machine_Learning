import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix, classification_report
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ImmoPredict — Plateforme ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --cream: #F7F3EE;
    --dark: #1A1A2E;
    --gold: #C9A84C;
    --rust: #C0392B;
    --teal: #1A6B72;
    --card-bg: #FFFFFF;
    --border: #E8E0D5;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--cream);
    color: var(--dark);
}

.main { background-color: var(--cream); }

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

/* Hero Banner */
.hero {
    background: linear-gradient(135deg, #1A1A2E 0%, #16213E 60%, #1A6B72 100%);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(201,168,76,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    color: white;
    font-size: 2.8rem;
    font-weight: 900;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.hero p {
    color: rgba(255,255,255,0.72);
    font-size: 1.1rem;
    margin: 0;
    font-weight: 300;
}
.hero .badge {
    display: inline-block;
    background: var(--gold);
    color: var(--dark);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 14px;
}

/* Cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 24px 20px;
    border: 1px solid var(--border);
    text-align: center;
    box-shadow: 0 2px 12px rgba(26,26,46,0.06);
}
.metric-card .value {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--dark);
}
.metric-card .label {
    font-size: 0.78rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Section titles */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--dark);
    padding-bottom: 8px;
    border-bottom: 3px solid var(--gold);
    margin-bottom: 24px;
    display: inline-block;
}

/* Prediction result */
.price-result {
    background: linear-gradient(135deg, #1A6B72, #1A1A2E);
    color: white;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.price-result .label { font-size: 0.9rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 1px; }
.price-result .price {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    margin: 8px 0 0 0;
}

.class-result {
    background: linear-gradient(135deg, #C9A84C, #a07830);
    color: white;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.class-result .label { font-size: 0.9rem; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px; }
.class-result .btype {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 900;
    margin: 8px 0 0 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1A1A2E;
}
[data-testid="stSidebar"] .css-1d391kg, 
[data-testid="stSidebar"] * {
    color: white;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: rgba(255,255,255,0.85) !important;
    font-size: 0.85rem;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}

/* Info box */
.info-box {
    background: #EEF7F8;
    border-left: 4px solid var(--teal);
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 12px 0;
    font-size: 0.9rem;
    color: var(--dark);
}

/* Stbutton */
.stButton > button {
    background: linear-gradient(135deg, var(--teal), #0d4a50);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.5px;
    transition: transform 0.1s;
    width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); }

/* Sidebar upload box premium */
.upload-box {
    background: linear-gradient(135deg, #16213E, #1A6B72);
    padding: 14px 16px;
    border-radius: 10px;
    border-left: 4px solid #C9A84C;
    margin-bottom: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.15);
}

.upload-box span {
    color: white;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* File uploader container styling */
section[data-testid="stFileUploader"] {
    background: #16213E;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #C9A84C;
}

/* ── Browse files button ── */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, var(--gold), #a07830) !important;
    color: var(--dark) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 8px 20px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
[data-testid="stFileUploader"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 14px rgba(201, 168, 76, 0.45) !important;
    color: var(--dark) !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────
REG_NUM_FEATURES = ['GrLivArea','TotalBsmtSF','LotArea','BedroomAbvGr','FullBath',
                    'TotRmsAbvGrd','OverallQual','OverallCond','YearBuilt',
                    'YearRemodAdd','GarageCars','GarageArea','PoolArea','Fireplaces']
REG_CAT_FEATURES = ['Neighborhood']
REG_TARGET       = 'SalePrice'

CLF_NUM_FEATURES = ['GrLivArea','TotRmsAbvGrd','OverallQual','YearBuilt','GarageCars']
CLF_CAT_FEATURES = ['Neighborhood','HouseStyle']
CLF_TARGET       = 'BldgType'

BLDGTYPE_LABELS = {
    '1Fam':   '🏠 Maison unifamiliale',
    '2FmCon': '🏘️ Maison 2 familles (convertie)',
    'Duplx':  '🏢 Duplex',
    'TwnhsE': '🏙️ Maison de ville (angle)',
    'TwnhsI': '🏙️ Maison de ville (intérieur)',
}

NEIGHBORHOODS = ['NAmes','CollgCr','OldTown','Edwards','Somerst','NridgHt','Gilbert',
                 'Sawyer','NWAmes','SawyerW','Mitchel','BrkSide','Crawfor','IDOTRR',
                 'Timber','NoRidge','StoneBr','SWISU','ClearCr','MeadowV','Blmngtn',
                 'BrDale','Veenker','NPkVill','Blueste']

HOUSESTYLES = ['1Story','2Story','1.5Fin','1.5Unf','SFoyer','SLvl','2.5Fin','2.5Unf']

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def build_preprocessor(num_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer,  num_features),
        ('cat', categorical_transformer, cat_features)
    ])

@st.cache_resource(show_spinner=False)
def load_and_train(csv_path):
    df = pd.read_csv(csv_path)

    # ── Régression ──────────────────────────────────────────
    X_reg = df[REG_NUM_FEATURES + REG_CAT_FEATURES].copy()
    y_reg = df[REG_TARGET].copy()
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    pipe_rf_reg = Pipeline([
        ('preprocessor', build_preprocessor(REG_NUM_FEATURES, REG_CAT_FEATURES)),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=None,
                                        min_samples_split=2, max_features='sqrt',
                                        random_state=42, n_jobs=-1))
    ])
    pipe_dt_reg = Pipeline([
        ('preprocessor', build_preprocessor(REG_NUM_FEATURES, REG_CAT_FEATURES)),
        ('model', DecisionTreeRegressor(max_depth=10, min_samples_split=5,
                                         min_samples_leaf=2, random_state=42))
    ])
    pipe_rf_reg.fit(X_reg_train, y_reg_train)
    pipe_dt_reg.fit(X_reg_train, y_reg_train)

    y_pred_rf_reg = pipe_rf_reg.predict(X_reg_test)
    y_pred_dt_reg = pipe_dt_reg.predict(X_reg_test)

    reg_metrics = {
        'Random Forest': {
            'MAE':  mean_absolute_error(y_reg_test, y_pred_rf_reg),
            'RMSE': np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg)),
            'R²':   r2_score(y_reg_test, y_pred_rf_reg),
            'y_pred': y_pred_rf_reg,
        },
        'Decision Tree': {
            'MAE':  mean_absolute_error(y_reg_test, y_pred_dt_reg),
            'RMSE': np.sqrt(mean_squared_error(y_reg_test, y_pred_dt_reg)),
            'R²':   r2_score(y_reg_test, y_pred_dt_reg),
            'y_pred': y_pred_dt_reg,
        },
    }

    # ── Classification ──────────────────────────────────────
    X_clf = df[CLF_NUM_FEATURES + CLF_CAT_FEATURES].copy()
    y_clf = df[CLF_TARGET].copy()
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    pipe_rf_clf = Pipeline([
        ('preprocessor', build_preprocessor(CLF_NUM_FEATURES, CLF_CAT_FEATURES)),
        ('model', RandomForestClassifier(n_estimators=200, max_depth=10,
                                          class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    pipe_svm_clf = Pipeline([
        ('preprocessor', build_preprocessor(CLF_NUM_FEATURES, CLF_CAT_FEATURES)),
        ('model', SVC(C=10, kernel='rbf', gamma='scale',
                      class_weight='balanced', random_state=42))
    ])
    pipe_rf_clf.fit(X_clf_train, y_clf_train)
    pipe_svm_clf.fit(X_clf_train, y_clf_train)

    y_pred_rf_clf  = pipe_rf_clf.predict(X_clf_test)
    y_pred_svm_clf = pipe_svm_clf.predict(X_clf_test)

    clf_metrics = {
        'Random Forest': {
            'Accuracy': accuracy_score(y_clf_test, y_pred_rf_clf),
            'F1':       f1_score(y_clf_test, y_pred_rf_clf, average='weighted', zero_division=0),
            'cm':       confusion_matrix(y_clf_test, y_pred_rf_clf),
            'report':   classification_report(y_clf_test, y_pred_rf_clf, zero_division=0),
            'y_pred':   y_pred_rf_clf,
        },
        'SVM': {
            'Accuracy': accuracy_score(y_clf_test, y_pred_svm_clf),
            'F1':       f1_score(y_clf_test, y_pred_svm_clf, average='weighted', zero_division=0),
            'cm':       confusion_matrix(y_clf_test, y_pred_svm_clf),
            'report':   classification_report(y_clf_test, y_pred_svm_clf, zero_division=0),
            'y_pred':   y_pred_svm_clf,
        },
    }
    labels_clf = sorted(y_clf.unique())

    return (df,
            pipe_rf_reg, pipe_dt_reg, reg_metrics,
            X_reg_test, y_reg_test,
            pipe_rf_clf, pipe_svm_clf, clf_metrics,
            X_clf_test, y_clf_test, labels_clf)

# ─────────────────────────────────────────────
# SIDEBAR — Chargement du CSV
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px'>
      <span style='font-family:Playfair Display,serif;font-size:1.5rem;font-weight:900;color:white'>
        🏠 ImmoPredict
      </span><br>
      <span style='font-size:0.78rem;color:rgba(255,255,255,0.5);letter-spacing:1px'>ML PLATFORM</span>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div class="upload-box">
        <span>📂 Charger le dataset</span>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Fichier CSV (train.csv Kaggle)",
        type=['csv'],
        label_visibility='collapsed'
    )
    st.markdown("<p style='color:rgba(255,255,255,0.45);font-size:0.76rem;margin-top:8px'>Dataset Kaggle : House Prices - Advanced Regression Techniques</p>", unsafe_allow_html=True)
    st.divider()
    st.markdown("<p style='color:rgba(255,255,255,0.6);font-size:0.82rem'>Les modèles sont entraînés automatiquement à l'importation.</p>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">Plateforme Machine Learning</div>
  <h1>ImmoPredict</h1>
  <p>Estimation du prix & classification automatique des biens immobiliers<br>
  Modèles : Random Forest · Decision Tree · SVM</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GUARD — Aucun fichier
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
      ⬅️  <strong>Commencez par charger votre fichier CSV</strong> dans le panneau latéral gauche.<br>
      Le dataset attendu est <em>train.csv</em> du challenge Kaggle 
      <em>House Prices – Advanced Regression Techniques</em>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
          <div class="value">2</div>
          <div class="label">Tâches ML</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
          <div class="value">3</div>
          <div class="label">Modèles entraînés</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
          <div class="value">1</div>
          <div class="label">Dataset chargé</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# CHARGEMENT & ENTRAÎNEMENT
# ─────────────────────────────────────────────
with st.spinner("⏳ Entraînement des modèles en cours…"):
    (df,
     pipe_rf_reg, pipe_dt_reg, reg_metrics,
     X_reg_test, y_reg_test,
     pipe_rf_clf, pipe_svm_clf, clf_metrics,
     X_clf_test, y_clf_test, labels_clf) = load_and_train(uploaded_file)

# ─────────────────────────────────────────────
# MÉTRIQUES RAPIDES
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
      <div class="value">{len(df):,}</div>
      <div class="label">Propriétés</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
      <div class="value">{reg_metrics['Random Forest']['R²']:.3f}</div>
      <div class="label">R² Régression (RF)</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
      <div class="value">{clf_metrics['Random Forest']['Accuracy']:.3f}</div>
      <div class="label">Accuracy Classif. (RF)</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
      <div class="value">{clf_metrics['SVM']['F1']:.3f}</div>
      <div class="label">F1 Classif. (SVM)</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Analyse exploratoire",
    "🤖 Performance des modèles",
    "🔮 Prédictions"
])

# ══════════════════════════════════════════════
# TAB 1 — EDA
# ══════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-title">Exploration des données</span>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df['SalePrice'], bins=50, color='#1A6B72', edgecolor='white', linewidth=0.4)
        ax.set_title('Distribution des prix de vente', fontsize=13, fontweight='bold')
        ax.set_xlabel('Prix ($)')
        ax.set_ylabel('Fréquence')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df['GrLivArea'], df['SalePrice'],
                   alpha=0.35, s=18, c='#C9A84C', edgecolors='none')
        ax.set_title('Surface habitable vs Prix', fontsize=13, fontweight='bold')
        ax.set_xlabel('Surface habitable (pi²)')
        ax.set_ylabel('Prix de vente ($)')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)

    col_c, col_d = st.columns(2)

    with col_c:
        fig, ax = plt.subplots(figsize=(7, 4))
        top_nb = df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).head(10)
        ax.barh(top_nb.index[::-1], top_nb.values[::-1], color='#1A1A2E', edgecolor='none')
        ax.set_title('Top 10 quartiers — Prix médian', fontsize=13, fontweight='bold')
        ax.set_xlabel('Prix médian ($)')
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_d:
        fig, ax = plt.subplots(figsize=(7, 4))
        qual_price = df.groupby('OverallQual')['SalePrice'].median()
        ax.bar(qual_price.index, qual_price.values, color='#1A6B72', edgecolor='white', linewidth=0.4)
        ax.set_title('Qualité globale vs Prix médian', fontsize=13, fontweight='bold')
        ax.set_xlabel('Note de qualité (1–10)')
        ax.set_ylabel('Prix médian ($)')
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Matrice de corrélation — variables numériques clés**")
    corr_cols = ['SalePrice','GrLivArea','TotalBsmtSF','OverallQual',
                 'GarageArea','YearBuilt','TotRmsAbvGrd','FullBath']
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='YlOrBr',
                linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Corrélations entre variables clés', fontsize=13, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-title">Performance des modèles</span>', unsafe_allow_html=True)

    # ── Régression ──────────────────────────────
    st.markdown("### 📈 Régression — Prédiction du prix")
    r1, r2 = st.columns(2)
    for col, (name, m) in zip([r1, r2], reg_metrics.items()):
        with col:
            st.markdown(f"**{name}**")
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("MAE", f"${m['MAE']:,.0f}")
            mc2.metric("RMSE", f"${m['RMSE']:,.0f}")
            mc3.metric("R²", f"{m['R²']:.4f}")

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_reg_test, m['y_pred'], alpha=0.35, s=14,
                       c='#1A6B72' if name == 'Random Forest' else '#C9A84C',
                       edgecolors='none')
            lims = [min(y_reg_test.min(), m['y_pred'].min()),
                    max(y_reg_test.max(), m['y_pred'].max())]
            ax.plot(lims, lims, 'r--', lw=1.2, alpha=0.7)
            ax.set_xlabel('Valeur réelle ($)')
            ax.set_ylabel('Valeur prédite ($)')
            ax.set_title(f'{name} — Réel vs Prédit', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.4)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # ── Classification ──────────────────────────
    st.markdown("### 🏷️ Classification — Type de bâtiment")
    clf1, clf2 = st.columns(2)
    for col, (name, m) in zip([clf1, clf2], clf_metrics.items()):
        with col:
            st.markdown(f"**{name}**")
            ca, cf = st.columns(2)
            ca.metric("Accuracy", f"{m['Accuracy']:.4f}")
            cf.metric("F1 (weighted)", f"{m['F1']:.4f}")

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(m['cm'], annot=True, fmt='d',
                        cmap='Blues' if name == 'Random Forest' else 'Oranges',
                        xticklabels=labels_clf, yticklabels=labels_clf,
                        linewidths=0.5, ax=ax)
            ax.set_xlabel('Prédit')
            ax.set_ylabel('Réel')
            ax.set_title(f'{name} — Matrice de confusion', fontsize=11, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander(f"Rapport détaillé — {name}"):
                st.code(m['report'])

# ══════════════════════════════════════════════
# TAB 3 — PRÉDICTIONS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-title">Faire une prédiction</span>', unsafe_allow_html=True)

    pred_col, res_col = st.columns([1.4, 1])

    with pred_col:
        st.markdown("#### Caractéristiques du bien")

        p1, p2 = st.columns(2)
        with p1:
            gr_liv   = st.number_input("Surface habitable (pi²)", 400, 6000, 1500, 50)
            bsmt     = st.number_input("Surface sous-sol (pi²)",  0, 4000, 800,  50)
            lot      = st.number_input("Surface terrain (pi²)",   1000, 200000, 8000, 500)
            bedrooms = st.slider("Chambres", 0, 8, 3)
            fullbath = st.slider("Salles de bain complètes", 0, 4, 2)
            totrms   = st.slider("Total pièces (hors bains)", 2, 14, 7)
        with p2:
            qual     = st.slider("Qualité globale (1–10)", 1, 10, 6)
            cond     = st.slider("État général (1–10)",    1, 10, 5)
            yr_built = st.slider("Année construction", 1872, 2010, 1990)
            yr_remod = st.slider("Année rénovation",   1950, 2010, 2000)
            gar_cars = st.slider("Capacité garage (voitures)", 0, 4, 2)
            gar_area = st.number_input("Surface garage (pi²)", 0, 1500, 480, 20)
            pool     = st.number_input("Surface piscine (pi²)", 0, 800, 0, 10)
            fires    = st.slider("Cheminées", 0, 4, 1)

        neighborhood = st.selectbox("Quartier", NEIGHBORHOODS)
        housestyle   = st.selectbox("Style de maison", HOUSESTYLES)

        predict_btn = st.button("🔮 Lancer la prédiction")

    with res_col:
        st.markdown("#### Résultats")
        if predict_btn:
            # Régression
            input_reg = pd.DataFrame([{
                'GrLivArea': gr_liv, 'TotalBsmtSF': bsmt, 'LotArea': lot,
                'BedroomAbvGr': bedrooms, 'FullBath': fullbath,
                'TotRmsAbvGrd': totrms, 'OverallQual': qual, 'OverallCond': cond,
                'YearBuilt': yr_built, 'YearRemodAdd': yr_remod,
                'GarageCars': gar_cars, 'GarageArea': gar_area,
                'PoolArea': pool, 'Fireplaces': fires,
                'Neighborhood': neighborhood
            }])
            price_rf = pipe_rf_reg.predict(input_reg)[0]
            price_dt = pipe_dt_reg.predict(input_reg)[0]

            # Classification
            input_clf = pd.DataFrame([{
                'GrLivArea': gr_liv, 'TotRmsAbvGrd': totrms,
                'OverallQual': qual, 'YearBuilt': yr_built,
                'GarageCars': gar_cars,
                'Neighborhood': neighborhood, 'HouseStyle': housestyle
            }])
            btype_rf  = pipe_rf_clf.predict(input_clf)[0]
            btype_svm = pipe_svm_clf.predict(input_clf)[0]

            st.markdown(f"""
            <div class="price-result">
              <div class="label">Prix estimé — Random Forest</div>
              <div class="price">${price_rf:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="price-result" style="background:linear-gradient(135deg,#2c3e50,#1A1A2E)">
              <div class="label">Prix estimé — Decision Tree</div>
              <div class="price">${price_dt:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="class-result">
              <div class="label">Type de bâtiment — Random Forest</div>
              <div class="btype">{BLDGTYPE_LABELS.get(btype_rf, btype_rf)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="class-result" style="background:linear-gradient(135deg,#1A6B72,#0d4a50)">
              <div class="label">Type de bâtiment — SVM</div>
              <div class="btype">{BLDGTYPE_LABELS.get(btype_svm, btype_svm)}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="info-box">
              Renseignez les caractéristiques du bien puis cliquez sur <strong>Lancer la prédiction</strong>.
            </div>
            """, unsafe_allow_html=True)
