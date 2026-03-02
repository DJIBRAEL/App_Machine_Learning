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
    st.markdown("<p style='color:rgba(255,255,255,0.7);font-size:0.9rem;margin-bottom:8px'>📂 Charger le dataset</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Fichier CSV (train.csv Kaggle)", type=['csv'], label_visibility='collapsed')
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
          <div class="value">+80</div>
          <div class="label">Features analysées</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# ENTRAÎNEMENT
# ─────────────────────────────────────────────
with st.spinner("⚙️  Entraînement des modèles en cours…"):
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    (df,
     pipe_rf_reg, pipe_dt_reg, reg_metrics,
     X_reg_test, y_reg_test,
     pipe_rf_clf, pipe_svm_clf, clf_metrics,
     X_clf_test, y_clf_test, labels_clf) = load_and_train(tmp_path)

    os.unlink(tmp_path)

# ─────────────────────────────────────────────
# MÉTRIQUES RAPIDES
# ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{df.shape[0]:,}</div>
        <div class="label">Propriétés</div></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="value">${reg_metrics['Random Forest']['MAE']:,.0f}</div>
        <div class="label">MAE Régression (RF)</div></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{reg_metrics['Random Forest']['R²']:.3f}</div>
        <div class="label">R² Régression (RF)</div></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="value">{clf_metrics['Random Forest']['Accuracy']:.1%}</div>
        <div class="label">Accuracy Classification</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prédiction Prix",
    "🏷️ Classification Type",
    "📊 Performance Régression",
    "📈 Performance Classification"
])

# ════════════════════════════════════════════
# TAB 1 — PRÉDICTION PRIX
# ════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-title">Estimation du prix de vente</span>', unsafe_allow_html=True)

    with st.form("form_reg"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**🏗️ Surface & Structure**")
            GrLivArea   = st.number_input("Surface habitable (pi²)", min_value=300, max_value=6000, value=1500)
            TotalBsmtSF = st.number_input("Surface sous-sol (pi²)",  min_value=0,   max_value=3000, value=800)
            LotArea     = st.number_input("Superficie du terrain (pi²)", min_value=1000, max_value=200000, value=10000)
            GarageArea  = st.number_input("Surface garage (pi²)",    min_value=0,   max_value=1400, value=400)

        with col_b:
            st.markdown("**🛏️ Pièces & Équipements**")
            BedroomAbvGr = st.slider("Chambres (au-dessus RdC)",  1, 8, 3)
            FullBath     = st.slider("Salles de bain complètes",  0, 4, 2)
            TotRmsAbvGrd = st.slider("Total pièces (au-dessus RdC)", 2, 14, 7)
            GarageCars   = st.slider("Places de garage",          0, 4, 2)
            Fireplaces   = st.slider("Cheminées",                  0, 4, 1)
            PoolArea     = st.number_input("Surface piscine (pi²)", min_value=0, max_value=800, value=0)

        with col_c:
            st.markdown("**⭐ Qualité & Année**")
            OverallQual  = st.slider("Qualité globale (1-10)",    1, 10, 7)
            OverallCond  = st.slider("Condition globale (1-10)",  1, 10, 5)
            YearBuilt    = st.slider("Année de construction",     1872, 2010, 2000)
            YearRemodAdd = st.slider("Année de rénovation",       1950, 2010, 2005)
            Neighborhood = st.selectbox("Quartier", sorted(NEIGHBORHOODS))

        submitted_reg = st.form_submit_button("🔍 Estimer le prix")

    if submitted_reg:
        input_df = pd.DataFrame([{
            'GrLivArea': GrLivArea, 'TotalBsmtSF': TotalBsmtSF, 'LotArea': LotArea,
            'BedroomAbvGr': BedroomAbvGr, 'FullBath': FullBath, 'TotRmsAbvGrd': TotRmsAbvGrd,
            'OverallQual': OverallQual, 'OverallCond': OverallCond, 'YearBuilt': YearBuilt,
            'YearRemodAdd': YearRemodAdd, 'GarageCars': GarageCars, 'GarageArea': GarageArea,
            'PoolArea': PoolArea, 'Fireplaces': Fireplaces, 'Neighborhood': Neighborhood
        }])

        pred_rf = pipe_rf_reg.predict(input_df)[0]
        pred_dt = pipe_dt_reg.predict(input_df)[0]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown(f"""
            <div class="price-result">
              <div class="label">Random Forest (recommandé)</div>
              <div class="price">${pred_rf:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#2C3E50,#4A5568);color:white;border-radius:16px;padding:32px;text-align:center">
              <div style="font-size:0.9rem;opacity:0.7;text-transform:uppercase;letter-spacing:1px">Decision Tree</div>
              <div style="font-family:'Playfair Display',serif;font-size:3.2rem;font-weight:900;margin:8px 0 0 0">${pred_dt:,.0f}</div>
            </div>""", unsafe_allow_html=True)

        diff = abs(pred_rf - pred_dt)
        st.markdown(f"""
        <div class="info-box">
          📌 Écart entre les deux modèles : <strong>${diff:,.0f}</strong> — 
          Nous recommandons d'utiliser l'estimation <strong>Random Forest</strong> (R² = {reg_metrics['Random Forest']['R²']:.3f}).
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════
# TAB 2 — CLASSIFICATION TYPE
# ════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-title">Classification du type de bien</span>', unsafe_allow_html=True)

    with st.form("form_clf"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**🏗️ Caractéristiques physiques**")
            c_GrLivArea    = st.number_input("Surface habitable (pi²)", min_value=300, max_value=6000, value=1500, key='c1')
            c_TotRmsAbvGrd = st.slider("Total pièces (au-dessus RdC)", 2, 14, 7, key='c2')
            c_OverallQual  = st.slider("Qualité globale (1-10)", 1, 10, 7, key='c3')
            c_YearBuilt    = st.slider("Année de construction", 1872, 2010, 2000, key='c4')
            c_GarageCars   = st.slider("Places de garage", 0, 4, 2, key='c5')

        with col_b:
            st.markdown("**📍 Localisation & Style**")
            c_Neighborhood = st.selectbox("Quartier", sorted(NEIGHBORHOODS), key='c6')
            c_HouseStyle   = st.selectbox("Style architectural", HOUSESTYLES, key='c7')
            model_choice   = st.radio("Modèle de classification", ['Random Forest', 'SVM'], horizontal=True)

        submitted_clf = st.form_submit_button("🏷️ Classifier le bien")

    if submitted_clf:
        input_clf = pd.DataFrame([{
            'GrLivArea': c_GrLivArea, 'TotRmsAbvGrd': c_TotRmsAbvGrd,
            'OverallQual': c_OverallQual, 'YearBuilt': c_YearBuilt,
            'GarageCars': c_GarageCars,
            'Neighborhood': c_Neighborhood, 'HouseStyle': c_HouseStyle
        }])

        model = pipe_rf_clf if model_choice == 'Random Forest' else pipe_svm_clf
        pred_type = model.predict(input_clf)[0]
        label = BLDGTYPE_LABELS.get(pred_type, pred_type)

        proba_text = ""
        if model_choice == 'Random Forest':
            proba = pipe_rf_clf.predict_proba(input_clf)[0]
            classes = pipe_rf_clf.classes_
            proba_df = pd.DataFrame({'Type': classes, 'Probabilité': proba}).sort_values('Probabilité', ascending=False)
            proba_text = "  ".join([f"{r['Type']}: {r['Probabilité']:.1%}" for _, r in proba_df.head(3).iterrows()])

        st.markdown(f"""
        <div class="class-result">
          <div class="label">Type prédit — {model_choice}</div>
          <div class="btype">{label}</div>
          <div style="margin-top:12px;font-size:0.85rem;opacity:0.85">Code : {pred_type}</div>
        </div>""", unsafe_allow_html=True)

        if proba_text:
            st.markdown(f"""
            <div class="info-box">
              📊 <strong>Probabilités (top 3) :</strong> {proba_text}
            </div>""", unsafe_allow_html=True)

        # Mini barre de proba
        if model_choice == 'Random Forest':
            fig_p, ax_p = plt.subplots(figsize=(7, 2.5))
            fig_p.patch.set_facecolor('#F7F3EE')
            ax_p.set_facecolor('#F7F3EE')
            colors = ['#C9A84C' if c == pred_type else '#CBD5E0' for c in proba_df['Type']]
            ax_p.barh(proba_df['Type'], proba_df['Probabilité'], color=colors, edgecolor='none', height=0.5)
            ax_p.set_xlim(0, 1)
            ax_p.set_xlabel('Probabilité')
            ax_p.set_title('Distribution des probabilités de classe', fontsize=10, pad=8)
            for i, (_, row) in enumerate(proba_df.iterrows()):
                ax_p.text(row['Probabilité'] + 0.01, i, f"{row['Probabilité']:.1%}", va='center', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close(fig_p)

# ════════════════════════════════════════════
# TAB 3 — PERFORMANCE RÉGRESSION
# ════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-title">Analyse des modèles de régression</span>', unsafe_allow_html=True)

    # Tableau des métriques
    df_reg_metrics = pd.DataFrame({
        'Modèle':    list(reg_metrics.keys()),
        'MAE ($)':   [f"{v['MAE']:,.0f}" for v in reg_metrics.values()],
        'RMSE ($)':  [f"{v['RMSE']:,.0f}" for v in reg_metrics.values()],
        'R²':        [f"{v['R²']:.4f}" for v in reg_metrics.values()],
    })
    st.table(df_reg_metrics.set_index('Modèle'))

    # Scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#F7F3EE')
    colors_m = {'Random Forest': '#1A6B72', 'Decision Tree': '#C9A84C'}

    for ax, (name, metrics) in zip(axes, reg_metrics.items()):
        y_pred = metrics['y_pred']
        ax.set_facecolor('#FAFAFA')
        ax.scatter(y_reg_test, y_pred, alpha=0.35, s=18,
                   color=colors_m[name], edgecolors='none')
        lim = [y_reg_test.min(), y_reg_test.max()]
        ax.plot(lim, lim, 'k--', linewidth=1.2, alpha=0.5)
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Prix réel ($)', fontsize=9)
        ax.set_ylabel('Prix prédit ($)', fontsize=9)
        ax.text(0.05, 0.93, f'R² = {metrics["R²"]:.4f}', transform=ax.transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.suptitle('Prédictions vs Valeurs réelles — SalePrice', fontsize=13, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Feature importance RF
    st.markdown("**Importance des features — Random Forest Regressor**")
    rf_step = pipe_rf_reg.named_steps['model']
    feat_names = REG_NUM_FEATURES + REG_CAT_FEATURES
    importances = pd.Series(rf_step.feature_importances_, index=feat_names).sort_values()

    fig_i, ax_i = plt.subplots(figsize=(9, 5))
    fig_i.patch.set_facecolor('#F7F3EE')
    ax_i.set_facecolor('#FAFAFA')
    colors_i = ['#C9A84C' if v > 0.1 else '#1A6B72' for v in importances]
    importances.plot(kind='barh', ax=ax_i, color=colors_i, edgecolor='none')
    ax_i.set_title('Feature Importance', fontsize=11, fontweight='bold')
    ax_i.set_xlabel('Importance relative')
    plt.tight_layout()
    st.pyplot(fig_i)
    plt.close(fig_i)

# ════════════════════════════════════════════
# TAB 4 — PERFORMANCE CLASSIFICATION
# ════════════════════════════════════════════
with tab4:
    st.markdown('<span class="section-title">Analyse des modèles de classification</span>', unsafe_allow_html=True)

    # Tableau métriques
    df_clf_metrics = pd.DataFrame({
        'Modèle':    list(clf_metrics.keys()),
        'Accuracy':  [f"{v['Accuracy']:.4f}" for v in clf_metrics.values()],
        'F1 (wt.)':  [f"{v['F1']:.4f}" for v in clf_metrics.values()],
    })
    st.table(df_clf_metrics.set_index('Modèle'))

    model_sel = st.selectbox("Sélectionner le modèle à analyser :", ['Random Forest', 'SVM'])

    chosen = clf_metrics[model_sel]

    col_cm, col_rp = st.columns([1, 1])
    with col_cm:
        st.markdown(f"**Matrice de confusion — {model_sel}**")
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        fig_cm.patch.set_facecolor('#F7F3EE')
        sns.heatmap(chosen['cm'], annot=True, fmt='d', cmap='YlGnBu',
                    xticklabels=labels_clf, yticklabels=labels_clf,
                    ax=ax_cm, linewidths=0.5)
        ax_cm.set_xlabel('Prédit', fontsize=10)
        ax_cm.set_ylabel('Réel', fontsize=10)
        ax_cm.set_title('Confusion Matrix', fontsize=11, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with col_rp:
        st.markdown(f"**Rapport de classification — {model_sel}**")
        st.code(chosen['report'], language=None)

    # Distribution prédictions vs réel
    fig_d, axes_d = plt.subplots(1, 2, figsize=(12, 4))
    fig_d.patch.set_facecolor('#F7F3EE')
    for ax, (title, data) in zip(axes_d, [('Réel', y_clf_test), ('Prédit', chosen['y_pred'])]):
        counts = pd.Series(data).value_counts().sort_index()
        ax.set_facecolor('#FAFAFA')
        bars = ax.bar(counts.index, counts.values,
                      color='#1A6B72' if title=='Réel' else '#C9A84C', edgecolor='none')
        ax.set_title(f'Distribution {title}', fontsize=11, fontweight='bold')
        ax.set_xlabel('BldgType')
        ax.set_ylabel('Nombre')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    str(int(bar.get_height())), ha='center', fontsize=9)
    plt.suptitle(f'Distribution des classes — {model_sel}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_d)
    plt.close(fig_d)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:40px;padding:20px;opacity:0.4;font-size:0.8rem">
  ImmoPredict — Plateforme ML · Régression & Classification · Dataset Kaggle House Prices
</div>
""", unsafe_allow_html=True)