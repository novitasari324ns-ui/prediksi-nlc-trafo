# ============================================================
#  PREDIKSI NO LOAD CURRENT TRAFO DISTRIBUSI 3 FASA
#  PT Bambang Djaja — Random Forest
#  v3.0 — 6 Kapasitas + Type Test Status ±30%
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib, os, io
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error

st.set_page_config(
    page_title="Prediksi NLC Trafo — PT Bambang Djaja",
    page_icon="⚡", layout="wide", initial_sidebar_state="expanded"
)

# ------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------
defaults = {
    'dark_mode': True,
    'kapasitas': 100,
    'tipe_inti': 'CRGO',
    'nl_loss': 129,
    'hasil_pred': None,
    'batch_rows': [{'kapasitas': 100, 'tipe_inti': 'CRGO', 'nl_loss': 129}],
    'batch_results': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------------------------------------------------
# TEMA
# ------------------------------------------------------------
D       = st.session_state.dark_mode
BG      = '#0f0f1a'  if D else '#f5f7fa'
BG2     = '#13132a'  if D else '#ffffff'
BG3     = '#1a1a2e'  if D else '#eef2f7'
BORDER  = '#2a2a4a'  if D else '#dde3ed'
TEXT    = '#e0e0e0'  if D else '#1a1a2e'
TEXT2   = '#aaaaaa'  if D else '#555555'
TEXT3   = '#555555'  if D else '#999999'
ACCENT  = '#185FA5'
ACCENT2 = '#60a5fa'  if D else '#185FA5'
GREEN   = '#0F6E56'
GREEN2  = '#4ade80'  if D else '#0F6E56'
WARN    = '#d97706'
WARN2   = '#fbbf24'  if D else '#d97706'
RED     = '#991b1b'
RED2    = '#f87171'  if D else '#dc2626'
PLOTBG  = '#16213e'  if D else '#ffffff'
PLOTFACE= '#1a1a2e'  if D else '#f5f7fa'
GRID    = '#2a2a4a'  if D else '#e0e0e0'
LEGBG   = '#13132a'  if D else '#ffffff'

# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html,body,.stApp {{ background:{BG} !important; font-family:'Inter',sans-serif; color:{TEXT} !important; }}
section[data-testid="stSidebar"] {{ background:{BG2} !important; border-right:1px solid {BORDER}; }}
section[data-testid="stSidebar"] * {{ color:{TEXT} !important; }}
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stNumberInput input {{ background:{BG3} !important; border-color:{BORDER} !important; color:{TEXT} !important; }}
#MainMenu,footer,header {{ visibility:hidden; }}
.stSelectbox div[data-baseweb="select"] > div,
.stNumberInput input {{ background:{BG3} !important; border-color:{BORDER} !important; color:{TEXT} !important; }}
div[data-baseweb="popover"] ul {{ background:{BG2} !important; color:{TEXT} !important; }}
[data-testid="metric-container"] {{ background:{BG2} !important; border:0.5px solid {BORDER} !important;
    border-radius:10px !important; padding:12px !important; }}
[data-testid="metric-container"] label {{ color:{TEXT2} !important; font-size:11px !important; }}
[data-testid="metric-container"] [data-testid="metric-value"] {{ color:{ACCENT2} !important; font-size:22px !important; }}
.stTabs [data-baseweb="tab-list"] {{ background:{BG2}; border-radius:10px; padding:4px; gap:4px; border:0.5px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ color:{TEXT2} !important; border-radius:8px !important; font-size:13px !important; }}
.stTabs [aria-selected="true"] {{ background:{ACCENT} !important; color:white !important; }}
div[data-testid="stButton"] > button {{
    background:linear-gradient(135deg,#185FA5,#0f4a8a) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:14px !important;
    width:100% !important; padding:10px !important; }}
div[data-testid="stButton"] > button:hover {{ background:linear-gradient(135deg,#1e6fc2,#185FA5) !important; }}
.stDataFrame {{ border:0.5px solid {BORDER}; border-radius:10px; }}
div[data-testid="stDownloadButton"] > button {{
    background:linear-gradient(135deg,#0F6E56,#0a4f3d) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-weight:600 !important; font-size:13px !important;
    width:100% !important; padding:8px !important; }}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# KONSTANTA
# ------------------------------------------------------------
CAPS = [50, 100, 160, 250, 400, 630]

# Tipe inti yang tersedia per kapasitas (kondisi nyata industri)
TIPE_PER_KVA = {
    50:  ['CRGO', 'Amorphous'],
    100: ['CRGO', 'Amorphous'],
    160: ['CRGO', 'Amorphous'],
    250: ['CRGO'],
    400: ['CRGO'],
    630: ['CRGO'],
}

LOSS_REF = {
    50:  {'CRGO': 84,  'Amorphous': 36},
    100: {'CRGO': 129, 'Amorphous': 40},
    160: {'CRGO': 183, 'Amorphous': 77},
    250: {'CRGO': 274},
    400: {'CRGO': 389},
    630: {'CRGO': 555},
}

# NLC range dari data aktual (min, max) — untuk info chart
NLC_REF = {
    50:  {'CRGO': (0.294, 0.472), 'Amorphous': (0.134, 0.200)},
    100: {'CRGO': (0.160, 0.419), 'Amorphous': (0.197, 0.258)},
    160: {'CRGO': (0.187, 0.241), 'Amorphous': (0.089, 0.126)},
    250: {'CRGO': (0.254, 0.344)},
    400: {'CRGO': (0.131, 0.172)},
    630: {'CRGO': (0.110, 0.136)},
}

# Type Test standard (dari kolom 'standar' di CSV)
TYPE_TEST = {
    50:  {'CRGO': 0.40, 'Amorphous': 0.16},
    100: {'CRGO': 0.31, 'Amorphous': 0.24},
    160: {'CRGO': 0.25, 'Amorphous': 0.10},
    250: {'CRGO': 0.45},
    400: {'CRGO': 0.18},
    630: {'CRGO': 0.14},
}

MODEL_PATH = 'model/model_trafo.pkl'
LE_PATH    = 'model/label_encoder.pkl'
DATA_PATH  = 'data_asli.csv'

# ------------------------------------------------------------
# LOAD MODEL & DATA
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
        return joblib.load(MODEL_PATH), joblib.load(LE_PATH)
    df = pd.read_csv(DATA_PATH)
    le = LabelEncoder()
    df['enc'] = le.fit_transform(df['tipe_inti'])
    X = df[['kapasitas','enc','nl_loss']].values
    y = df['nlc_persen'].values
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    os.makedirs('model', exist_ok=True)
    joblib.dump(rf, MODEL_PATH); joblib.dump(le, LE_PATH)
    return rf, le

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def get_metrics():
    df = load_data()
    le_tmp = LabelEncoder()
    df['enc'] = le_tmp.fit_transform(df['tipe_inti'])
    X = df[['kapasitas','enc','nl_loss']].values
    y = df['nlc_persen'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf, _ = load_model()
    yp = rf.predict(Xte)
    return (r2_score(yte, yp),
            np.sqrt(mean_squared_error(yte, yp)),
            mean_absolute_error(yte, yp),
            np.mean(np.abs(yte - yp) <= 0.01) * 100,
            Xte, yte, yp)

rf, le = load_model()
df     = load_data()
r2, rmse, mae, acc, X_test, y_test, y_pred = get_metrics()

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def set_plot():
    matplotlib.rcParams.update({
        'figure.facecolor': PLOTFACE, 'axes.facecolor': PLOTBG,
        'axes.edgecolor': BORDER, 'text.color': TEXT,
        'axes.labelcolor': TEXT, 'xtick.color': TEXT2,
        'ytick.color': TEXT2, 'grid.color': GRID,
        'legend.facecolor': LEGBG, 'legend.edgecolor': BORDER, 'legend.labelcolor': TEXT,
    })

def sec(title, icon=""):
    st.markdown(
        f'<div style="font-size:13px;font-weight:600;color:{TEXT};'
        f'border-bottom:2px solid {ACCENT};padding-bottom:6px;margin-bottom:14px;">'
        f'{icon} {title}</div>',
        unsafe_allow_html=True
    )

def predict_single(kva, tipe, loss):
    enc = le.transform([tipe])[0]
    return rf.predict([[kva, enc, loss]])[0]

def get_type_test_status(pred, kva, tipe):
    """
    Kembalikan (status_label, color, bg_color, border_color)
    berdasarkan 3 level type test:
      ✅ LULUS TYPE TEST        : NLC ≤ Type Test %
      ⚠️ DALAM TOLERANSI ±30%  : Type Test < NLC ≤ Type Test × 1.30
      ❌ TIDAK MEMENUHI STANDAR : NLC > Type Test × 1.30
    """
    tt = TYPE_TEST[kva][tipe]
    tol = tt * 1.30
    if pred <= tt:
        return (
            "✅  LULUS TYPE TEST",
            GREEN2,
            '#0d3d2e' if D else '#e6f7ef',
            GREEN,
            tt, tol
        )
    elif pred <= tol:
        return (
            "⚠️  DALAM TOLERANSI ±30%",
            WARN2,
            '#3d2a00' if D else '#fffbeb',
            WARN,
            tt, tol
        )
    else:
        return (
            "❌  TIDAK MEMENUHI STANDAR",
            RED2,
            '#3d0d0d' if D else '#fdecea',
            RED,
            tt, tol
        )

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    cl, cr = st.columns([3, 1])
    with cl:
        st.markdown(f'<div style="font-size:15px;font-weight:700;color:{ACCENT2};">⚡ NLC Predictor</div>',
                    unsafe_allow_html=True)
    with cr:
        if st.button("🌙" if D else "☀️", key="toggle_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;font-weight:600;color:{TEXT2};margin-bottom:12px;">🔧 INPUT DATA TRAFO</div>',
                unsafe_allow_html=True)

    # --- Kapasitas ---
    kva_idx   = CAPS.index(st.session_state.kapasitas) if st.session_state.kapasitas in CAPS else 1
    kapasitas = st.selectbox("Kapasitas (kVA)", CAPS, index=kva_idx, key="kva_select")

    if kapasitas != st.session_state.kapasitas:
        st.session_state.kapasitas = kapasitas
        # Reset tipe jika kapasitas baru tidak support tipe lama
        valid_tipe = TIPE_PER_KVA[kapasitas]
        if st.session_state.tipe_inti not in valid_tipe:
            st.session_state.tipe_inti = 'CRGO'
        st.session_state.nl_loss = LOSS_REF[kapasitas][st.session_state.tipe_inti]

    # --- Tipe Inti — dibatasi sesuai kapasitas ---
    valid_tipe = TIPE_PER_KVA[kapasitas]
    ti_idx     = valid_tipe.index(st.session_state.tipe_inti) if st.session_state.tipe_inti in valid_tipe else 0
    tipe_inti  = st.selectbox("Tipe Inti", valid_tipe, index=ti_idx, key="ti_select")

    # Info jika kapasitas hanya CRGO
    if len(valid_tipe) == 1:
        st.markdown(
            f'<div style="font-size:10px;color:{WARN2};margin:-8px 0 8px;'
            f'padding:4px 8px;background:{"#3d2a00" if D else "#fffbeb"};border-radius:6px;">'
            f'⚠️ {kapasitas} kVA hanya tersedia CRGO (kondisi nyata industri)</div>',
            unsafe_allow_html=True
        )

    if tipe_inti != st.session_state.tipe_inti:
        st.session_state.tipe_inti = tipe_inti
        st.session_state.nl_loss   = LOSS_REF[kapasitas][tipe_inti]

    ref_loss  = LOSS_REF[kapasitas][tipe_inti]
    nlc_range = NLC_REF[kapasitas][tipe_inti]
    tt_val    = TYPE_TEST[kapasitas][tipe_inti]

    st.markdown(
        f'<div style="font-size:11px;color:{TEXT3};margin:4px 0;">'
        f'Referensi NL Loss: <b style="color:{ACCENT2};">~{ref_loss} W</b></div>',
        unsafe_allow_html=True
    )

    # --- NL Loss ---
    nl_loss = st.number_input(
        "No Load Loss (W)", min_value=10, max_value=800,
        value=st.session_state.nl_loss, step=1, key="loss_input"
    )
    st.session_state.nl_loss = nl_loss

    st.markdown(
        f'<div style="font-size:11px;color:{TEXT3};margin:4px 0 4px;">'
        f'Type Test Standar: <b style="color:{ACCENT2};">{tt_val:.4f} %</b>'
        f'&nbsp;|&nbsp;Toleransi +30%: <b style="color:{WARN2};">{tt_val*1.30:.4f} %</b></div>',
        unsafe_allow_html=True
    )
    st.markdown(f'<div style="height:1px;background:{BORDER};margin:12px 0 16px;"></div>', unsafe_allow_html=True)

    prediksi_btn = st.button("⚡ Prediksi NLC%", use_container_width=True, key="pred_btn")

    if st.button("🔄 Reset Input", use_container_width=True, key="reset_btn"):
        st.session_state.kapasitas  = 100
        st.session_state.tipe_inti  = 'CRGO'
        st.session_state.nl_loss    = 129
        st.session_state.hasil_pred = None
        st.rerun()

    st.markdown(
        f'<div style="font-size:10px;color:{TEXT3};margin-top:14px;text-align:center;">'
        f'PT Bambang Djaja · Random Forest<br>431 Unit Asli · 50/100/160/250/400/630 kVA</div>',
        unsafe_allow_html=True
    )

# Simpan hasil prediksi
if prediksi_btn:
    hasil = predict_single(kapasitas, tipe_inti, nl_loss)
    st.session_state.hasil_pred = {
        'nilai': hasil, 'kapasitas': kapasitas, 'tipe_inti': tipe_inti,
        'nl_loss': nl_loss, 'ref_loss': ref_loss, 'nlc_range': nlc_range,
    }

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.markdown(f"""
<div style="border-left:3px solid {ACCENT};padding:10px 16px;margin-bottom:1rem;
    background:{BG2};border-radius:0 10px 10px 0;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div>
    <div style="font-size:16px;font-weight:700;color:{TEXT};">⚡ Prediksi No Load Current Trafo Distribusi 3 Fasa</div>
    <div style="font-size:11px;color:{TEXT2};margin-top:3px;">PT Bambang Djaja · Random Forest · 431 Unit Asli · 50 / 100 / 160 / 250 / 400 / 630 kVA</div>
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
    <span style="font-size:11px;padding:4px 14px;border-radius:20px;font-weight:600;
        background:{'#0d2a4a' if D else '#e8f0fb'};color:{ACCENT2};border:1.5px solid {ACCENT};">CRGO</span>
    <span style="font-size:11px;padding:4px 14px;border-radius:20px;font-weight:600;
        background:{'#0d3d2e' if D else '#e6f7ef'};color:{GREEN2};border:1.5px solid {GREEN};">Amorphous</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# METRIK ATAS
# ------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
for col, val, label, sub, color in [
    (c1, "431",          "Total Unit",      "6 kapasitas · 2 tipe inti",  TEXT),
    (c2, f"{r2:.4f}",    "R² Score",        "mendekati 1 = bagus",        ACCENT2),
    (c3, f"{rmse:.6f}",  "RMSE (%)",        "error rata-rata",            "#a78bfa"),
    (c4, f"{mae:.6f}",   "MAE (%)",         "selisih rata-rata",          GREEN2),
    (c5, f"{acc:.1f}%",  "Akurasi ±0.01%", "unit dalam toleransi",       "#fbbf24"),
]:
    with col:
        st.markdown(f"""
        <div style="background:{BG2};border-radius:10px;padding:14px 12px;
            text-align:center;border:0.5px solid {BORDER};margin-bottom:8px;">
            <div style="font-size:11px;color:{TEXT2};">{label}</div>
            <div style="font-size:22px;font-weight:700;color:{color};margin:4px 0;">{val}</div>
            <div style="font-size:10px;color:{TEXT3};">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div style="margin-top:0.5rem;"></div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# LEGENDA STATUS — pill compact di atas tab (style screenshot)
# ------------------------------------------------------------
st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
  <span style="background:{'#0d3d2e' if D else '#e6f7ef'};color:{GREEN2};
      border:1.5px solid {GREEN};border-radius:20px;padding:5px 14px;
      font-size:12px;font-weight:600;white-space:nowrap;">
    &#9989; NLC &#8804; Type Test &#8594; <b>LULUS</b>
  </span>
  <span style="background:{'#3d2a00' if D else '#fffbeb'};color:{WARN2};
      border:1.5px solid {WARN};border-radius:20px;padding:5px 14px;
      font-size:12px;font-weight:600;white-space:nowrap;">
    &#9888;&#65039; Type Test &lt; NLC &#8804; +30% &#8594; <b>TOLERANSI</b>
  </span>
  <span style="background:{'#3d0d0d' if D else '#fdecea'};color:{RED2};
      border:1.5px solid {RED};border-radius:20px;padding:5px 14px;
      font-size:12px;font-weight:600;white-space:nowrap;">
    &#10060; NLC &gt; +30% &#8594; <b>TIDAK MEMENUHI</b>
  </span>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🎯  Prediksi", "📦  Batch Prediksi", "📊  Evaluasi Model", "📋  Data"])

# ====================== TAB 1: PREDIKSI TUNGGAL ======================
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        sec("Input yang Dimasukkan")
        ia, ib, ic = st.columns(3)
        with ia: st.metric("Kapasitas", f"{kapasitas} kVA")
        with ib: st.metric("Tipe Inti",  tipe_inti)
        with ic: st.metric("NL Loss",    f"{nl_loss} W")

        if st.session_state.hasil_pred is not None:
            p     = st.session_state.hasil_pred
            hasil = p['nilai']
            kva_p = p['kapasitas']
            tip_p = p['tipe_inti']
            lbl, clr, bg_clr, bdr, tt, tol = get_type_test_status(hasil, kva_p, tip_p)
            dev = (p['nl_loss'] - p['ref_loss']) / p['ref_loss'] * 100

            # Nilai prediksi utama
            st.markdown(f"""
            <div style="background:{'#0d2a4a' if D else '#e8f0fb'};border:2px solid {ACCENT};
                border-radius:14px;padding:28px;text-align:center;margin:14px 0;">
                <div style="font-size:12px;color:{TEXT2};">Prediksi No Load Current</div>
                <div style="font-size:56px;font-weight:800;color:{ACCENT2};line-height:1.1;margin:8px 0;">{hasil:.6f}</div>
                <div style="font-size:16px;color:{TEXT2};">%</div>
                <div style="font-size:11px;color:{TEXT3};margin-top:8px;">
                    Type Test Standar: {tt:.4f} % &nbsp;|&nbsp; Toleransi +30%: {tol:.4f} %
                </div>
            </div>""", unsafe_allow_html=True)

            # Badge status type test
            st.markdown(
                f'<div style="background:{bg_clr};color:{clr};border:1px solid {bdr};'
                f'border-radius:10px;padding:14px;text-align:center;font-weight:700;'
                f'font-size:15px;margin-bottom:12px;">{lbl}</div>',
                unsafe_allow_html=True
            )

            # Tabel perbandingan type test
            if hasil <= tt:
                comp_icon = "✅"; comp_color = GREEN2
                comp_text = f"NLC {hasil:.6f}% ≤ Type Test {tt:.4f}% → Lulus"
            elif hasil <= tol:
                comp_icon = "⚠️"; comp_color = WARN2
                comp_text = f"Type Test {tt:.4f}% < NLC {hasil:.6f}% ≤ +30% ({tol:.4f}%) → Toleransi"
            else:
                comp_icon = "❌"; comp_color = RED2
                comp_text = f"NLC {hasil:.6f}% > +30% ({tol:.4f}%) → Tidak Memenuhi"

            st.markdown(f"""
            <div style="background:{BG2};border:0.5px solid {BORDER};border-radius:10px;padding:14px;font-size:12px;color:{TEXT2};">
                <b style="color:{TEXT};">📋 Analisis Type Test:</b><br><br>
                <span style="color:{comp_color};font-weight:600;">{comp_icon} {comp_text}</span><br><br>
                📌 NL Loss <b style="color:{TEXT};">{p['nl_loss']} W</b> →
                <b style="color:{'#4ade80' if dev<=0 else '#f87171'};">{'+' if dev>=0 else ''}{dev:.1f}%</b>
                dari referensi ~{p['ref_loss']} W<br><br>
                🔩 Tipe inti <b style="color:{TEXT};">{tip_p}</b> →
                {'NLC lebih tinggi (standar CRGO)' if tip_p=='CRGO' else 'NLC lebih rendah (efisiensi tinggi)'}
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div style="text-align:center;padding:80px 0;color:{TEXT3};">
                <div style="font-size:52px;">⚡</div>
                <div style="font-size:13px;margin-top:10px;">
                    Isi input di sidebar lalu klik<br>
                    <b style="color:{ACCENT2};">Prediksi NLC%</b>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        sec("Rata-rata NLC% per Kapasitas")
        set_plot()
        crgo = [df[(df['kapasitas']==c) & (df['tipe_inti']=='CRGO')]['nlc_persen'].mean() for c in CAPS]
        amor = [df[(df['kapasitas']==c) & (df['tipe_inti']=='Amorphous')]['nlc_persen'].mean()
                if len(df[(df['kapasitas']==c) & (df['tipe_inti']=='Amorphous')]) > 0 else np.nan for c in CAPS]
        tt_crgo = [TYPE_TEST[c]['CRGO'] for c in CAPS]
        tt_amor = [TYPE_TEST[c].get('Amorphous', np.nan) for c in CAPS]

        fig, ax = plt.subplots(figsize=(6, 4.5))
        x, w = np.arange(len(CAPS)), 0.28
        b1 = ax.bar(x-w, crgo, w, label='CRGO',       color='#185FA5', alpha=0.85, edgecolor='none', zorder=3)
        b2 = ax.bar(x,   [v if not np.isnan(v) else 0 for v in amor], w,
                    label='Amorphous', color='#0F6E56', alpha=0.85, edgecolor='none', zorder=3)

        # Type Test line markers
        for i, (tc, ta) in enumerate(zip(tt_crgo, tt_amor)):
            ax.plot([i-w-w/2, i-w/2], [tc, tc], color='#f87171', linewidth=1.5, zorder=4)
            if not np.isnan(ta):
                ax.plot([i-w/2, i+w/2], [ta, ta], color='#f87171', linewidth=1.5, zorder=4)

        # Label nilai
        for b in list(b1):
            if b.get_height() > 0.001:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                        f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=7.5)
        for b in list(b2):
            if b.get_height() > 0.001:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                        f'{b.get_height():.4f}', ha='center', va='bottom', fontsize=7.5)

        ax.plot([], [], color='#f87171', linewidth=1.5, label='Type Test Std')
        ax.set_xticks(x - w/2); ax.set_xticklabels([f'{c} kVA' for c in CAPS], fontsize=9)
        ax.set_ylabel('NLC (%)'); ax.set_title('Rata-rata NLC% vs Type Test Standar')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y', zorder=0)
        ax.set_ylim(0, max([v for v in crgo if not np.isnan(v)]) * 1.35)
        ax.text(0.99, 0.02, '*250/400/630 kVA: CRGO only',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='gray', style='italic')
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Tabel type test reference
        sec("Referensi Type Test Standar", "📏")
        tt_rows = []
        for c in CAPS:
            for t in TIPE_PER_KVA[c]:
                tv = TYPE_TEST[c][t]
                tt_rows.append({
                    'kVA': c, 'Tipe': t,
                    'Type Test (%)': tv,
                    'Toleransi +30% (%)': round(tv * 1.30, 4)
                })
        df_tt = pd.DataFrame(tt_rows)
        st.dataframe(df_tt, use_container_width=True, height=280, hide_index=True)

# ====================== TAB 2: BATCH PREDIKSI ======================
with tab2:
    sec("Batch Prediksi — Input Manual", "📦")

    st.markdown(
        f'<div style="font-size:12px;color:{TEXT2};margin-bottom:14px;">'
        f'Masukkan beberapa unit sekaligus, lalu unduh hasilnya sebagai CSV.</div>',
        unsafe_allow_html=True
    )

    upload_tab, manual_tab = st.tabs(["📁  Upload CSV", "✏️  Input Manual"])

    # ---- Fungsi batch prediksi ----
    def run_batch(df_valid):
        results = []
        for _, row in df_valid.iterrows():
            kva  = int(row['kapasitas'])
            tipe = row['tipe_inti']
            loss = float(row['nl_loss'])
            pred = predict_single(kva, tipe, loss)
            ref_l = LOSS_REF[kva][tipe]
            tt    = TYPE_TEST[kva][tipe]
            tol   = tt * 1.30
            dev   = (loss - ref_l) / ref_l * 100
            if pred <= tt:
                status = "✅ LULUS TYPE TEST"
            elif pred <= tol:
                status = "⚠️ DALAM TOLERANSI ±30%"
            else:
                status = "❌ TIDAK MEMENUHI STANDAR"
            results.append({
                'kapasitas': kva, 'tipe_inti': tipe, 'nl_loss': loss,
                'nlc_prediksi_%': round(pred, 6),
                'type_test_%': tt,
                'toleransi_+30%': round(tol, 4),
                'status': status,
                'deviasi_loss_%': round(dev, 2)
            })
        return pd.DataFrame(results)

    def show_batch_summary(df_res):
        n_lulus = (df_res['status'] == '✅ LULUS TYPE TEST').sum()
        n_tol   = (df_res['status'] == '⚠️ DALAM TOLERANSI ±30%').sum()
        n_fail  = (df_res['status'] == '❌ TIDAK MEMENUHI STANDAR').sum()
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total Unit",                len(df_res))
        with s2: st.metric("✅ Lulus Type Test",        n_lulus)
        with s3: st.metric("⚠️ Dalam Toleransi",        n_tol)
        with s4: st.metric("❌ Tidak Memenuhi",          n_fail)

    with upload_tab:
        st.markdown(
            f'<div style="font-size:12px;color:{TEXT3};margin-bottom:10px;">'
            f'Format CSV: kolom <b>kapasitas</b> (50/100/160/250/400/630), '
            f'<b>tipe_inti</b> (CRGO/Amorphous), <b>nl_loss</b></div>',
            unsafe_allow_html=True
        )
        uploaded = st.file_uploader("Upload file CSV", type=['csv'], key="batch_upload")

        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded)
                required_cols = {'kapasitas', 'tipe_inti', 'nl_loss'}
                if not required_cols.issubset(df_up.columns):
                    st.error(f"❌ Kolom wajib: {required_cols}. Ditemukan: {set(df_up.columns)}")
                else:
                    invalid_kva  = df_up[~df_up['kapasitas'].isin(CAPS)]
                    if len(invalid_kva) > 0:
                        st.warning(f"⚠️ {len(invalid_kva)} baris kapasitas tidak valid (harus 50/100/160/250/400/630)")

                    # Validasi kombinasi kapasitas + tipe inti
                    def valid_combo(row):
                        return row['tipe_inti'] in TIPE_PER_KVA.get(row['kapasitas'], [])
                    df_up['_valid'] = df_up.apply(valid_combo, axis=1)
                    invalid_combo  = df_up[~df_up['_valid']]
                    if len(invalid_combo) > 0:
                        st.warning(f"⚠️ {len(invalid_combo)} baris kombinasi kapasitas+tipe tidak valid "
                                   f"(250/400/630 kVA hanya CRGO)")

                    df_valid = df_up[df_up['_valid']].drop(columns=['_valid']).copy()

                    if len(df_valid) > 0:
                        st.success(f"✅ {len(df_valid)} baris valid siap diprediksi")
                        st.dataframe(df_valid.head(10), use_container_width=True)

                        if st.button("⚡ Prediksi Semua (dari CSV)", use_container_width=True, key="pred_csv"):
                            df_res = run_batch(df_valid)
                            st.session_state.batch_results = df_res
                            show_batch_summary(df_res)
                            st.dataframe(df_res, use_container_width=True, height=300)
                            csv_out = df_res.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Hasil Prediksi CSV", data=csv_out,
                                file_name="hasil_prediksi_nlc.csv", mime="text/csv",
                                use_container_width=True, key="dl_csv_result"
                            )
            except Exception as e:
                st.error(f"❌ Error membaca CSV: {e}")

        # Template download — semua 6 kapasitas
        template_df = pd.DataFrame({
            'kapasitas': [50,  50,  100, 100, 160, 160, 250, 400, 630],
            'tipe_inti': ['CRGO','Amorphous','CRGO','Amorphous','CRGO','Amorphous','CRGO','CRGO','CRGO'],
            'nl_loss':   [84,   36,  129, 40,  183, 77,  274, 389, 555],
        })
        st.download_button(
            "📄 Download Template CSV",
            data=template_df.to_csv(index=False).encode('utf-8'),
            file_name="template_batch_nlc.csv", mime="text/csv",
            use_container_width=True, key="dl_template"
        )

    with manual_tab:
        st.markdown(
            f'<div style="font-size:12px;color:{TEXT3};margin-bottom:10px;">'
            f'Tambah unit satu per satu, lalu klik Prediksi Semua.</div>',
            unsafe_allow_html=True
        )

        ca, cb, cc, cd = st.columns([2, 2, 2, 1])
        with ca: new_kva  = st.selectbox("Kapasitas", CAPS, key="new_kva")
        with cb:
            valid_new = TIPE_PER_KVA[new_kva]
            new_tipe  = st.selectbox("Tipe Inti", valid_new, key="new_tipe")
        with cc: new_loss = st.number_input("NL Loss (W)", 10, 800,
                                             LOSS_REF[new_kva][new_tipe], key="new_loss")
        with cd:
            st.markdown('<div style="margin-top:24px;"></div>', unsafe_allow_html=True)
            if st.button("➕ Tambah", key="add_row"):
                st.session_state.batch_rows.append(
                    {'kapasitas': new_kva, 'tipe_inti': new_tipe, 'nl_loss': new_loss}
                )
                st.rerun()

        if st.session_state.batch_rows:
            df_manual = pd.DataFrame(st.session_state.batch_rows)
            st.dataframe(df_manual, use_container_width=True, height=180)

            col_pred, col_clear = st.columns(2)
            with col_pred:
                if st.button("⚡ Prediksi Semua Unit", use_container_width=True, key="pred_manual"):
                    df_res = run_batch(df_manual)
                    st.session_state.batch_results = df_res

            with col_clear:
                if st.button("🗑️ Hapus Semua", use_container_width=True, key="clear_rows"):
                    st.session_state.batch_rows   = []
                    st.session_state.batch_results = None
                    st.rerun()

        if st.session_state.batch_results is not None:
            df_res = st.session_state.batch_results
            st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)
            sec("Hasil Prediksi Batch", "📊")
            show_batch_summary(df_res)
            st.dataframe(df_res, use_container_width=True, height=250)
            csv_out = df_res.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil Prediksi CSV", data=csv_out,
                file_name="hasil_prediksi_nlc.csv", mime="text/csv",
                use_container_width=True, key="dl_manual_result"
            )

# ====================== TAB 3: EVALUASI MODEL ======================
with tab3:
    set_plot()
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("R² Score",       f"{r2:.4f}")
    with m2: st.metric("RMSE (%)",       f"{rmse:.6f}")
    with m3: st.metric("MAE (%)",        f"{mae:.6f}")
    with m4: st.metric("Akurasi ±0.01%", f"{acc:.1f}%")
    st.markdown('<div style="margin-top:1rem;"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        sec("Aktual vs Prediksi")
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for tipe, color in [('CRGO', '#185FA5'), ('Amorphous', '#0F6E56')]:
            idx = [i for i, x in enumerate(X_test) if le.classes_[int(round(x[1]))] == tipe]
            ax.scatter([y_test[i] for i in idx], [y_pred[i] for i in idx],
                       c=color, label=tipe, alpha=0.7, s=45, edgecolors='none', zorder=3)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, alpha=0.6, label='Ideal')
        ax.set_xlabel('Aktual NLC (%)'); ax.set_ylabel('Prediksi NLC (%)')
        ax.set_title(f'Aktual vs Prediksi  (R² = {r2:.4f})')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("Feature Importance")
        imps   = rf.feature_importances_
        fnames = ['Kapasitas (kVA)', 'Tipe Inti', 'NL Loss (W)']
        fclrs  = ['#185FA5', '#0F6E56', '#BA7517']
        si     = np.argsort(imps)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        bars = ax.barh(
            [fnames[i] for i in si], [imps[i] for i in si],
            color=[fclrs[i] for i in si], edgecolor='none', height=0.5, zorder=3
        )
        for bar, val in zip(bars, [imps[i] for i in si]):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f'{val*100:.1f}%', va='center', fontsize=11, fontweight='700')
        ax.set_xlabel('Importance'); ax.set_xlim(0, max(imps) * 1.4)
        ax.set_title('Feature Importance')
        ax.grid(True, alpha=0.3, axis='x', zorder=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    sec("Distribusi Error Prediksi")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    errors = y_pred - y_test
    ax.hist(errors, bins=25, color='#185FA5', alpha=0.8, edgecolor='none', zorder=3)
    ax.axvline(0, color='#f87171', linestyle='--', linewidth=2, label='Error=0', zorder=4)
    ax.axvline(errors.mean(), color='#fbbf24', linestyle='-', linewidth=2,
               label=f'Mean={errors.mean():.6f}', zorder=4)
    ax.set_xlabel('Error Prediksi (%)'); ax.set_ylabel('Jumlah unit')
    ax.set_title('Distribusi Error — Mayoritas mendekati nol = model akurat')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ====================== TAB 4: DATA ======================
with tab4:
    set_plot()
    sec("Dataset Asli dari Testing Pabrik", "📋")

    f1, f2, f3 = st.columns(3)
    with f1: fkva  = st.multiselect("Filter Kapasitas", CAPS, default=CAPS, key="flt_kva")
    with f2: fcore = st.multiselect("Filter Tipe Inti", ['CRGO', 'Amorphous'],
                                    default=['CRGO', 'Amorphous'], key="flt_core")
    with f3: st.metric("Total data", len(df))

    df_f = df[(df['kapasitas'].isin(fkva)) & (df['tipe_inti'].isin(fcore))]
    st.dataframe(
        df_f.style.format({'nlc_persen': '{:.6f}', 'nl_loss': '{:.4f}'}),
        use_container_width=True, height=300
    )
    st.markdown(
        f'<div style="font-size:12px;color:{TEXT3};margin-top:4px;">Menampilkan '
        f'<b style="color:{ACCENT2};">{len(df_f)}</b> dari <b>{len(df)}</b> data</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download Data Terfilter CSV",
        data=df_f.to_csv(index=False).encode('utf-8'),
        file_name="data_terfilter.csv", mime="text/csv", key="dl_filtered"
    )

    st.markdown('<div style="margin-top:1.2rem;"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")

    with col1:
        sec("Statistik per Kapasitas & Tipe Inti (sesuai filter)")
        if len(df_f) > 0:
            stats = df_f.groupby(['kapasitas', 'tipe_inti'])['nlc_persen'].agg(
                ['mean', 'min', 'max', 'std']
            ).round(6)
            stats.columns = ['Rata-rata', 'Min', 'Max', 'Std Dev']
            st.dataframe(stats, use_container_width=True)
        else:
            st.info("Tidak ada data untuk filter yang dipilih.")

    with col2:
        sec("NL Loss (W) vs NLC (%)")
        fig, ax = plt.subplots(figsize=(5, 3.8))
        for tipe, color, mk in [('CRGO', '#185FA5', 'o'), ('Amorphous', '#0F6E56', '^')]:
            d = df_f[df_f['tipe_inti'] == tipe]
            if len(d) > 0:
                ax.scatter(d['nl_loss'], d['nlc_persen'],
                           c=color, alpha=0.65, s=35, label=tipe, marker=mk, edgecolors='none', zorder=3)
        ax.set_xlabel('NL Loss (W)'); ax.set_ylabel('NLC (%)')
        ax.set_title('Hubungan NL Loss vs NLC%')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, zorder=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# Footer
st.markdown(
    f'<div style="text-align:center;color:{TEXT3};font-size:11px;margin-top:2rem;'
    f'padding:12px;border-top:1px solid {BORDER};">'
    f'Prediksi No Load Current Trafo · PT Bambang Djaja · Random Forest · 431 Unit Data Asli · v3.0</div>',
    unsafe_allow_html=True
)
