# ============================================================
#  PREDIKSI NO LOAD CURRENT TRAFO DISTRIBUSI
#  PT Bambang Djaja — Random Forest
#  v4.0 — 3 Fasa + 1 Fasa dalam satu aplikasi
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

# ── TEMA ────────────────────────────────────────────────────
BG      = '#f5f7fa'
BG2     = '#ffffff'
BG3     = '#eef2f7'
BORDER  = '#dde3ed'
TEXT    = '#1a1a2e'
TEXT2   = '#555555'
TEXT3   = '#999999'
ACCENT  = '#185FA5'
ACCENT2 = '#185FA5'
GREEN   = '#0F6E56'
GREEN2  = '#0F6E56'
WARN    = '#d97706'
WARN2   = '#d97706'
RED     = '#991b1b'
RED2    = '#dc2626'
PLOTBG  = '#ffffff'
PLOTFACE= '#f5f7fa'
GRID    = '#e0e0e0'
LEGBG   = '#ffffff'

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
/* Mode toggle pills */
.mode-pill {{
    display:inline-block; padding:6px 18px; border-radius:20px;
    font-size:12px; font-weight:600; cursor:pointer; border:1.5px solid;
}}
/* Sembunyikan collapse sidebar */
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"],
button[aria-label="Close sidebar"],
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] > div > div > div > button {{
    display:none !important; visibility:hidden !important;
    pointer-events:none !important; width:0 !important; height:0 !important; opacity:0 !important;
}}
section[data-testid="stSidebar"] {{
    transform:none !important; min-width:260px !important; width:260px !important;
}}
</style>
""", unsafe_allow_html=True)

# ── KONSTANTA 3 FASA ─────────────────────────────────────────
CAPS_3F = [50, 100, 160, 250, 400, 630]
TIPE_PER_KVA_3F = {
    50:  ['CRGO','Amorphous'], 100: ['CRGO','Amorphous'],
    160: ['CRGO','Amorphous'], 250: ['CRGO'],
    400: ['CRGO'],             630: ['CRGO'],
}
LOSS_REF_3F = {
    50:  {'CRGO':84,  'Amorphous':36},
    100: {'CRGO':129, 'Amorphous':40},
    160: {'CRGO':183, 'Amorphous':77},
    250: {'CRGO':274}, 400: {'CRGO':389}, 630: {'CRGO':555},
}
NLC_REF_3F = {
    50:  {'CRGO':(0.294,0.472),'Amorphous':(0.134,0.200)},
    100: {'CRGO':(0.160,0.419),'Amorphous':(0.197,0.258)},
    160: {'CRGO':(0.187,0.241),'Amorphous':(0.089,0.126)},
    250: {'CRGO':(0.254,0.344)}, 400: {'CRGO':(0.131,0.172)}, 630: {'CRGO':(0.110,0.136)},
}
TYPE_TEST_3F = {
    50:  {'CRGO':0.40,'Amorphous':0.16},
    100: {'CRGO':0.31,'Amorphous':0.24},
    160: {'CRGO':0.25,'Amorphous':0.10},
    250: {'CRGO':0.45}, 400: {'CRGO':0.18}, 630: {'CRGO':0.14},
}

# ── KONSTANTA 1 FASA ─────────────────────────────────────────
TIPE_1F = ['CRGO', 'Amorphous']
LOSS_REF_1F = {'CRGO': 113, 'Amorphous': 122}
TYPE_TEST_1F = {'CRGO': 1.23, 'Amorphous': 0.40}

MODEL_3F_PATH = 'model/model_trafo.pkl'
LE_3F_PATH    = 'model/label_encoder.pkl'
DATA_3F_PATH  = 'data_asli.csv'

MODEL_1F_PATH = 'model_1fasa/model_trafo_1fasa.pkl'
LE_1F_PATH    = 'model_1fasa/label_encoder_1fasa.pkl'
DATA_1F_PATH  = 'data_1fasa.csv'

# ── LOAD MODEL & DATA ────────────────────────────────────────
@st.cache_resource
def load_model_3f():
    os.makedirs('model', exist_ok=True)
    if os.path.exists(MODEL_3F_PATH) and os.path.exists(LE_3F_PATH):
        return joblib.load(MODEL_3F_PATH), joblib.load(LE_3F_PATH)
    df = pd.read_csv(DATA_3F_PATH)
    le = LabelEncoder(); df['enc'] = le.fit_transform(df['tipe_inti'])
    X = df[['kapasitas','enc','nl_loss']].values; y = df['nlc_persen'].values
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_3F_PATH); joblib.dump(le, LE_3F_PATH)
    return rf, le

@st.cache_resource
def load_model_1f():
    os.makedirs('model_1fasa', exist_ok=True)
    if os.path.exists(MODEL_1F_PATH) and os.path.exists(LE_1F_PATH):
        return joblib.load(MODEL_1F_PATH), joblib.load(LE_1F_PATH)
    df = pd.read_csv(DATA_1F_PATH)
    le = LabelEncoder(); df['enc'] = le.fit_transform(df['tipe_inti'])
    X = df[['nl_loss','enc']].values; y = df['nlc_persen'].values
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_1F_PATH); joblib.dump(le, LE_1F_PATH)
    return rf, le

@st.cache_data
def load_data_3f():
    return pd.read_csv(DATA_3F_PATH)

@st.cache_data
def load_data_1f():
    return pd.read_csv(DATA_1F_PATH)

@st.cache_data
def get_metrics_3f():
    df = load_data_3f()
    le_tmp = LabelEncoder(); df['enc'] = le_tmp.fit_transform(df['tipe_inti'])
    X = df[['kapasitas','enc','nl_loss']].values; y = df['nlc_persen'].values
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf, _ = load_model_3f(); yp = rf.predict(Xte)
    return (r2_score(yte,yp), np.sqrt(mean_squared_error(yte,yp)),
            mean_absolute_error(yte,yp), np.mean(np.abs(yte-yp)<=0.01)*100,
            Xte, yte, yp)

@st.cache_data
def get_metrics_1f():
    df = load_data_1f()
    le_tmp = LabelEncoder(); df['enc'] = le_tmp.fit_transform(df['tipe_inti'])
    X = df[['nl_loss','enc']].values; y = df['nlc_persen'].values
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf, _ = load_model_1f(); yp = rf.predict(Xte)
    return (r2_score(yte,yp), np.sqrt(mean_squared_error(yte,yp)),
            mean_absolute_error(yte,yp), np.mean(np.abs(yte-yp)<=0.01)*100,
            Xte, yte, yp)

# ── HELPERS ──────────────────────────────────────────────────
def set_plot():
    matplotlib.rcParams.update({
        'figure.facecolor':PLOTFACE,'axes.facecolor':PLOTBG,'axes.edgecolor':BORDER,
        'text.color':TEXT,'axes.labelcolor':TEXT,'xtick.color':TEXT2,'ytick.color':TEXT2,
        'grid.color':GRID,'legend.facecolor':LEGBG,'legend.edgecolor':BORDER,'legend.labelcolor':TEXT,
    })

def sec(title, icon=""):
    st.markdown(
        f'<div style="font-size:13px;font-weight:600;color:{TEXT};'
        f'border-bottom:2px solid {ACCENT};padding-bottom:6px;margin-bottom:14px;">'
        f'{icon} {title}</div>', unsafe_allow_html=True)

def status_type_test(pred, tt):
    tol = tt * 1.30
    if pred <= tt:
        return ("✅  LULUS TYPE TEST", GREEN2, '#e6f7ef', GREEN, tt, tol)
    elif pred <= tol:
        return ("⚠️  DALAM TOLERANSI ±30%", WARN2, '#fffbeb', WARN, tt, tol)
    else:
        return ("❌  TIDAK MEMENUHI STANDAR", RED2, '#fdecea', RED, tt, tol)

def render_pred_result(pred, tt, nl_loss, ref_loss, tipe, label_extra=""):
    lbl, clr, bg_clr, bdr, tt_v, tol_v = status_type_test(pred, tt)
    dev = (nl_loss - ref_loss) / ref_loss * 100
    st.markdown(f"""
    <div style="background:#e8f0fb;border:2px solid {ACCENT};
        border-radius:14px;padding:28px;text-align:center;margin:14px 0;">
        <div style="font-size:12px;color:{TEXT2};">Prediksi No Load Current {label_extra}</div>
        <div style="font-size:52px;font-weight:800;color:{ACCENT2};line-height:1.1;margin:8px 0;">{pred:.6f}</div>
        <div style="font-size:16px;color:{TEXT2};">%</div>
        <div style="font-size:11px;color:{TEXT3};margin-top:8px;">
            Type Test Standar: {tt_v:.4f} % &nbsp;|&nbsp; Toleransi +30%: {tol_v:.4f} %
        </div>
    </div>""", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{bg_clr};color:{clr};border:1px solid {bdr};'
        f'border-radius:10px;padding:14px;text-align:center;font-weight:700;'
        f'font-size:15px;margin-bottom:12px;">{lbl}</div>', unsafe_allow_html=True)
    if pred <= tt_v:
        comp_icon,comp_color = "✅",GREEN2
        comp_text = f"NLC {pred:.6f}% ≤ Type Test {tt_v:.4f}% → Lulus"
    elif pred <= tol_v:
        comp_icon,comp_color = "⚠️",WARN2
        comp_text = f"Type Test {tt_v:.4f}% < NLC {pred:.6f}% ≤ +30% ({tol_v:.4f}%) → Toleransi"
    else:
        comp_icon,comp_color = "❌",RED2
        comp_text = f"NLC {pred:.6f}% > +30% ({tol_v:.4f}%) → Tidak Memenuhi"
    st.markdown(f"""
    <div style="background:{BG2};border:0.5px solid {BORDER};border-radius:10px;padding:14px;font-size:12px;color:{TEXT2};">
        <b style="color:{TEXT};">📋 Analisis Type Test:</b><br><br>
        <span style="color:{comp_color};font-weight:600;">{comp_icon} {comp_text}</span><br><br>
        📌 NL Loss <b style="color:{TEXT};">{nl_loss} W</b> →
        <b style="color:{'#4ade80' if dev<=0 else '#f87171'};">{'+' if dev>=0 else ''}{dev:.1f}%</b>
        dari referensi ~{ref_loss} W<br><br>
        🔩 Tipe inti <b style="color:{TEXT};">{tipe}</b> →
        {'NLC lebih rendah (efisiensi tinggi)' if tipe=='Amorphous' else 'NLC standar (CRGO)'}
    </div>""", unsafe_allow_html=True)

# ── SESSION STATE ─────────────────────────────────────────────
defaults = {
    'mode': '3 Fasa',
    # 3F
    'kapasitas': 100, 'tipe_inti': 'CRGO', 'nl_loss': 129, 'hasil_pred': None,
    'batch_rows': [{'kapasitas':100,'tipe_inti':'CRGO','nl_loss':129}], 'batch_results': None,
    # 1F
    'tipe_1f': 'CRGO', 'nl_loss_1f': 113, 'hasil_pred_1f': None,
    'batch_rows_1f': [{'tipe_inti':'CRGO','nl_loss':113}], 'batch_results_1f': None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── LOAD ─────────────────────────────────────────────────────
rf3, le3   = load_model_3f()
rf1, le1   = load_model_1f()
df3        = load_data_3f()
df1        = load_data_1f()
r2_3, rmse_3, mae_3, acc_3, Xte_3, yte_3, ype_3 = get_metrics_3f()
r2_1, rmse_1, mae_1, acc_1, Xte_1, yte_1, ype_1 = get_metrics_1f()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<div style="font-size:15px;font-weight:700;color:{ACCENT2};">⚡ NLC Predictor — PT Bambang Djaja</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="height:1px;background:{BORDER};margin:12px 0;"></div>', unsafe_allow_html=True)

    # Mode selector
    st.markdown(f'<div style="font-size:11px;font-weight:600;color:{TEXT2};margin-bottom:8px;">🔀 MODE TRAFO</div>',
                unsafe_allow_html=True)
    mode_sel = st.radio("", ['3 Fasa','1 Fasa'], index=0 if st.session_state.mode=='3 Fasa' else 1,
                        horizontal=True, key="mode_radio", label_visibility="collapsed")
    if mode_sel != st.session_state.mode:
        st.session_state.mode = mode_sel
        st.rerun()

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:12px 0;"></div>', unsafe_allow_html=True)

    # ── SIDEBAR 3 FASA ──
    if st.session_state.mode == '3 Fasa':
        st.markdown(f'<div style="font-size:12px;font-weight:600;color:{TEXT2};margin-bottom:12px;">🔧 INPUT TRAFO 3 FASA</div>',
                    unsafe_allow_html=True)

        kva_idx   = CAPS_3F.index(st.session_state.kapasitas) if st.session_state.kapasitas in CAPS_3F else 1
        kapasitas = st.selectbox("Kapasitas (kVA)", CAPS_3F, index=kva_idx, key="kva_select")
        if kapasitas != st.session_state.kapasitas:
            st.session_state.kapasitas = kapasitas
            valid = TIPE_PER_KVA_3F[kapasitas]
            if st.session_state.tipe_inti not in valid:
                st.session_state.tipe_inti = 'CRGO'
            st.session_state.nl_loss = LOSS_REF_3F[kapasitas][st.session_state.tipe_inti]

        valid_tipe = TIPE_PER_KVA_3F[kapasitas]
        ti_idx     = valid_tipe.index(st.session_state.tipe_inti) if st.session_state.tipe_inti in valid_tipe else 0
        tipe_inti  = st.selectbox("Tipe Inti", valid_tipe, index=ti_idx, key="ti_select")
        if len(valid_tipe)==1:
            st.markdown(f'<div style="font-size:10px;color:{WARN2};margin:-8px 0 8px;'
                        f'padding:4px 8px;background:#fffbeb;border-radius:6px;">'
                        f'⚠️ {kapasitas} kVA hanya tersedia CRGO (kondisi nyata industri)</div>',
                        unsafe_allow_html=True)
        if tipe_inti != st.session_state.tipe_inti:
            st.session_state.tipe_inti = tipe_inti
            st.session_state.nl_loss   = LOSS_REF_3F[kapasitas][tipe_inti]

        ref_loss = LOSS_REF_3F[kapasitas][tipe_inti]
        tt_val   = TYPE_TEST_3F[kapasitas][tipe_inti]
        st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin:4px 0;">'
                    f'Referensi NL Loss: <b style="color:{ACCENT2};">~{ref_loss} W</b></div>',
                    unsafe_allow_html=True)
        nl_loss = st.number_input("No Load Loss (W)", min_value=10, max_value=800,
                                  value=st.session_state.nl_loss, step=1, key="loss_input")
        st.session_state.nl_loss = nl_loss
        st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin:4px 0;">'
                    f'Type Test Standar: <b style="color:{ACCENT2};">{tt_val:.4f} %</b>'
                    f'&nbsp;|&nbsp;Toleransi +30%: <b style="color:{WARN2};">{tt_val*1.30:.4f} %</b></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="height:1px;background:{BORDER};margin:12px 0 16px;"></div>', unsafe_allow_html=True)

        prediksi_btn = st.button("⚡ Prediksi NLC%", use_container_width=True, key="pred_btn_3f")
        if st.button("🔄 Reset Input", use_container_width=True, key="reset_btn_3f"):
            st.session_state.kapasitas=100; st.session_state.tipe_inti='CRGO'
            st.session_state.nl_loss=129; st.session_state.hasil_pred=None; st.rerun()

        st.markdown(f'<div style="font-size:10px;color:{TEXT3};margin-top:14px;text-align:center;">'
                    f'Random Forest · 431 Unit Asli<br>50/100/160/250/400/630 kVA · CRGO & Amorphous</div>',
                    unsafe_allow_html=True)

        if prediksi_btn:
            enc  = le3.transform([tipe_inti])[0]
            pred = rf3.predict([[kapasitas, enc, nl_loss]])[0]
            st.session_state.hasil_pred = {
                'nilai':pred, 'kapasitas':kapasitas, 'tipe_inti':tipe_inti,
                'nl_loss':nl_loss, 'ref_loss':ref_loss,
                'nlc_range':NLC_REF_3F[kapasitas][tipe_inti], 'tt':tt_val,
            }

    # ── SIDEBAR 1 FASA ──
    else:
        st.markdown(f'<div style="font-size:12px;font-weight:600;color:{TEXT2};margin-bottom:12px;">🔧 INPUT TRAFO 1 FASA</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin-bottom:8px;">50 kVA · CRGO & Amorphous</div>',
                    unsafe_allow_html=True)

        ti_idx_1f = TIPE_1F.index(st.session_state.tipe_1f) if st.session_state.tipe_1f in TIPE_1F else 0
        tipe_1f   = st.selectbox("Tipe Inti", TIPE_1F, index=ti_idx_1f, key="ti_1f")
        if tipe_1f != st.session_state.tipe_1f:
            st.session_state.tipe_1f   = tipe_1f
            st.session_state.nl_loss_1f = LOSS_REF_1F[tipe_1f]

        ref_loss_1f = LOSS_REF_1F[tipe_1f]
        tt_val_1f   = TYPE_TEST_1F[tipe_1f]
        st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin:4px 0;">'
                    f'Referensi NL Loss: <b style="color:{ACCENT2};">~{ref_loss_1f} W</b></div>',
                    unsafe_allow_html=True)

        nl_loss_1f = st.number_input("No Load Loss (W)", min_value=50, max_value=250,
                                     value=st.session_state.nl_loss_1f, step=1, key="loss_1f")
        st.session_state.nl_loss_1f = nl_loss_1f
        st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin:4px 0;">'
                    f'Type Test Standar: <b style="color:{ACCENT2};">{tt_val_1f:.4f} %</b>'
                    f'&nbsp;|&nbsp;Toleransi +30%: <b style="color:{WARN2};">{tt_val_1f*1.30:.4f} %</b></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div style="height:1px;background:{BORDER};margin:12px 0 16px;"></div>', unsafe_allow_html=True)

        prediksi_btn_1f = st.button("⚡ Prediksi NLC%", use_container_width=True, key="pred_btn_1f")
        if st.button("🔄 Reset Input", use_container_width=True, key="reset_btn_1f"):
            st.session_state.tipe_1f='CRGO'; st.session_state.nl_loss_1f=113
            st.session_state.hasil_pred_1f=None; st.rerun()

        st.markdown(f'<div style="font-size:10px;color:{TEXT3};margin-top:14px;text-align:center;">'
                    f'Random Forest · 96 Unit Asli<br>50 kVA · CRGO & Amorphous</div>',
                    unsafe_allow_html=True)

        if prediksi_btn_1f:
            enc_1f  = le1.transform([tipe_1f])[0]
            pred_1f = rf1.predict([[nl_loss_1f, enc_1f]])[0]
            st.session_state.hasil_pred_1f = {
                'nilai':pred_1f, 'tipe_inti':tipe_1f,
                'nl_loss':nl_loss_1f, 'ref_loss':ref_loss_1f, 'tt':tt_val_1f,
            }

# ── HEADER ───────────────────────────────────────────────────
mode_now = st.session_state.mode
badge_color = ACCENT if mode_now=='3 Fasa' else GREEN2
badge_bg    = '#e8f0fb' if mode_now=='3 Fasa' else '#e6f7ef'
desc        = '431 Unit Asli · 50/100/160/250/400/630 kVA' if mode_now=='3 Fasa' else '96 Unit Asli · 50 kVA · CRGO & Amorphous'

st.markdown(f"""
<div style="border-left:3px solid {ACCENT};padding:10px 16px;margin-bottom:1rem;
    background:{BG2};border-radius:0 10px 10px 0;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
  <div>
    <div style="font-size:16px;font-weight:700;color:{TEXT};">
      ⚡ Prediksi No Load Current Trafo Distribusi — {mode_now}</div>
    <div style="font-size:11px;color:{TEXT2};margin-top:3px;">PT Bambang Djaja · Random Forest · {desc}</div>
  </div>
  <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;">
    <span style="font-size:11px;padding:4px 14px;border-radius:20px;font-weight:600;
        background:#e8f0fb;color:{ACCENT2};border:1.5px solid {ACCENT};">CRGO</span>
    <span style="font-size:11px;padding:4px 14px;border-radius:20px;font-weight:600;
        background:#e6f7ef;color:{GREEN2};border:1.5px solid {GREEN};">Amorphous</span>
    <span style="font-size:11px;padding:4px 14px;border-radius:20px;font-weight:600;
        background:{badge_bg};color:{badge_color};border:1.5px solid {badge_color};">{mode_now}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── METRIK ATAS ──────────────────────────────────────────────
if mode_now == '3 Fasa':
    r2, rmse, mae, acc = r2_3, rmse_3, mae_3, acc_3
    n_total = "431"
else:
    r2, rmse, mae, acc = r2_1, rmse_1, mae_1, acc_1
    n_total = "96"

c1,c2,c3,c4,c5 = st.columns(5)
for col, val, label, sub, color in [
    (c1, n_total,       "Total Unit",      "data asli pabrik",       TEXT),
    (c2, f"{r2:.4f}",   "R² Score",        "mendekati 1 = bagus",    ACCENT2),
    (c3, f"{rmse:.6f}", "RMSE (%)",        "error rata-rata",        "#a78bfa"),
    (c4, f"{mae:.6f}",  "MAE (%)",         "selisih rata-rata",      GREEN2),
    (c5, f"{acc:.1f}%", "Akurasi ±0.01%", "unit dalam toleransi",   "#fbbf24"),
]:
    with col:
        st.markdown(f"""
        <div style="background:{BG2};border-radius:10px;padding:14px 12px;
            text-align:center;border:0.5px solid {BORDER};margin-bottom:8px;">
            <div style="font-size:11px;color:{TEXT2};">{label}</div>
            <div style="font-size:22px;font-weight:700;color:{color};margin:4px 0;">{val}</div>
            <div style="font-size:10px;color:{TEXT3};">{sub}</div>
        </div>""", unsafe_allow_html=True)

# Legenda status
st.markdown(f"""
<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px;">
  <span style="background:#e6f7ef;color:{GREEN2};border:1.5px solid {GREEN};border-radius:20px;
      padding:5px 14px;font-size:12px;font-weight:600;white-space:nowrap;">
    &#9989; NLC &#8804; Type Test &#8594; <b>LULUS</b></span>
  <span style="background:#fffbeb;color:{WARN2};border:1.5px solid {WARN};border-radius:20px;
      padding:5px 14px;font-size:12px;font-weight:600;white-space:nowrap;">
    &#9888;&#65039; Type Test &lt; NLC &#8804; +30% &#8594; <b>TOLERANSI</b></span>
  <span style="background:#fdecea;color:{RED2};border:1.5px solid {RED};border-radius:20px;
      padding:5px 14px;font-size:12px;font-weight:600;white-space:nowrap;">
    &#10060; NLC &gt; +30% &#8594; <b>TIDAK MEMENUHI</b></span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  MODE: 3 FASA
# ══════════════════════════════════════════════════════════════
if mode_now == '3 Fasa':
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediksi", "📦 Batch Prediksi", "📊 Evaluasi Model", "📋 Data"])

    # ── TAB 1: PREDIKSI TUNGGAL ──
    with tab1:
        col1, col2 = st.columns([1,1], gap="large")
        with col1:
            sec("Input yang Dimasukkan")
            ia,ib,ic = st.columns(3)
            with ia: st.metric("Kapasitas", f"{st.session_state.kapasitas} kVA")
            with ib: st.metric("Tipe Inti",  st.session_state.tipe_inti)
            with ic: st.metric("NL Loss",    f"{st.session_state.nl_loss} W")

            if st.session_state.hasil_pred:
                p = st.session_state.hasil_pred
                render_pred_result(p['nilai'], p['tt'], p['nl_loss'], p['ref_loss'], p['tipe_inti'])
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
            crgo = [df3[(df3['kapasitas']==c)&(df3['tipe_inti']=='CRGO')]['nlc_persen'].mean() for c in CAPS_3F]
            amor = [df3[(df3['kapasitas']==c)&(df3['tipe_inti']=='Amorphous')]['nlc_persen'].mean()
                    if len(df3[(df3['kapasitas']==c)&(df3['tipe_inti']=='Amorphous')])>0 else np.nan for c in CAPS_3F]
            tt_crgo = [TYPE_TEST_3F[c]['CRGO'] for c in CAPS_3F]
            tt_amor = [TYPE_TEST_3F[c].get('Amorphous',np.nan) for c in CAPS_3F]
            fig,ax = plt.subplots(figsize=(6,4.5))
            x,w = np.arange(len(CAPS_3F)), 0.28
            b1 = ax.bar(x-w, crgo, w, label='CRGO', color='#185FA5', alpha=0.85, edgecolor='none', zorder=3)
            b2 = ax.bar(x, [v if not np.isnan(v) else 0 for v in amor], w,
                        label='Amorphous', color='#0F6E56', alpha=0.85, edgecolor='none', zorder=3)
            for i,(tc,ta) in enumerate(zip(tt_crgo,tt_amor)):
                ax.plot([i-w-w/2,i-w/2],[tc,tc],color='#f87171',linewidth=1.5,zorder=4)
                if not np.isnan(ta):
                    ax.plot([i-w/2,i+w/2],[ta,ta],color='#f87171',linewidth=1.5,zorder=4)
            for b in b1:
                if b.get_height()>0.001:
                    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.004,f'{b.get_height():.4f}',ha='center',va='bottom',fontsize=7.5)
            for b in b2:
                if b.get_height()>0.001:
                    ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.004,f'{b.get_height():.4f}',ha='center',va='bottom',fontsize=7.5)
            ax.plot([],[],color='#f87171',linewidth=1.5,label='Type Test Std')
            ax.set_xticks(x-w/2); ax.set_xticklabels([f'{c} kVA' for c in CAPS_3F],fontsize=9)
            ax.set_ylabel('NLC (%)'); ax.set_title('Rata-rata NLC% vs Type Test Standar')
            ax.legend(fontsize=8); ax.grid(True,alpha=0.3,axis='y',zorder=0)
            ax.set_ylim(0,max([v for v in crgo if not np.isnan(v)])*1.35)
            ax.text(0.99,0.02,'*250/400/630 kVA: CRGO only',transform=ax.transAxes,
                    ha='right',va='bottom',fontsize=7,color='gray',style='italic')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            sec("Referensi Type Test Standar","📏")
            tt_rows=[]
            for c in CAPS_3F:
                for t in TIPE_PER_KVA_3F[c]:
                    tv=TYPE_TEST_3F[c][t]
                    tt_rows.append({'kVA':c,'Tipe':t,'Type Test (%)':tv,'Toleransi +30% (%)':round(tv*1.30,4)})
            st.dataframe(pd.DataFrame(tt_rows),use_container_width=True,height=280,hide_index=True)

    # ── TAB 2: BATCH ──
    with tab2:
        sec("Batch Prediksi","📦")

        def run_batch_3f(df_in):
            results=[]
            for _,row in df_in.iterrows():
                kva=int(row['kapasitas']); tipe=row['tipe_inti']; loss=float(row['nl_loss'])
                enc=le3.transform([tipe])[0]; pred=rf3.predict([[kva,enc,loss]])[0]
                tt=TYPE_TEST_3F[kva][tipe]; tol=tt*1.30
                ref_l=LOSS_REF_3F[kva][tipe]; dev=(loss-ref_l)/ref_l*100
                status="✅ LULUS TYPE TEST" if pred<=tt else ("⚠️ DALAM TOLERANSI ±30%" if pred<=tol else "❌ TIDAK MEMENUHI STANDAR")
                results.append({'kapasitas':kva,'tipe_inti':tipe,'nl_loss':loss,
                    'nlc_prediksi_%':round(pred,6),'type_test_%':tt,
                    'toleransi_+30%':round(tol,4),'status':status,'deviasi_loss_%':round(dev,2)})
            return pd.DataFrame(results)

        def show_batch_summary(df_res, status_col='status'):
            n_l=(df_res[status_col]=='✅ LULUS TYPE TEST').sum()
            n_t=(df_res[status_col]=='⚠️ DALAM TOLERANSI ±30%').sum()
            n_f=(df_res[status_col]=='❌ TIDAK MEMENUHI STANDAR').sum()
            s1,s2,s3,s4=st.columns(4)
            with s1: st.metric("Total Unit",len(df_res))
            with s2: st.metric("✅ Lulus",n_l)
            with s3: st.metric("⚠️ Toleransi",n_t)
            with s4: st.metric("❌ Tidak Memenuhi",n_f)

        up_tab, man_tab = st.tabs(["📁 Upload CSV","✏️ Input Manual"])
        with up_tab:
            st.markdown(f'<div style="font-size:12px;color:{TEXT3};margin-bottom:10px;">'
                        f'Format: <b>kapasitas</b> (50/100/160/250/400/630), <b>tipe_inti</b>, <b>nl_loss</b></div>',
                        unsafe_allow_html=True)
            uploaded=st.file_uploader("Upload CSV",type=['csv'],key="batch_up_3f")
            if uploaded:
                try:
                    df_up=pd.read_csv(uploaded)
                    req={'kapasitas','tipe_inti','nl_loss'}
                    if not req.issubset(df_up.columns):
                        st.error(f"❌ Kolom wajib: {req}")
                    else:
                        def valid_combo(row):
                            return row['tipe_inti'] in TIPE_PER_KVA_3F.get(row['kapasitas'],[])
                        df_up['_v']=df_up.apply(valid_combo,axis=1)
                        df_valid=df_up[df_up['_v']].drop(columns=['_v'])
                        if len(df_valid):
                            st.success(f"✅ {len(df_valid)} baris valid")
                            st.dataframe(df_valid.head(10),use_container_width=True)
                            if st.button("⚡ Prediksi dari CSV",key="pred_csv_3f"):
                                df_res=run_batch_3f(df_valid)
                                show_batch_summary(df_res)
                                st.dataframe(df_res,use_container_width=True)
                                st.download_button("📥 Download",data=df_res.to_csv(index=False).encode(),
                                    file_name="hasil_batch_3f.csv",mime="text/csv",key="dl_csv_3f")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            tmpl=pd.DataFrame({'kapasitas':[50,50,100,100,160,160,250,400,630],
                'tipe_inti':['CRGO','Amorphous','CRGO','Amorphous','CRGO','Amorphous','CRGO','CRGO','CRGO'],
                'nl_loss':[84,36,129,40,183,77,274,389,555]})
            st.download_button("📄 Download Template",data=tmpl.to_csv(index=False).encode(),
                file_name="template_3fasa.csv",mime="text/csv",key="dl_tmpl_3f")

        with man_tab:
            ca,cb,cc,cd=st.columns([2,2,2,1])
            with ca: new_kva=st.selectbox("Kapasitas",CAPS_3F,key="new_kva_3f")
            with cb:
                valid_n=TIPE_PER_KVA_3F[new_kva]
                new_tipe=st.selectbox("Tipe Inti",valid_n,key="new_ti_3f")
            with cc: new_loss=st.number_input("NL Loss (W)",10,800,LOSS_REF_3F[new_kva][new_tipe],key="new_loss_3f")
            with cd:
                st.markdown('<div style="margin-top:24px;"></div>',unsafe_allow_html=True)
                if st.button("➕ Tambah",key="add_3f"):
                    st.session_state.batch_rows.append({'kapasitas':new_kva,'tipe_inti':new_tipe,'nl_loss':new_loss})
                    st.rerun()
            if st.session_state.batch_rows:
                df_m=pd.DataFrame(st.session_state.batch_rows)
                st.dataframe(df_m,use_container_width=True,height=180)
                cp,cc2=st.columns(2)
                with cp:
                    if st.button("⚡ Prediksi Semua",key="pred_man_3f"):
                        st.session_state.batch_results=run_batch_3f(df_m)
                with cc2:
                    if st.button("🗑️ Hapus Semua",key="clear_3f"):
                        st.session_state.batch_rows=[]; st.session_state.batch_results=None; st.rerun()
            if st.session_state.batch_results is not None:
                df_res=st.session_state.batch_results
                show_batch_summary(df_res)
                st.dataframe(df_res,use_container_width=True,height=250)
                st.download_button("📥 Download",data=df_res.to_csv(index=False).encode(),
                    file_name="hasil_batch_3f.csv",mime="text/csv",key="dl_man_3f")

    # ── TAB 3: EVALUASI ──
    with tab3:
        set_plot()
        m1,m2,m3,m4=st.columns(4)
        with m1: st.metric("R² Score",f"{r2_3:.4f}")
        with m2: st.metric("RMSE (%)",f"{rmse_3:.6f}")
        with m3: st.metric("MAE (%)",f"{mae_3:.6f}")
        with m4: st.metric("Akurasi ±0.01%",f"{acc_3:.1f}%")
        st.markdown('<div style="margin-top:1rem;"></div>',unsafe_allow_html=True)

        col1,col2=st.columns(2,gap="large")
        with col1:
            sec("Aktual vs Prediksi")
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            for tipe,color in [('CRGO','#185FA5'),('Amorphous','#0F6E56')]:
                idx=[i for i,x in enumerate(Xte_3) if le3.classes_[int(round(x[1]))]==tipe]
                ax.scatter([yte_3[i] for i in idx],[ype_3[i] for i in idx],c=color,label=tipe,alpha=0.7,s=45,edgecolors='none',zorder=3)
            mn=min(yte_3.min(),ype_3.min()); mx=max(yte_3.max(),ype_3.max())
            ax.plot([mn,mx],[mn,mx],'r--',linewidth=1.5,alpha=0.6,label='Ideal')
            ax.set_xlabel('Aktual NLC (%)'); ax.set_ylabel('Prediksi NLC (%)')
            ax.set_title(f'Aktual vs Prediksi  (R² = {r2_3:.4f})')
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()
        with col2:
            sec("Feature Importance")
            imps=rf3.feature_importances_
            fnames=['Kapasitas (kVA)','Tipe Inti','NL Loss (W)']; fclrs=['#185FA5','#0F6E56','#BA7517']
            si=np.argsort(imps)
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            bars=ax.barh([fnames[i] for i in si],[imps[i] for i in si],
                         color=[fclrs[i] for i in si],edgecolor='none',height=0.5,zorder=3)
            for bar,val in zip(bars,[imps[i] for i in si]):
                ax.text(bar.get_width()+0.008,bar.get_y()+bar.get_height()/2,
                        f'{val*100:.1f}%',va='center',fontsize=11,fontweight='700')
            ax.set_xlabel('Importance'); ax.set_xlim(0,max(imps)*1.4)
            ax.set_title('Feature Importance'); ax.grid(True,alpha=0.3,axis='x',zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        sec("Distribusi Error Prediksi")
        fig,ax=plt.subplots(figsize=(11,3.2))
        errors=ype_3-yte_3
        ax.hist(errors,bins=25,color='#185FA5',alpha=0.8,edgecolor='none',zorder=3)
        ax.axvline(0,color='#f87171',linestyle='--',linewidth=2,label='Error=0',zorder=4)
        ax.axvline(errors.mean(),color='#fbbf24',linestyle='-',linewidth=2,label=f'Mean={errors.mean():.6f}',zorder=4)
        ax.set_xlabel('Error (%)'); ax.set_ylabel('Jumlah unit')
        ax.set_title('Distribusi Error — Mayoritas mendekati nol = model akurat')
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TAB 4: DATA ──
    with tab4:
        set_plot()
        sec("Dataset Asli — 3 Fasa","📋")
        f1,f2,f3=st.columns(3)
        with f1: fkva=st.multiselect("Filter Kapasitas",CAPS_3F,default=CAPS_3F,key="flt_kva_3f")
        with f2: fcore=st.multiselect("Filter Tipe Inti",['CRGO','Amorphous'],default=['CRGO','Amorphous'],key="flt_core_3f")
        with f3: st.metric("Total data",len(df3))
        df_f=df3[(df3['kapasitas'].isin(fkva))&(df3['tipe_inti'].isin(fcore))]
        st.dataframe(df_f.style.format({'nlc_persen':'{:.6f}','nl_loss':'{:.4f}'}),use_container_width=True,height=300)
        st.markdown(f'<div style="font-size:12px;color:{TEXT3};margin-top:4px;">Menampilkan '
                    f'<b style="color:{ACCENT2};">{len(df_f)}</b> dari <b>{len(df3)}</b> data</div>',
                    unsafe_allow_html=True)
        st.download_button("📥 Download CSV",data=df_f.to_csv(index=False).encode(),
            file_name="data_3fasa_filter.csv",mime="text/csv",key="dl_data_3f")

        st.markdown('<div style="margin-top:1.2rem;"></div>',unsafe_allow_html=True)
        dc1,dc2=st.columns(2,gap="large")
        with dc1:
            sec("Statistik per Kapasitas & Tipe Inti")
            if len(df_f):
                stats=df_f.groupby(['kapasitas','tipe_inti'])['nlc_persen'].agg(['mean','min','max','std']).round(6)
                stats.columns=['Rata-rata','Min','Max','Std Dev']
                st.dataframe(stats,use_container_width=True)
        with dc2:
            sec("NL Loss vs NLC%")
            fig,ax=plt.subplots(figsize=(5,3.8))
            for tipe,color,mk in [('CRGO','#185FA5','o'),('Amorphous','#0F6E56','^')]:
                d=df_f[df_f['tipe_inti']==tipe]
                if len(d):
                    ax.scatter(d['nl_loss'],d['nlc_persen'],c=color,alpha=0.65,s=35,label=tipe,marker=mk,edgecolors='none',zorder=3)
            ax.set_xlabel('NL Loss (W)'); ax.set_ylabel('NLC (%)'); ax.set_title('NL Loss vs NLC%')
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
#  MODE: 1 FASA
# ══════════════════════════════════════════════════════════════
else:
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediksi", "📦 Batch Prediksi", "📊 Evaluasi Model", "📋 Data"])

    # ── TAB 1: PREDIKSI TUNGGAL 1F ──
    with tab1:
        col1,col2=st.columns([1,1],gap="large")
        with col1:
            sec("Input yang Dimasukkan")
            ia,ib=st.columns(2)
            with ia: st.metric("Kapasitas","50 kVA")
            with ib: st.metric("Tipe Inti",st.session_state.tipe_1f)
            st.metric("NL Loss",f"{st.session_state.nl_loss_1f} W")

            if st.session_state.hasil_pred_1f:
                p=st.session_state.hasil_pred_1f
                render_pred_result(p['nilai'],p['tt'],p['nl_loss'],p['ref_loss'],p['tipe_inti'],"(1 Fasa)")
            else:
                st.markdown(f"""
                <div style="text-align:center;padding:80px 0;color:{TEXT3};">
                    <div style="font-size:52px;">⚡</div>
                    <div style="font-size:13px;margin-top:10px;">
                        Pilih tipe inti & NL Loss di sidebar<br>lalu klik
                        <b style="color:{ACCENT2};">Prediksi NLC%</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            sec("NLC% vs NL Loss — Data Asli")
            set_plot()
            fig,ax=plt.subplots(figsize=(6,4.5))
            for tipe,color,mk in [('CRGO','#185FA5','o'),('Amorphous','#0F6E56','^')]:
                d=df1[df1['tipe_inti']==tipe]
                if len(d):
                    ax.scatter(d['nl_loss'],d['nlc_persen'],c=color,alpha=0.6,s=40,label=f'{tipe} Data',marker=mk,edgecolors='none',zorder=3)
                    nl_s=np.sort(d['nl_loss'].values).reshape(-1,1)
                    enc_s=np.full(len(nl_s),le1.transform([tipe])[0])
                    pred_s=rf1.predict(np.column_stack([nl_s,enc_s]))
                    ax.plot(nl_s,pred_s,color=color,linewidth=2,linestyle='-',label=f'{tipe} Model',zorder=4)
                    tt_v=TYPE_TEST_1F[tipe]
                    clr_tt='#f87171' if tipe=='CRGO' else '#4ade80'
                    ax.axhline(tt_v,color=clr_tt,linestyle='--',linewidth=1.2,alpha=0.8,label=f'Type Test {tipe} {tt_v}%')
                    ax.axhline(tt_v*1.3,color=clr_tt,linestyle=':',linewidth=1.2,alpha=0.6,label=f'Tol+30% {tipe} {tt_v*1.3:.3f}%')
            ax.set_xlabel('NL Loss (W)'); ax.set_ylabel('NLC (%)'); ax.set_title('NLC% vs NL Loss — 50 kVA 1 Fasa')
            ax.legend(fontsize=7.5,loc='upper right'); ax.grid(True,alpha=0.3,zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            sec("Referensi Type Test Standar","📏")
            tt_rows_1f=[
                {'Tipe Inti':'CRGO','Kapasitas':'50 kVA','Type Test (%)':1.23,'Toleransi +30% (%)':round(1.23*1.3,4)},
                {'Tipe Inti':'Amorphous','Kapasitas':'50 kVA','Type Test (%)':0.40,'Toleransi +30% (%)':round(0.40*1.3,4)},
            ]
            st.dataframe(pd.DataFrame(tt_rows_1f),use_container_width=True,hide_index=True)

    # ── TAB 2: BATCH 1F ──
    with tab2:
        sec("Batch Prediksi — 1 Fasa","📦")

        def run_batch_1f(df_in):
            results=[]
            for _,row in df_in.iterrows():
                tipe=row['tipe_inti']; loss=float(row['nl_loss'])
                enc=le1.transform([tipe])[0]; pred=rf1.predict([[loss,enc]])[0]
                tt=TYPE_TEST_1F[tipe]; tol=tt*1.30
                ref_l=LOSS_REF_1F[tipe]; dev=(loss-ref_l)/ref_l*100
                status="✅ LULUS TYPE TEST" if pred<=tt else ("⚠️ DALAM TOLERANSI ±30%" if pred<=tol else "❌ TIDAK MEMENUHI STANDAR")
                results.append({'tipe_inti':tipe,'nl_loss':loss,'nlc_prediksi_%':round(pred,6),
                    'type_test_%':tt,'toleransi_+30%':round(tol,4),'status':status,'deviasi_loss_%':round(dev,2)})
            return pd.DataFrame(results)

        def show_batch_summary_1f(df_res):
            n_l=(df_res['status']=='✅ LULUS TYPE TEST').sum()
            n_t=(df_res['status']=='⚠️ DALAM TOLERANSI ±30%').sum()
            n_f=(df_res['status']=='❌ TIDAK MEMENUHI STANDAR').sum()
            s1,s2,s3,s4=st.columns(4)
            with s1: st.metric("Total Unit",len(df_res))
            with s2: st.metric("✅ Lulus",n_l)
            with s3: st.metric("⚠️ Toleransi",n_t)
            with s4: st.metric("❌ Tidak Memenuhi",n_f)

        up_tab_1f,man_tab_1f=st.tabs(["📁 Upload CSV","✏️ Input Manual"])
        with up_tab_1f:
            st.markdown(f'<div style="font-size:12px;color:{TEXT3};margin-bottom:10px;">'
                        f'Format: <b>tipe_inti</b> (CRGO/Amorphous), <b>nl_loss</b></div>',
                        unsafe_allow_html=True)
            uploaded_1f=st.file_uploader("Upload CSV",type=['csv'],key="batch_up_1f")
            if uploaded_1f:
                try:
                    df_up_1f=pd.read_csv(uploaded_1f)
                    req={'tipe_inti','nl_loss'}
                    if not req.issubset(df_up_1f.columns):
                        st.error(f"❌ Kolom wajib: {req}")
                    else:
                        df_valid_1f=df_up_1f[df_up_1f['tipe_inti'].isin(TIPE_1F)]
                        if len(df_valid_1f):
                            st.success(f"✅ {len(df_valid_1f)} baris valid")
                            st.dataframe(df_valid_1f.head(10),use_container_width=True)
                            if st.button("⚡ Prediksi dari CSV",key="pred_csv_1f"):
                                df_res_1f=run_batch_1f(df_valid_1f)
                                show_batch_summary_1f(df_res_1f)
                                st.dataframe(df_res_1f,use_container_width=True)
                                st.download_button("📥 Download",data=df_res_1f.to_csv(index=False).encode(),
                                    file_name="hasil_batch_1f.csv",mime="text/csv",key="dl_csv_1f_res")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            tmpl_1f=pd.DataFrame({'tipe_inti':['CRGO','CRGO','Amorphous','Amorphous'],
                                   'nl_loss':[110,115,120,125]})
            st.download_button("📄 Download Template",data=tmpl_1f.to_csv(index=False).encode(),
                file_name="template_1fasa.csv",mime="text/csv",key="dl_tmpl_1f")

        with man_tab_1f:
            ca_1f,cb_1f,cc_1f=st.columns([2,3,1])
            with ca_1f: new_tipe_1f=st.selectbox("Tipe Inti",TIPE_1F,key="new_ti_1f")
            with cb_1f: new_loss_1f=st.number_input("NL Loss (W)",50,250,LOSS_REF_1F[new_tipe_1f],key="new_loss_1f")
            with cc_1f:
                st.markdown('<div style="margin-top:24px;"></div>',unsafe_allow_html=True)
                if st.button("➕",key="add_1f"):
                    st.session_state.batch_rows_1f.append({'tipe_inti':new_tipe_1f,'nl_loss':new_loss_1f})
                    st.rerun()
            if st.session_state.batch_rows_1f:
                df_m_1f=pd.DataFrame(st.session_state.batch_rows_1f)
                st.dataframe(df_m_1f,use_container_width=True,height=180)
                cp1f,cc_1f2=st.columns(2)
                with cp1f:
                    if st.button("⚡ Prediksi Semua",key="pred_man_1f"):
                        st.session_state.batch_results_1f=run_batch_1f(df_m_1f)
                with cc_1f2:
                    if st.button("🗑️ Hapus Semua",key="clear_1f"):
                        st.session_state.batch_rows_1f=[]; st.session_state.batch_results_1f=None; st.rerun()
            if st.session_state.batch_results_1f is not None:
                df_res_1f=st.session_state.batch_results_1f
                show_batch_summary_1f(df_res_1f)
                st.dataframe(df_res_1f,use_container_width=True,height=250)
                st.download_button("📥 Download",data=df_res_1f.to_csv(index=False).encode(),
                    file_name="hasil_batch_1f.csv",mime="text/csv",key="dl_man_1f")

    # ── TAB 3: EVALUASI 1F ──
    with tab3:
        set_plot()
        m1,m2,m3,m4=st.columns(4)
        with m1: st.metric("R² Score",f"{r2_1:.4f}")
        with m2: st.metric("RMSE (%)",f"{rmse_1:.6f}")
        with m3: st.metric("MAE (%)",f"{mae_1:.6f}")
        with m4: st.metric("Akurasi ±0.01%",f"{acc_1:.1f}%")
        st.markdown('<div style="margin-top:1rem;"></div>',unsafe_allow_html=True)

        col1,col2=st.columns(2,gap="large")
        with col1:
            sec("Aktual vs Prediksi")
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            for tipe,color in [('CRGO','#185FA5'),('Amorphous','#0F6E56')]:
                idx=[i for i,x in enumerate(Xte_1) if le1.classes_[int(round(x[1]))]==tipe]
                ax.scatter([yte_1[i] for i in idx],[ype_1[i] for i in idx],c=color,label=tipe,alpha=0.7,s=45,edgecolors='none',zorder=3)
            mn=min(yte_1.min(),ype_1.min()); mx=max(yte_1.max(),ype_1.max())
            ax.plot([mn,mx],[mn,mx],'r--',linewidth=1.5,alpha=0.6,label='Ideal')
            ax.set_xlabel('Aktual NLC (%)'); ax.set_ylabel('Prediksi NLC (%)')
            ax.set_title(f'Aktual vs Prediksi  (R² = {r2_1:.4f})')
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            sec("NLC% vs NL Loss per Tipe Inti")
            fig,ax=plt.subplots(figsize=(5.5,4.5))
            for tipe,color in [('CRGO','#185FA5'),('Amorphous','#0F6E56')]:
                d=df1[df1['tipe_inti']==tipe]
                ax.scatter(d['nl_loss'],d['nlc_persen'],c=color,alpha=0.5,s=30,label=f'{tipe}',edgecolors='none',zorder=3)
                nl_s=np.sort(d['nl_loss'].values).reshape(-1,1)
                enc_s=np.full(len(nl_s),le1.transform([tipe])[0])
                pred_s=rf1.predict(np.column_stack([nl_s,enc_s]))
                ax.plot(nl_s,pred_s,color=color,linewidth=2,label=f'{tipe} Model',zorder=4)
            ax.set_xlabel('NL Loss (W)'); ax.set_ylabel('NLC (%)')
            ax.set_title('NLC% vs NL Loss — Model RF')
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        sec("Distribusi Error Prediksi")
        fig,ax=plt.subplots(figsize=(11,3.2))
        errors_1=ype_1-yte_1
        ax.hist(errors_1,bins=20,color='#185FA5',alpha=0.8,edgecolor='none',zorder=3)
        ax.axvline(0,color='#f87171',linestyle='--',linewidth=2,label='Error=0',zorder=4)
        ax.axvline(errors_1.mean(),color='#fbbf24',linestyle='-',linewidth=2,label=f'Mean={errors_1.mean():.6f}',zorder=4)
        ax.set_xlabel('Error (%)'); ax.set_ylabel('Jumlah unit')
        ax.set_title('Distribusi Error — 1 Fasa 50 kVA CRGO & Amorphous')
        ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── TAB 4: DATA 1F ──
    with tab4:
        set_plot()
        sec("Dataset Asli — 1 Fasa (50 kVA)","📋")
        ff1,ff2,ff3=st.columns(3)
        with ff1: fcore_1f=st.multiselect("Filter Tipe Inti",TIPE_1F,default=TIPE_1F,key="flt_1f")
        with ff3: st.metric("Total data",len(df1))
        df_f1=df1[df1['tipe_inti'].isin(fcore_1f)]
        st.dataframe(df_f1.style.format({'nlc_persen':'{:.6f}','nl_loss':'{:.4f}'}),use_container_width=True,height=300)
        st.markdown(f'<div style="font-size:12px;color:{TEXT3};margin-top:4px;">Menampilkan '
                    f'<b style="color:{ACCENT2};">{len(df_f1)}</b> dari <b>{len(df1)}</b> data</div>',
                    unsafe_allow_html=True)
        st.download_button("📥 Download CSV",data=df_f1.to_csv(index=False).encode(),
            file_name="data_1fasa_filter.csv",mime="text/csv",key="dl_data_1f")

        st.markdown('<div style="margin-top:1.2rem;"></div>',unsafe_allow_html=True)
        dc1,dc2=st.columns(2,gap="large")
        with dc1:
            sec("Statistik per Tipe Inti")
            stats_1f=df_f1.groupby('tipe_inti')['nlc_persen'].agg(['mean','min','max','std','count']).round(6)
            stats_1f.columns=['Rata-rata','Min','Max','Std Dev','Jumlah']
            st.dataframe(stats_1f,use_container_width=True)
        with dc2:
            sec("Distribusi NLC% per Tipe Inti")
            fig,ax=plt.subplots(figsize=(5,3.8))
            for tipe,color in [('CRGO','#185FA5'),('Amorphous','#0F6E56')]:
                d=df_f1[df_f1['tipe_inti']==tipe]
                if len(d):
                    ax.hist(d['nlc_persen'],bins=12,alpha=0.65,color=color,label=tipe,edgecolor='none',zorder=3)
            ax.set_xlabel('NLC (%)'); ax.set_ylabel('Jumlah unit')
            ax.set_title('Distribusi NLC% — 1 Fasa 50 kVA')
            ax.legend(fontsize=9); ax.grid(True,alpha=0.3,zorder=0)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ── FOOTER ───────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;color:{TEXT3};font-size:11px;margin-top:2rem;'
    f'padding:12px;border-top:1px solid {BORDER};">'
    f'Prediksi No Load Current Trafo · PT Bambang Djaja · Random Forest · v4.0 · '
    f'3 Fasa (431 unit) + 1 Fasa (96 unit)</div>',
    unsafe_allow_html=True
)