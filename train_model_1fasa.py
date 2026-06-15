# ============================================================
#  PEMODELAN NO-LOAD CURRENT (%) TRANSFORMATOR DISTRIBUSI 1 FASA
#  PT Bambang Djaja — Random Forest
#  Input  : nl_loss (W), tipe_inti (CRGO / Amorphous)
#  Output : nlc_persen (%)
#  Data   : 96 unit asli (50 kVA · CRGO 48 + Amorphous 48)
#  v2.0   : Support 2 tipe inti
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
import joblib, os

DATA_PATH  = 'data_1fasa.csv'
MODEL_DIR  = 'model_1fasa'
MODEL_PATH = os.path.join(MODEL_DIR, 'model_trafo_1fasa.pkl')
LE_PATH    = os.path.join(MODEL_DIR, 'label_encoder_1fasa.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("  PEMODELAN NO-LOAD CURRENT (%) TRAFO 1 FASA — PT BAMBANG DJAJA")
print("  v2.0 — CRGO & Amorphous")
print("="*60)

# 1. LOAD DATA
print("\n[1] Load data...")
df = pd.read_csv(DATA_PATH)
print(f"    Total data : {len(df)} unit")
print(f"\n    Distribusi dataset:")
dist = df.groupby('tipe_inti').size().reset_index(name='jumlah')
for _, row in dist.iterrows():
    print(f"    50 kVA  {row['tipe_inti']:<12}  {row['jumlah']:>3} data")

# 2. STATISTIK
print("\n[2] Statistik NLC% per Tipe Inti:")
stats = df.groupby('tipe_inti')['nlc_persen'].agg(['mean','min','max','std']).round(6)
stats.columns = ['Rata-rata','Min','Max','Std Dev']
print(stats.to_string())

# 3. PREPROCESSING
print("\n[3] Preprocessing...")
le = LabelEncoder()
df['tipe_inti_enc'] = le.fit_transform(df['tipe_inti'])
print(f"    Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
X = df[['nl_loss', 'tipe_inti_enc']].values
y = df['nlc_persen'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

# 4. TRAINING
print("\n[4] Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=100, max_depth=10,
    min_samples_split=2, min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("    Selesai!")

# 5. EVALUASI
print("\n[5] Evaluasi model...")
yp_tr = rf.predict(X_train)
yp_te = rf.predict(X_test)
cv    = cross_val_score(rf, X, y, cv=5, scoring='r2')

print(f"\n    {'Metrik':<20} {'Train':>10} {'Test':>10}")
print(f"    {'-'*42}")
print(f"    {'R² Score':<20} {r2_score(y_train,yp_tr):>10.4f} {r2_score(y_test,yp_te):>10.4f}")
print(f"    {'RMSE':<20} {np.sqrt(mean_squared_error(y_train,yp_tr)):>10.6f} {np.sqrt(mean_squared_error(y_test,yp_te)):>10.6f}")
print(f"    {'MAE':<20} {mean_absolute_error(y_train,yp_tr):>10.6f} {mean_absolute_error(y_test,yp_te):>10.6f}")
print(f"\n    Cross Val R² (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

for tol in [0.005, 0.01, 0.02]:
    acc = np.mean(np.abs(y_test - yp_te) <= tol) * 100
    print(f"    Akurasi ±{tol:.3f}: {acc:.1f}%")

imps   = rf.feature_importances_
fnames = ['NL Loss (W)', 'Tipe Inti']
print(f"\n    Feature Importance:")
for n, v in sorted(zip(fnames, imps), key=lambda x: -x[1]):
    print(f"    {n:<20} {v:.4f}  {'█'*int(v*40)}")

# 6. SIMPAN
print(f"\n[6] Simpan model...")
joblib.dump(rf, MODEL_PATH)
joblib.dump(le, LE_PATH)
print(f"    {MODEL_PATH} ✅")
print(f"    {LE_PATH} ✅")

# 7. VISUALISASI
print("\n[7] Membuat grafik...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    'Pemodelan No-Load Current (%) Transformator Distribusi 1 Fasa — PT Bambang Djaja\n'
    'Random Forest · Data Asli 96 Unit · 50 kVA · CRGO & Amorphous',
    fontsize=13, fontweight='bold'
)

r2_te = r2_score(y_test, yp_te)
colors_map = {'CRGO': '#185FA5', 'Amorphous': '#0F6E56'}

# Plot 1: Aktual vs Estimasi
ax1 = axes[0]
for tipe in ['CRGO', 'Amorphous']:
    idx = [i for i, x in enumerate(X_test) if le.classes_[int(round(x[1]))] == tipe]
    if idx:
        ax1.scatter([y_test[i] for i in idx], [yp_te[i] for i in idx],
                    c=colors_map[tipe], label=tipe, alpha=0.7, s=50, edgecolors='none')
mn = min(y_test.min(), yp_te.min())
mx = max(y_test.max(), yp_te.max())
ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, alpha=0.6, label='Ideal')
ax1.set_xlabel('Aktual NLC (%)'); ax1.set_ylabel('Estimasi NLC (%)')
ax1.set_title(f'Aktual vs Estimasi  (R²={r2_te:.4f})')
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

# Plot 2: NLC vs NL Loss per Tipe Inti
ax2 = axes[1]
for tipe in ['CRGO', 'Amorphous']:
    d = df[df['tipe_inti'] == tipe]
    ax2.scatter(d['nl_loss'], d['nlc_persen'],
                c=colors_map[tipe], alpha=0.55, s=35, label=f'{tipe} Data', edgecolors='none')
    nl_s   = np.sort(d['nl_loss'].values).reshape(-1, 1)
    enc_s  = np.full(len(nl_s), le.transform([tipe])[0])
    pred_s = rf.predict(np.column_stack([nl_s, enc_s]))
    ax2.plot(nl_s, pred_s, color=colors_map[tipe], linewidth=2, label=f'{tipe} Model')
    standar = d['standar'].iloc[0]
    ax2.axhline(standar,        color=colors_map[tipe], linestyle='--', linewidth=1.2,
                alpha=0.7, label=f'Std {tipe} {standar}%')
    ax2.axhline(standar * 1.3,  color=colors_map[tipe], linestyle=':',  linewidth=1.2,
                alpha=0.5, label=f'Tol+30% {standar*1.3:.3f}%')
ax2.set_xlabel('NL Loss (W)'); ax2.set_ylabel('NLC (%)')
ax2.set_title('NLC% vs NL Loss per Tipe Inti')
ax2.legend(fontsize=7, loc='upper right'); ax2.grid(True, alpha=0.3)

# Plot 3: Distribusi Error
ax3 = axes[2]
errors = yp_te - y_test
ax3.hist(errors, bins=15, color='#185FA5', alpha=0.75, edgecolor='white', linewidth=0.5)
ax3.axvline(0,             color='red',    linestyle='--', linewidth=1.5, label='Error=0')
ax3.axvline(errors.mean(), color='orange', linestyle='-',  linewidth=1.5, label=f'Mean={errors.mean():.6f}')
ax3.set_xlabel('Error (%)'); ax3.set_ylabel('Jumlah unit')
ax3.set_title('Distribusi Residual Estimasi')
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hasil_evaluasi_1fasa.png', dpi=150, bbox_inches='tight')
plt.show()
print("    hasil_evaluasi_1fasa.png ✅")

# 8. CONTOH ESTIMASI
print("\n[8] Contoh estimasi unit baru:")
print(f"    {'Tipe Inti':<12} {'NL Loss':>10}  {'Estimasi NLC%':>14}")
print(f"    {'-'*40}")
contoh = [
    ('CRGO',      106), ('CRGO',      110), ('CRGO',      115), ('CRGO',      120),
    ('Amorphous', 110), ('Amorphous', 118), ('Amorphous', 124), ('Amorphous', 128),
]
for tipe, loss in contoh:
    enc  = le.transform([tipe])[0]
    pred = rf.predict([[loss, enc]])[0]
    print(f"    {tipe:<12} {loss:>10} W  {pred:>14.6f} %")

print("\n" + "="*60)
print("  SELESAI! Model 1 fasa (CRGO & Amorphous) siap digunakan.")
print("="*60)