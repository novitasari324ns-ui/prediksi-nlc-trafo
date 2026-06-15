# ============================================================
#  PEMODELAN NO-LOAD CURRENT (%) TRANSFORMATOR DISTRIBUSI 3 FASA
#  PT Bambang Djaja — Random Forest
#  Input  : kapasitas (kVA), tipe inti, nl_loss (W)
#  Output : nlc_persen (%)
#  Data   : 431 unit asli (50,100,160,250,400,630 kVA)
#  Note   : 250,400,630 kVA hanya CRGO (kondisi nyata industri)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble        import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import r2_score, mean_squared_error, mean_absolute_error
import joblib, os

DATA_PATH  = 'data_asli.csv'
MODEL_DIR  = 'model'
MODEL_PATH = os.path.join(MODEL_DIR,'model_trafo.pkl')
LE_PATH    = os.path.join(MODEL_DIR,'label_encoder.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*60)
print("  PEMODELAN NO-LOAD CURRENT (%) — PT BAMBANG DJAJA")
print("="*60)

# 1. LOAD DATA
print("\n[1] Load data...")
df = pd.read_csv(DATA_PATH)
print(f"    Total data : {len(df)} unit")
print(f"\n    Distribusi dataset (kondisi nyata industri):")
dist = df.groupby(['kapasitas','tipe_inti']).size().reset_index(name='jumlah')
for _,row in dist.iterrows():
    note = " ← CRGO & Amorphous" if row['kapasitas']<=160 else " ← CRGO only (kondisi nyata)"
    print(f"    {row['kapasitas']:>4} kVA  {row['tipe_inti']:<12}  {row['jumlah']:>3} data{note}")

print(f"\n    📌 Note: Kapasitas 250, 400, 630 kVA hanya tersedia")
print(f"       dengan inti CRGO sesuai implementasi aktual")
print(f"       PT Bambang Djaja. Dataset TIDAK menggunakan")
print(f"       data sintetis atau manipulasi data.")

# 2. STATISTIK
print("\n[2] Statistik NLC% per kapasitas & tipe inti:")
stats = df.groupby(['kapasitas','tipe_inti'])['nlc_persen'].agg(['mean','min','max','std']).round(6)
stats.columns = ['Rata-rata','Min','Max','Std Dev']
print(stats.to_string())

# 3. PREPROCESSING
print("\n[3] Preprocessing...")
le = LabelEncoder()
df['tipe_inti_enc'] = le.fit_transform(df['tipe_inti'])
print(f"    Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
X = df[['kapasitas','tipe_inti_enc','nl_loss']].values
y = df['nlc_persen'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

# 4. TRAINING
print("\n[4] Training Random Forest...")
rf = RandomForestRegressor(n_estimators=100,max_depth=10,
    min_samples_split=2,min_samples_leaf=1,random_state=42,n_jobs=-1)
rf.fit(X_train, y_train)
print("    Selesai!")

# 5. EVALUASI
print("\n[5] Evaluasi model...")
yp_tr = rf.predict(X_train)
yp_te = rf.predict(X_test)
cv    = cross_val_score(rf,X,y,cv=5,scoring='r2')

print(f"\n    {'Metrik':<20} {'Train':>10} {'Test':>10}")
print(f"    {'-'*42}")
print(f"    {'R² Score':<20} {r2_score(y_train,yp_tr):>10.4f} {r2_score(y_test,yp_te):>10.4f}")
print(f"    {'RMSE':<20} {np.sqrt(mean_squared_error(y_train,yp_tr)):>10.6f} {np.sqrt(mean_squared_error(y_test,yp_te)):>10.6f}")
print(f"    {'MAE':<20} {mean_absolute_error(y_train,yp_tr):>10.6f} {mean_absolute_error(y_test,yp_te):>10.6f}")
print(f"\n    Cross Val R² (5-fold): {cv.mean():.4f} ± {cv.std():.4f}")

for tol in [0.005,0.01,0.02]:
    acc = np.mean(np.abs(y_test-yp_te)<=tol)*100
    print(f"    Akurasi ±{tol:.3f}: {acc:.1f}%")

imps = rf.feature_importances_
fnames = ['Kapasitas (kVA)','Tipe Inti','NL Loss (W)']
print(f"\n    Feature Importance:")
for n,v in sorted(zip(fnames,imps),key=lambda x:-x[1]):
    print(f"    {n:<20} {v:.4f}  {'█'*int(v*40)}")

# 6. SIMPAN
print(f"\n[6] Simpan model...")
joblib.dump(rf,MODEL_PATH); joblib.dump(le,LE_PATH)
print(f"    {MODEL_PATH} ✅")
print(f"    {LE_PATH} ✅")

# 7. VISUALISASI
print("\n[7] Membuat grafik...")
fig,axes=plt.subplots(2,2,figsize=(14,10))
fig.suptitle('Pemodelan No-Load Current (%) Transformator Distribusi — PT Bambang Djaja\nRandom Forest · Data Asli 431 Unit · 6 Kapasitas',
             fontsize=13,fontweight='bold')

r2_te = r2_score(y_test,yp_te)
# Plot 1: Aktual vs Estimasi
ax1=axes[0,0]
colors_map={'CRGO':'#185FA5','Amorphous':'#0F6E56'}
for tipe in ['CRGO','Amorphous']:
    idx=[i for i,x in enumerate(X_test) if le.classes_[int(round(x[1]))]==tipe]
    if idx:
        ax1.scatter([y_test[i] for i in idx],[yp_te[i] for i in idx],
                    c=colors_map[tipe],label=tipe,alpha=0.6,s=30,edgecolors='none')
mn,mx=min(y_test.min(),yp_te.min()),max(y_test.max(),yp_te.max())
ax1.plot([mn,mx],[mn,mx],'r--',linewidth=1.5,alpha=0.6,label='Ideal')
ax1.set_xlabel('Aktual NLC (%)'); ax1.set_ylabel('Estimasi NLC (%)')
ax1.set_title(f'Aktual vs Estimasi  (R²={r2_te:.4f})')
ax1.legend(fontsize=9); ax1.grid(True,alpha=0.3)

# Plot 2: Feature Importance
ax2=axes[0,1]
si=np.argsort(imps)
ax2.barh([fnames[i] for i in si],[imps[i] for i in si],
         color=['#185FA5','#0F6E56','#BA7517'],edgecolor='none',height=0.5)
for i,v in enumerate([imps[j] for j in si]):
    ax2.text(v+0.005,i,f'{v*100:.1f}%',va='center',fontsize=10,fontweight='bold')
ax2.set_xlabel('Importance'); ax2.set_xlim(0,max(imps)*1.35)
ax2.set_title('Feature Importance'); ax2.grid(True,alpha=0.3,axis='x')

# Plot 3: NLC per Kapasitas
ax3=axes[1,0]
caps_all=[50,100,160,250,400,630]
x=np.arange(len(caps_all)); w=0.35
crgo=[df[(df['kapasitas']==c)&(df['tipe_inti']=='CRGO')]['nlc_persen'].mean() for c in caps_all]
amor=[df[(df['kapasitas']==c)&(df['tipe_inti']=='Amorphous')]['nlc_persen'].mean() if len(df[(df['kapasitas']==c)&(df['tipe_inti']=='Amorphous')])>0 else 0 for c in caps_all]
b1=ax3.bar(x-w/2,crgo,w,label='CRGO',color='#185FA5',alpha=0.85,edgecolor='none')
b2=ax3.bar(x+w/2,[v if v>0 else np.nan for v in amor],w,label='Amorphous',color='#0F6E56',alpha=0.85,edgecolor='none')
for b in list(b1):
    if b.get_height()>0:
        ax3.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f'{b.get_height():.4f}',ha='center',va='bottom',fontsize=7)
for b in list(b2):
    if b.get_height()>0 and not np.isnan(b.get_height()):
        ax3.text(b.get_x()+b.get_width()/2,b.get_height()+0.003,f'{b.get_height():.4f}',ha='center',va='bottom',fontsize=7)
ax3.set_xticks(x); ax3.set_xticklabels([f'{c}\nkVA' for c in caps_all],fontsize=8)
ax3.set_ylabel('NLC (%)'); ax3.set_title('Rata-rata NLC% per Kapasitas')
ax3.legend(fontsize=9); ax3.grid(True,alpha=0.3,axis='y')
ax3.text(0.99,0.98,'*250,400,630 kVA: CRGO only\n(kondisi nyata industri)',
         transform=ax3.transAxes,ha='right',va='top',fontsize=7,color='gray',style='italic')

# Plot 4: Error
ax4=axes[1,1]
errors=yp_te-y_test
ax4.hist(errors,bins=25,color='#185FA5',alpha=0.75,edgecolor='white',linewidth=0.5)
ax4.axvline(0,color='red',linestyle='--',linewidth=1.5,label='Error=0')
ax4.axvline(errors.mean(),color='orange',linestyle='-',linewidth=1.5,label=f'Mean={errors.mean():.6f}')
ax4.set_xlabel('Error (%)'); ax4.set_ylabel('Jumlah unit')
ax4.set_title('Distribusi Residual Estimasi')
ax4.legend(fontsize=9); ax4.grid(True,alpha=0.3)

plt.tight_layout()
plt.savefig('hasil_evaluasi.png',dpi=150,bbox_inches='tight')
plt.show()
print("    hasil_evaluasi.png ✅")

# 8. CONTOH ESTIMASI
print("\n[8] Contoh estimasi unit baru:")
print(f"    {'kVA':<6} {'Tipe Inti':<12} {'NL Loss':>10}  {'Estimasi NLC%':>14}")
print(f"    {'-'*50}")
contoh=[(50,'Amorphous',38),(50,'CRGO',84),(100,'Amorphous',35),(100,'CRGO',129),
        (160,'Amorphous',78),(160,'CRGO',183),(250,'CRGO',295),(400,'CRGO',393),(630,'CRGO',553)]
for kva,tipe,loss in contoh:
    pred=rf.predict([[kva,le.transform([tipe])[0],loss]])[0]
    print(f"    {kva:<6} {tipe:<12} {loss:>10} W  {pred:>14.6f} %")

print("\n"+"="*60)
print("  SELESAI! Model siap digunakan.")
print("="*60)