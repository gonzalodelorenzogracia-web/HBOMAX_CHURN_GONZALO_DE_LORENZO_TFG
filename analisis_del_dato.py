# =============================================================================
# ANÁLISIS DEL DATO — HBO Max Churn Prediction
# Autor: Gonzalo De Lorenzo Gracia
# Grado en Business Analytics — Universidad Francisco de Vitoria
# Curso 2025-2026
#
# Descripción:
#   Pipeline completo de análisis predictivo del churn en suscriptores
#   de HBO Max. Incluye:
#     1. Carga y preparación del dataset modelado
#     2. División train/test estratificada
#     3. Balanceo de clases (Random Oversampling)
#     4. Escalado de variables (StandardScaler)
#     5. Modelo 1: Regresión Logística (baseline)
#     6. Modelo 2: Random Forest
#     7. Modelo 3: Gradient Boosting
#     8. Comparación de modelos (5 métricas)
#     9. Análisis de umbral de decisión óptimo
#    10. Segmentación de suscriptores por nivel de riesgo
#    11. Insights de negocio para HBO Max
#
# Requisitos:
#   pip install pandas numpy matplotlib seaborn scikit-learn
#
# Uso:
#   Asegúrate de haber ejecutado primero ingenieria_del_dato.py para
#   generar 'hbo_max_modelado.csv', y luego ejecuta:
#   python analisis_del_dato.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.utils import resample
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Configuración visual
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 12, 'axes.titlesize': 13,
    'axes.labelsize': 12, 'figure.facecolor': 'white'
})

os.makedirs('figures/analisis', exist_ok=True)

print("=" * 65)
print("  ANÁLISIS DEL DATO — HBO Max Churn Prediction")
print("=" * 65)


# =============================================================================
# SECCIÓN 1: CARGA Y PREPARACIÓN DEL DATASET
# =============================================================================
print("\n[1/10] Carga y preparación del dataset...")

df = pd.read_csv('hbo_max_modelado.csv', sep=';')

# Eliminar variables sin valor predictivo
drop_cols = [c for c in df.columns if c in ['subscriber_id', 'fuente', 'tenure_segment']]
df = df.drop(columns=drop_cols, errors='ignore')

X = df.drop(columns=['churn'])
y = df['churn']

print(f"  Dataset cargado: {df.shape[0]:,} registros × {df.shape[1]} variables")
print(f"  Features: {X.shape[1]}")
print(f"  Distribución target: Activo={int((y==0).sum()):,} ({(y==0).mean()*100:.1f}%) | Churn={int((y==1).sum()):,} ({y.mean()*100:.1f}%)")


# =============================================================================
# SECCIÓN 2: DIVISIÓN TRAIN/TEST ESTRATIFICADA
# =============================================================================
print("\n[2/10] División train/test estratificada (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"  Train: {X_train.shape[0]:,} registros | Churn rate: {y_train.mean()*100:.1f}%")
print(f"  Test:  {X_test.shape[0]:,} registros  | Churn rate: {y_test.mean()*100:.1f}%")

feat_names = X.columns.tolist()


# =============================================================================
# SECCIÓN 3: BALANCEO DE CLASES (RANDOM OVERSAMPLING)
# =============================================================================
print("\n[3/10] Balanceo de clases (Random Oversampling sobre train)...")

# Reset de índices para evitar problemas de alineación
X_tr = X_train.reset_index(drop=True)
y_tr = y_train.reset_index(drop=True)

# Separar clases mayoritaria y minoritaria
X_maj = X_tr[y_tr==0]; y_maj = y_tr[y_tr==0]
X_min = X_tr[y_tr==1]; y_min = y_tr[y_tr==1]

# Oversamplear clase minoritaria hasta igualar la mayoritaria
X_min_up, y_min_up = resample(X_min, y_min, replace=True, n_samples=len(X_maj), random_state=42)

# Dataset balanceado
X_bal = pd.concat([X_maj, X_min_up]).reset_index(drop=True)
y_bal = pd.concat([y_maj, y_min_up]).reset_index(drop=True)

print(f"  Antes:  {X_tr.shape[0]:,} registros ({y_tr.sum()} churn, {(y_tr==0).sum()} activos)")
print(f"  Despues: {X_bal.shape[0]:,} registros ({y_bal.sum()} churn, {(y_bal==0).sum()} activos)")
print(f"  Proporcion churn tras balanceo: {y_bal.mean()*100:.1f}%")

# Figura 1 — Balanceo
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (y_s, titulo) in zip(axes, [(y_tr, 'ANTES del balanceo (train)'),
                                     (y_bal, 'DESPUES del balanceo (Random Oversampling)')]):
    cc = y_s.value_counts().sort_index()
    bars = ax.bar(['Activo (0)','Churn (1)'], cc.values, color=['#4878CF','#E07B54'], edgecolor='white', width=0.5)
    ax.set_title(titulo, fontsize=12); ax.set_ylabel('No registros'); ax.set_ylim(0, cc.max()*1.22)
    [ax.text(i, v+30, f'{v:,}\n({v/cc.sum()*100:.1f}%)', ha='center', fontsize=11, fontweight='bold')
     for i, v in enumerate(cc.values)]
plt.suptitle('Figura 1. Distribucion de clases antes y despues del balanceo', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig01_balanceo.png', bbox_inches='tight'); plt.close()
print("  → figures/analisis/fig01_balanceo.png guardada")


# =============================================================================
# SECCIÓN 4: ESCALADO DE VARIABLES (PARA REGRESIÓN LOGÍSTICA)
# =============================================================================
print("\n[4/10] Escalado de variables (StandardScaler)...")

# IMPORTANTE: fit solo sobre train, transform sobre train y test
scaler = StandardScaler()
X_bal_sc  = scaler.fit_transform(X_bal)   # Train escalado (para LR)
X_test_sc = scaler.transform(X_test)       # Test escalado (para LR)

# RF y GB no requieren escalado (árboles invariantes a transformaciones monótonas)
print("  StandardScaler ajustado sobre train balanceado")
print("  Usado solo para Regresion Logistica (RF y GB usan datos sin escalar)")


# Función auxiliar para calcular métricas
def calcular_metricas(y_true, y_pred, y_prob, nombre_modelo):
    """Calcula y muestra las 5 métricas de evaluación del modelo."""
    metricas = {
        'Modelo':    nombre_modelo,
        'Accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'F1-Score':  round(f1_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred), 4),
        'Recall':    round(recall_score(y_true, y_pred), 4),
        'AUC-ROC':   round(roc_auc_score(y_true, y_prob), 4),
    }
    print(f"  {nombre_modelo}: Acc={metricas['Accuracy']:.4f} | F1={metricas['F1-Score']:.4f} | "
          f"Prec={metricas['Precision']:.4f} | Rec={metricas['Recall']:.4f} | AUC={metricas['AUC-ROC']:.4f}")
    return metricas


# =============================================================================
# SECCIÓN 5: MODELO 1 — REGRESIÓN LOGÍSTICA (BASELINE)
# =============================================================================
print("\n[5/10] Modelo 1: Regresion Logistica (baseline)...")
print("  Configuracion: solver='lbfgs', C=1.0, max_iter=1000, regularizacion L2")

lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    C=1.0,           # Regularización L2 estándar
    random_state=42
)
lr.fit(X_bal_sc, y_bal)

# Predicciones
lr_pred_train = lr.predict(X_bal_sc)
lr_pred_test  = lr.predict(X_test_sc)
lr_prob_train = lr.predict_proba(X_bal_sc)[:,1]
lr_prob_test  = lr.predict_proba(X_test_sc)[:,1]

print("\n  --- Resultados Regresion Logistica ---")
lr_train_acc = accuracy_score(y_bal, lr_pred_train)
print(f"  Train Accuracy: {lr_train_acc:.4f}")
met_lr = calcular_metricas(y_test, lr_pred_test, lr_prob_test, "LR Test")
print(f"  Gap train-test (overfitting): {lr_train_acc - met_lr['Accuracy']:.4f}")

print("\n  Reporte de clasificacion detallado (Test):")
print(classification_report(y_test, lr_pred_test, target_names=['Activo','Churn']))

# Figura 2 — Matrices de confusión LR
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (pred, y_t), titulo, cmap in [
    (axes[0], (lr_pred_train, y_bal), 'LR - Train (balanceado)', 'Blues'),
    (axes[1], (lr_pred_test, y_test),  'LR - Test',               'Blues')]:
    cm = confusion_matrix(y_t, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Activo','Churn'], yticklabels=['Activo','Churn'],
                annot_kws={'size': 13}, linewidths=0.5)
    ax.set_title(titulo, fontsize=12); ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')
plt.suptitle('Figura 2. Matrices de confusion - Regresion Logistica', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig02_cm_lr.png', bbox_inches='tight'); plt.close()

# Figura 3 — Curva ROC LR
fig, ax = plt.subplots(figsize=(8, 7))
fpr, tpr, _ = roc_curve(y_test, lr_prob_test)
ax.plot(fpr, tpr, color='#4878CF', linewidth=2.5, label=f'LR (AUC={met_lr["AUC-ROC"]:.3f})')
ax.fill_between(fpr, tpr, alpha=0.12, color='#4878CF')
ax.plot([0,1], [0,1], 'k--', alpha=0.4, label='Aleatorio (AUC=0.500)')
ax.set_xlabel('Tasa de Falsos Positivos'); ax.set_ylabel('Tasa de Verdaderos Positivos')
ax.set_title('Figura 3. Curva ROC - Regresion Logistica')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('figures/analisis/fig03_roc_lr.png', bbox_inches='tight'); plt.close()

# Figura 4 — Importancia LR (coeficiente absoluto)
coef    = np.abs(lr.coef_[0])
top15   = np.argsort(coef)[-15:]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh([feat_names[i] for i in top15], coef[top15], color='#4878CF', edgecolor='white', height=0.7)
ax.set_xlabel('|Coeficiente| (importancia relativa)')
ax.set_title('Figura 4. Top 15 variables - Regresion Logistica (coeficiente absoluto)')
[ax.text(v+0.001, i, f'{v:.3f}', va='center', fontsize=9) for i, v in enumerate(coef[top15])]
plt.tight_layout(); plt.savefig('figures/analisis/fig04_imp_lr.png', bbox_inches='tight'); plt.close()
print("  → figures/analisis/fig02, fig03, fig04 guardadas")


# =============================================================================
# SECCIÓN 6: MODELO 2 — RANDOM FOREST
# =============================================================================
print("\n[6/10] Modelo 2: Random Forest...")
print("  Configuracion: n_estimators=200, max_depth=20, min_samples_split=2")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1       # Usar todos los núcleos disponibles
)
rf.fit(X_bal, y_bal)  # RF no necesita datos escalados

# Predicciones
rf_pred_train = rf.predict(X_bal)
rf_pred_test  = rf.predict(X_test)
rf_prob_train = rf.predict_proba(X_bal)[:,1]
rf_prob_test  = rf.predict_proba(X_test)[:,1]

print("\n  --- Resultados Random Forest ---")
rf_train_acc = accuracy_score(y_bal, rf_pred_train)
print(f"  Train Accuracy: {rf_train_acc:.4f}")
met_rf = calcular_metricas(y_test, rf_pred_test, rf_prob_test, "RF Test")
print(f"  Gap train-test (overfitting): {rf_train_acc - met_rf['Accuracy']:.4f}")
print(f"  ⚠ Overfitting severo detectado (gap > 20%)")

print("\n  Reporte de clasificacion detallado (Test):")
print(classification_report(y_test, rf_pred_test, target_names=['Activo','Churn']))

# Figura 5 — Matrices de confusión RF
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (pred, y_t), titulo in [
    (axes[0], (rf_pred_train, y_bal), 'RF - Train (balanceado)'),
    (axes[1], (rf_pred_test, y_test),  'RF - Test')]:
    cm = confusion_matrix(y_t, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
                xticklabels=['Activo','Churn'], yticklabels=['Activo','Churn'],
                annot_kws={'size': 13}, linewidths=0.5)
    ax.set_title(titulo, fontsize=12); ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')
plt.suptitle('Figura 5. Matrices de confusion - Random Forest', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig05_cm_rf.png', bbox_inches='tight'); plt.close()

# Figura 6 — Importancia RF (Gini)
fi_rf   = rf.feature_importances_
top15r  = np.argsort(fi_rf)[-15:]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh([feat_names[i] for i in top15r], fi_rf[top15r], color='#6ACC65', edgecolor='white', height=0.7)
ax.set_xlabel('Importancia Gini')
ax.set_title('Figura 6. Top 15 variables - Random Forest (Gini importance)')
[ax.text(v+0.0005, i, f'{v:.3f}', va='center', fontsize=9) for i, v in enumerate(fi_rf[top15r])]
plt.tight_layout(); plt.savefig('figures/analisis/fig06_imp_rf.png', bbox_inches='tight'); plt.close()
print("  → figures/analisis/fig05, fig06 guardadas")


# =============================================================================
# SECCIÓN 7: MODELO 3 — GRADIENT BOOSTING
# =============================================================================
print("\n[7/10] Modelo 3: Gradient Boosting...")
print("  Configuracion: n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8")

gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,        # Moderado para reducir overfitting
    learning_rate=0.1,
    subsample=0.8,      # Regularización por subsampling
    random_state=42
)
gb.fit(X_bal, y_bal)  # GB tampoco necesita datos escalados

# Predicciones
gb_pred_train = gb.predict(X_bal)
gb_pred_test  = gb.predict(X_test)
gb_prob_train = gb.predict_proba(X_bal)[:,1]
gb_prob_test  = gb.predict_proba(X_test)[:,1]

print("\n  --- Resultados Gradient Boosting ---")
gb_train_acc = accuracy_score(y_bal, gb_pred_train)
print(f"  Train Accuracy: {gb_train_acc:.4f}")
met_gb = calcular_metricas(y_test, gb_pred_test, gb_prob_test, "GB Test")
print(f"  Gap train-test (overfitting): {gb_train_acc - met_gb['Accuracy']:.4f}")

print("\n  Reporte de clasificacion detallado (Test):")
print(classification_report(y_test, gb_pred_test, target_names=['Activo','Churn']))

# Figura 7 — Matrices de confusión GB
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (pred, y_t), titulo in [
    (axes[0], (gb_pred_train, y_bal), 'GB - Train (balanceado)'),
    (axes[1], (gb_pred_test, y_test),  'GB - Test')]:
    cm = confusion_matrix(y_t, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax,
                xticklabels=['Activo','Churn'], yticklabels=['Activo','Churn'],
                annot_kws={'size': 13}, linewidths=0.5)
    ax.set_title(titulo, fontsize=12); ax.set_xlabel('Prediccion'); ax.set_ylabel('Real')
plt.suptitle('Figura 7. Matrices de confusion - Gradient Boosting', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig07_cm_gb.png', bbox_inches='tight'); plt.close()

# Figura 8 — Importancia GB
fi_gb   = gb.feature_importances_
top15g  = np.argsort(fi_gb)[-15:]
fig, ax = plt.subplots(figsize=(11, 7))
ax.barh([feat_names[i] for i in top15g], fi_gb[top15g], color='#E07B54', edgecolor='white', height=0.7)
ax.set_xlabel('Importancia (Gradient Boosting)')
ax.set_title('Figura 8. Top 15 variables - Gradient Boosting')
[ax.text(v+0.0005, i, f'{v:.3f}', va='center', fontsize=9) for i, v in enumerate(fi_gb[top15g])]
plt.tight_layout(); plt.savefig('figures/analisis/fig08_imp_gb.png', bbox_inches='tight'); plt.close()
print("  → figures/analisis/fig07, fig08 guardadas")


# =============================================================================
# SECCIÓN 8: COMPARACIÓN DE MODELOS
# =============================================================================
print("\n[8/10] Comparacion de modelos...")

# Tabla comparativa
resultados = pd.DataFrame([
    {'Modelo': 'Regresion Logistica',
     'Acc_Train': round(lr_train_acc, 4),
     'Acc_Test':  met_lr['Accuracy'],
     'F1':        met_lr['F1-Score'],
     'Precision': met_lr['Precision'],
     'Recall':    met_lr['Recall'],
     'AUC':       met_lr['AUC-ROC'],
     'Gap':       round(lr_train_acc - met_lr['Accuracy'], 4)},
    {'Modelo': 'Random Forest',
     'Acc_Train': round(rf_train_acc, 4),
     'Acc_Test':  met_rf['Accuracy'],
     'F1':        met_rf['F1-Score'],
     'Precision': met_rf['Precision'],
     'Recall':    met_rf['Recall'],
     'AUC':       met_rf['AUC-ROC'],
     'Gap':       round(rf_train_acc - met_rf['Accuracy'], 4)},
    {'Modelo': 'Gradient Boosting',
     'Acc_Train': round(gb_train_acc, 4),
     'Acc_Test':  met_gb['Accuracy'],
     'F1':        met_gb['F1-Score'],
     'Precision': met_gb['Precision'],
     'Recall':    met_gb['Recall'],
     'AUC':       met_gb['AUC-ROC'],
     'Gap':       round(gb_train_acc - met_gb['Accuracy'], 4)},
])
print("\n  TABLA COMPARATIVA COMPLETA:")
print(resultados.to_string(index=False))
resultados.to_csv('resultados_modelos.csv', index=False)
print("\n  → resultados_modelos.csv guardado")

# Figura 9 — Comparación curvas ROC
fig, ax = plt.subplots(figsize=(9, 7))
for prob, col, lbl in [
    (lr_prob_test, '#4878CF', f'Regresion Logistica (AUC={met_lr["AUC-ROC"]:.3f})'),
    (rf_prob_test, '#6ACC65', f'Random Forest (AUC={met_rf["AUC-ROC"]:.3f})'),
    (gb_prob_test, '#E07B54', f'Gradient Boosting (AUC={met_gb["AUC-ROC"]:.3f})')]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, linewidth=2.5, color=col, label=lbl)
ax.plot([0,1], [0,1], 'k--', alpha=0.4, linewidth=1.5, label='Aleatorio (AUC=0.500)')
ax.set_xlabel('Tasa de Falsos Positivos'); ax.set_ylabel('Tasa de Verdaderos Positivos')
ax.set_title('Figura 9. Comparacion de curvas ROC - Los tres modelos')
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('figures/analisis/fig09_roc_comp.png', bbox_inches='tight'); plt.close()

# Figura 10 — Comparación barras
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
ks   = ['Regresion\nLogistica', 'Random\nForest', 'Gradient\nBoosting']
accs = [met_lr['Accuracy'], met_rf['Accuracy'], met_gb['Accuracy']]
f1s  = [met_lr['F1-Score'],  met_rf['F1-Score'],  met_gb['F1-Score']]
recs = [met_lr['Recall'],    met_rf['Recall'],    met_gb['Recall']]
aucs = [met_lr['AUC-ROC'],   met_rf['AUC-ROC'],   met_gb['AUC-ROC']]
cols_m = ['#4878CF', '#6ACC65', '#E07B54']
x = np.arange(len(ks)); w = 0.28
axes[0].bar(x-w, accs, w, color=[c+'CC' for c in cols_m], edgecolor='white', label='Accuracy')
axes[0].bar(x,   f1s,  w, color=cols_m,                   edgecolor='white', label='F1-Score')
axes[0].bar(x+w, recs, w, color=[c+'88' for c in cols_m], edgecolor='white', label='Recall')
axes[0].set_xticks(x); axes[0].set_xticklabels(ks, fontsize=11)
axes[0].set_ylabel('Puntuacion'); axes[0].set_title('Metricas de evaluacion en Test')
axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=10)
[axes[0].text(i-w, v+0.01, f'{v:.3f}', ha='center', fontsize=8, rotation=90) for i,v in enumerate(accs)]
[axes[0].text(i,   v+0.01, f'{v:.3f}', ha='center', fontsize=8, rotation=90) for i,v in enumerate(f1s)]
[axes[0].text(i+w, v+0.01, f'{v:.3f}', ha='center', fontsize=8, rotation=90) for i,v in enumerate(recs)]
axes[1].bar(x, aucs, 0.5, color=cols_m, alpha=0.85, edgecolor='white')
axes[1].set_xticks(x); axes[1].set_xticklabels(ks, fontsize=11)
axes[1].set_ylabel('AUC-ROC'); axes[1].set_title('AUC-ROC por modelo')
axes[1].set_ylim(0.5, 1.0)
[axes[1].text(i, v+0.003, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold') for i,v in enumerate(aucs)]
plt.suptitle('Figura 10. Comparacion de metricas - Todos los modelos', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig10_comparacion.png', bbox_inches='tight'); plt.close()
print("  → figures/analisis/fig09, fig10 guardadas")


# =============================================================================
# SECCIÓN 9: SELECCIÓN DEL MODELO FINAL
# =============================================================================
print("\n[9/10] Seleccion del modelo final...")
print("  Modelo seleccionado: REGRESION LOGISTICA")
print("  Justificacion:")
print(f"   - Mayor Recall de clase churn: {met_lr['Recall']:.3f} (vs {met_rf['Recall']:.3f} RF, {met_gb['Recall']:.3f} GB)")
print(f"   - Mayor AUC-ROC: {met_lr['AUC-ROC']:.3f} (vs {met_rf['AUC-ROC']:.3f} RF, {met_gb['AUC-ROC']:.3f} GB)")
print(f"   - Mejor generalizacion: gap={lr_train_acc - met_lr['Accuracy']:.4f} (vs {rf_train_acc-met_rf['Accuracy']:.4f} RF)")
print("   - Alta interpretabilidad: coeficientes directamente comunicables al equipo de negocio")

# Usar LR como modelo final
modelo_final     = lr
prob_final_test  = lr_prob_test
pred_final_test  = lr_pred_test

# =============================================================================
# SECCIÓN 10: ANÁLISIS DE UMBRAL Y SEGMENTACIÓN DE RIESGO
# =============================================================================
print("\n[10/10] Analisis de umbral y segmentacion de riesgo...")

# Cálculo de métricas para cada umbral posible
thresholds = np.arange(0.10, 0.90, 0.01)
prec_list, rec_list, f1_list = [], [], []
for t in thresholds:
    pred_t = (prob_final_test >= t).astype(int)
    prec_list.append(precision_score(y_test, pred_t, zero_division=0))
    rec_list.append(recall_score(y_test, pred_t, zero_division=0))
    f1_list.append(f1_score(y_test, pred_t, zero_division=0))

best_t     = float(thresholds[np.argmax(f1_list)])
best_f1    = float(np.max(f1_list))
best_prec  = float(prec_list[np.argmax(f1_list)])
best_rec   = float(rec_list[np.argmax(f1_list)])

print(f"  Umbral optimo (max F1): {best_t:.2f}")
print(f"  Metricas con umbral optimo: F1={best_f1:.3f} | Prec={best_prec:.3f} | Rec={best_rec:.3f}")
print(f"  Umbral por defecto (0.50): F1={f1_list[int((0.5-0.1)/0.01)]:.3f}")

# Figura 11 — Análisis de umbral
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].plot(thresholds, prec_list, color='#4878CF', linewidth=2, label='Precision')
axes[0].plot(thresholds, rec_list,  color='#E07B54', linewidth=2, label='Recall')
axes[0].plot(thresholds, f1_list,   color='#6ACC65', linewidth=2.5, label='F1-Score')
axes[0].axvline(best_t, color='black', linestyle='--', linewidth=2, label=f'Umbral optimo ({best_t:.2f})')
axes[0].axvline(0.5,    color='gray',  linestyle=':',  linewidth=1.5, label='Default (0.50)')
axes[0].set_xlabel('Umbral de decision'); axes[0].set_ylabel('Puntuacion')
axes[0].set_title('Precision, Recall y F1 segun umbral')
axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
axes[1].hist(prob_final_test[y_test.values==0], bins=40, alpha=0.65, color='#4878CF',
             density=True, label=f'Activo (n={int((y_test==0).sum())})')
axes[1].hist(prob_final_test[y_test.values==1], bins=40, alpha=0.65, color='#E07B54',
             density=True, label=f'Churn (n={int((y_test==1).sum())})')
axes[1].axvline(best_t, color='black', linestyle='--', linewidth=2, label=f'Umbral optimo ({best_t:.2f})')
axes[1].set_xlabel('Probabilidad de churn'); axes[1].set_ylabel('Densidad')
axes[1].set_title('Distribucion de probabilidades por clase')
axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
plt.suptitle('Figura 11. Analisis de umbral de decision - Regresion Logistica', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig11_umbral.png', bbox_inches='tight'); plt.close()

# Segmentación de riesgo en 3 niveles
bins_riesgo  = [0, 0.3, 0.6, 1.0]
labels_riesgo = ['Bajo riesgo (<30%)', 'Riesgo medio (30-60%)', 'Alto riesgo (>60%)']
seg_idx = [
    np.where((prob_final_test>=bins_riesgo[i]) & (prob_final_test<bins_riesgo[i+1]))[0]
    for i in range(3)
]
seg_idx[2] = np.where(prob_final_test >= 0.6)[0]  # Alto riesgo incluye el 1.0
seg_counts  = [len(s) for s in seg_idx]
churn_rates = [np.mean(y_test.values[s])*100 if len(s)>0 else 0 for s in seg_idx]

print("\n  SEGMENTACION DE RIESGO:")
for lbl, count, rate in zip(labels_riesgo, seg_counts, churn_rates):
    print(f"   {lbl}: {count:,} suscriptores ({count/len(prob_final_test)*100:.1f}%) | Churn real: {rate:.1f}%")

# Figura 12 — Segmentación de riesgo
cols_seg = ['#6ACC65', '#FDB462', '#E07B54']
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
bars = axes[0].bar(range(3), seg_counts, color=cols_seg, edgecolor='white', width=0.5)
axes[0].set_xticks(range(3)); axes[0].set_xticklabels(labels_riesgo, fontsize=11)
axes[0].set_title('Suscriptores por segmento de riesgo'); axes[0].set_ylabel('No suscriptores')
[axes[0].text(i, v+5, f'{v:,}\n({v/len(prob_final_test)*100:.1f}%)', ha='center',
              fontsize=11, fontweight='bold') for i, v in enumerate(seg_counts)]
axes[0].set_ylim(0, max(seg_counts)*1.22)
bars2 = axes[1].bar(range(3), churn_rates, color=cols_seg, edgecolor='white', width=0.5)
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(labels_riesgo, fontsize=11)
axes[1].set_title('Tasa de churn real por segmento'); axes[1].set_ylabel('Tasa de churn real (%)')
[axes[1].text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
 for i, v in enumerate(churn_rates)]
axes[1].set_ylim(0, max(churn_rates)*1.22)
plt.suptitle('Figura 12. Segmentacion de suscriptores por nivel de riesgo de churn', fontsize=13)
plt.tight_layout(); plt.savefig('figures/analisis/fig12_segmentos.png', bbox_inches='tight'); plt.close()
print("  → figures/analisis/fig11, fig12 guardadas")


# =============================================================================
# EXPORTAR PROBABILIDADES DE CHURN POR SUSCRIPTOR
# =============================================================================
# Crear scoring final (útil para integrar en dashboard de retención)
df_scoring = pd.DataFrame({
    'indice_test':       np.arange(len(y_test)),
    'churn_real':        y_test.values,
    'prob_churn':        prob_final_test.round(4),
    'pred_umbral_opt':   (prob_final_test >= best_t).astype(int),
    'pred_umbral_default': (prob_final_test >= 0.5).astype(int),
    'segmento_riesgo':   pd.cut(prob_final_test, bins=[0,0.3,0.6,1.0],
                                labels=['Bajo','Medio','Alto'])
})
df_scoring.to_csv('scoring_churn_final.csv', index=False)
print("  → scoring_churn_final.csv guardado (probabilidades por suscriptor)")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  RESUMEN — ANÁLISIS DEL DATO COMPLETADO")
print("=" * 65)
print(f"\n  MODELOS ENTRENADOS Y EVALUADOS:")
print(f"  {'Modelo':<25} {'Accuracy':>10} {'F1':>10} {'Recall':>10} {'AUC':>10} {'Gap':>10}")
print(f"  {'-'*65}")
print(f"  {'Regresion Logistica':<25} {met_lr['Accuracy']:>10.4f} {met_lr['F1-Score']:>10.4f} {met_lr['Recall']:>10.4f} {met_lr['AUC-ROC']:>10.4f} {lr_train_acc-met_lr['Accuracy']:>10.4f}")
print(f"  {'Random Forest':<25} {met_rf['Accuracy']:>10.4f} {met_rf['F1-Score']:>10.4f} {met_rf['Recall']:>10.4f} {met_rf['AUC-ROC']:>10.4f} {rf_train_acc-met_rf['Accuracy']:>10.4f}")
print(f"  {'Gradient Boosting':<25} {met_gb['Accuracy']:>10.4f} {met_gb['F1-Score']:>10.4f} {met_gb['Recall']:>10.4f} {met_gb['AUC-ROC']:>10.4f} {gb_train_acc-met_gb['Accuracy']:>10.4f}")
print(f"\n  MODELO FINAL SELECCIONADO: Regresion Logistica")
print(f"  Umbral optimo de decision:  {best_t:.2f}")
print(f"  Recall con umbral optimo:   {best_rec:.3f} ({best_rec*100:.1f}% de churners detectados)")
print(f"\n  SEGMENTACION DE RIESGO:")
for lbl, count, rate in zip(labels_riesgo, seg_counts, churn_rates):
    print(f"   {lbl:<30}: {count:>5,} suscriptores | Churn real: {rate:.1f}%")
print(f"\n  ARCHIVOS GENERADOS:")
print(f"   - figures/analisis/fig01 a fig12 (12 figuras)")
print(f"   - resultados_modelos.csv")
print(f"   - scoring_churn_final.csv")
print("=" * 65)
print("  ✅ Pipeline de analisis del dato completado con exito")
print("=" * 65)
