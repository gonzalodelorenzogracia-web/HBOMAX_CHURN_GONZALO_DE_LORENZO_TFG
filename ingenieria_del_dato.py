# =============================================================================
# INGENIERÍA DEL DATO — HBO Max Churn Prediction
# Autor: Gonzalo De Lorenzo Gracia
# Grado en Business Analytics — Universidad Francisco de Vitoria
# Curso 2025-2026
#
# Descripción:
#   Pipeline completo de ingeniería del dato para el análisis de churn
#   en suscriptores de HBO Max. Incluye:
#     1. Carga e integración de fuentes de datos
#     2. Profiling inicial (análisis de calidad)
#     3. Limpieza y tratamiento de valores nulos
#     4. Tratamiento de outliers (winsorización)
#     5. Feature Engineering (variables derivadas)
#     6. Codificación de variables categóricas
#     7. Análisis Exploratorio de Datos (EDA) con visualizaciones
#     8. Exportación de datasets finales
#
# Requisitos:
#   pip install pandas numpy matplotlib seaborn scikit-learn
#
# Uso:
#   Coloca el archivo 'hbo_max_churn_dataset.csv' en el mismo directorio
#   y ejecuta: python ingenieria_del_dato.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# Configuración visual global
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.facecolor': 'white'
})

# Crear carpeta de figuras si no existe
os.makedirs('figures', exist_ok=True)

print("=" * 65)
print("  INGENIERÍA DEL DATO — HBO Max Churn Prediction")
print("=" * 65)


# =============================================================================
# SECCIÓN 1: CARGA E INTEGRACIÓN DE FUENTES DE DATOS
# =============================================================================
print("\n[1/8] Carga e integración de fuentes de datos...")

# ── Dataset original (5.000 registros, 11 variables) ─────────────────────────
# Generado con Python a partir de parámetros extraídos de la web de
# HBO Max España y estadísticas públicas del sector OTT (ver generar_dataset.py)
df_orig = pd.read_csv('hbo_max_churn_dataset.csv', sep=';', encoding='latin1')
print(f"  Dataset original cargado: {df_orig.shape}")

# ── Dataset enriquecido (5.000 registros adicionales, 9 variables nuevas) ─────
# Elaborado manualmente con parámetros de informes OTT públicos:
# Reuters Institute Digital Report, Statista, CNMC
np.random.seed(42)
n_new = 5000

tenure       = np.random.randint(1, 37, n_new)
plan_type    = np.random.choice(
    ['Mensual (14.99)', 'Anual (12.99/mes)', 'Familia (19.99/mes)'],
    n_new, p=[0.644, 0.213, 0.143])
payment_method = np.random.choice(
    ['Tarjeta', 'PayPal', 'Apple Pay/Google Pay'],
    n_new, p=[0.547, 0.253, 0.200])
base_hours   = np.random.normal(20, 8, n_new).clip(0.5, 60)
hbo_share    = np.random.beta(2, 3, n_new).round(3)
support_tick = np.random.poisson(0.8, n_new).clip(0, 8)
sat_raw = (base_hours/60)*3 + (1-hbo_share) + np.random.normal(0,0.5,n_new) - support_tick*0.3
satisfaction = np.clip(np.round(sat_raw*1.5+1.5).astype(int), 1, 5)
fee_map = {'Mensual (14.99)': 14.99, 'Anual (12.99/mes)': 12.99, 'Familia (19.99/mes)': 19.99}
monthly_fee = np.array([fee_map[p] for p in plan_type]) + np.random.normal(0,0.8,n_new)
monthly_fee = monthly_fee.round(2).clip(10, 25)
total_revenue = (monthly_fee * tenure).round(2)

# Variables de comportamiento de sesión (fuente enriquecida)
device       = np.random.choice(['Smart TV','Móvil','Ordenador','Tablet'], n_new, p=[0.40,0.30,0.20,0.10])
profiles     = np.random.choice([1,2,3,4,5], n_new, p=[0.35,0.30,0.20,0.10,0.05])
genre        = np.random.choice(['Drama','Acción','Comedia','Documental','Thriller','Familiar'],
                                 n_new, p=[0.28,0.22,0.18,0.12,0.12,0.08])
sessions_pw  = np.random.gamma(2.5, 1.5, n_new).clip(0.5, 14).round(1)
avg_session  = np.random.normal(45, 15, n_new).clip(5, 120).round(1)
used_trial   = np.random.choice([0,1], n_new, p=[0.35,0.65])
has_discount = np.random.choice([0,1], n_new, p=[0.70,0.30])
months_paused= np.random.choice([0,1,2,3], n_new, p=[0.75,0.15,0.07,0.03])
region       = np.random.choice(['España','México','Argentina','Colombia','Chile','Otros LATAM'],
                                 n_new, p=[0.35,0.25,0.15,0.10,0.08,0.07])

# Función de probabilidad de churn calibrada
churn_prob = (
    0.35 - 0.008*tenure - 0.004*base_hours - 0.05*hbo_share
    + 0.06*support_tick - 0.04*satisfaction
    + 0.10*(plan_type == 'Mensual (14.99)').astype(int)
    - 0.05*(plan_type == 'Anual (12.99/mes)').astype(int)
    + 0.03*months_paused - 0.02*sessions_pw
    + np.random.normal(0, 0.05, n_new)
).clip(0.02, 0.95)
churn_enrich = (np.random.rand(n_new) < churn_prob).astype(int)

df_new = pd.DataFrame({
    'subscriber_id': np.arange(5001, 5001+n_new),
    'tenure_months': tenure,
    'plan_type': plan_type,
    'payment_method': payment_method,
    'total_watch_hours': base_hours.round(1),
    'hbo_original_share': hbo_share,
    'support_tickets': support_tick,
    'satisfaction_score': satisfaction,
    'monthly_fee': monthly_fee,
    'total_revenue': total_revenue,
    'device_type': device,
    'active_profiles': profiles,
    'main_genre': genre,
    'sessions_per_week': sessions_pw,
    'avg_session_min': avg_session,
    'used_trial': used_trial,
    'has_discount': has_discount,
    'months_paused': months_paused,
    'region': region,
    'churn': churn_enrich
})

# Añadir columnas nuevas vacías al dataset original (NaN estructurales)
new_cols = ['device_type','active_profiles','main_genre','sessions_per_week',
            'avg_session_min','used_trial','has_discount','months_paused','region']
for col in new_cols:
    df_orig[col] = np.nan

# Integración (concatenación vertical)
df = pd.concat([df_orig, df_new], ignore_index=True)
df['subscriber_id'] = np.arange(1, len(df)+1)
df_before = df.copy()  # Guardar copia para comparaciones

print(f"  Dataset combinado: {df.shape[0]:,} registros × {df.shape[1]} variables")
print(f"  Tasa de churn global: {df['churn'].mean()*100:.1f}%")

# Guardar dataset combinado
df.to_csv('hbo_max_dataset_combinado.csv', index=False, sep=';')
print("  → hbo_max_dataset_combinado.csv guardado")


# =============================================================================
# SECCIÓN 2: PROFILING INICIAL
# =============================================================================
print("\n[2/8] Profiling inicial...")

print("\n  ESTADÍSTICOS DESCRIPTIVOS (variables numéricas):")
num_cols_desc = ['tenure_months','total_watch_hours','hbo_original_share',
                 'support_tickets','satisfaction_score','monthly_fee','total_revenue']
print(df[num_cols_desc].describe().round(3).to_string())

print("\n  ANÁLISIS DE VALORES NULOS:")
nulos = df.isnull().sum()
nulos_pct = (nulos / len(df) * 100).round(2)
nulos_df = pd.DataFrame({'Nulos': nulos[nulos>0], '% Nulos': nulos_pct[nulos>0]})
print(nulos_df.to_string())
print(f"\n  Total nulos: {df.isnull().sum().sum():,}")

print("\n  ANÁLISIS DE DUPLICADOS:")
print(f"  Filas duplicadas: {df.duplicated().sum()}")
print(f"  IDs duplicados: {df['subscriber_id'].duplicated().sum()}")

# Figura 1 — Nulos
fig, ax = plt.subplots(figsize=(13, 6))
cols_n = df_before.columns[df_before.isnull().any()]
pct = (df_before[cols_n].isnull().sum() / len(df_before) * 100).sort_values()
bars = ax.barh(pct.index, pct.values, color='#E07B54', edgecolor='white', height=0.6)
[ax.text(v+0.5, b.get_y()+b.get_height()/2, f'{v:.0f}%', va='center', fontsize=12, fontweight='bold')
 for b, v in zip(bars, pct.values)]
ax.axvline(50, color='#336699', linestyle='--', linewidth=1.5, label='50% — umbral estructural')
ax.set_xlabel('% valores nulos'); ax.set_xlim(0, 63)
ax.set_title('Figura 1. Distribución de valores nulos por variable', pad=12)
ax.legend(fontsize=11)
plt.tight_layout(); plt.savefig('figures/fig01_nulos.png', bbox_inches='tight'); plt.close()
print("  → figures/fig01_nulos.png guardada")

# Figura 2 — Distribución churn
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cc = df['churn'].value_counts()
axes[0].bar(['Activo (0)', 'Churn (1)'], cc.values, color=['#4878CF','#E07B54'], edgecolor='white', width=0.5)
axes[0].set_title('Distribución global de la variable churn')
axes[0].set_ylabel('Nº suscriptores'); axes[0].set_ylim(0, cc.max()*1.22)
[axes[0].text(i, v+80, f'{v:,}\n({v/len(df)*100:.1f}%)', ha='center', fontsize=12, fontweight='bold')
 for i, v in enumerate(cc.values)]
src = np.where(df['subscriber_id']<=5000, 'Original\n(5.000 reg.)', 'Enriquecido\n(5.000 reg.)')
cs = pd.Series(df['churn'].values); cs.index = src
cr = cs.groupby(level=0).mean() * 100
axes[1].bar(cr.index, cr.values, color=['#4878CF','#6ACC65'], edgecolor='white', width=0.45)
axes[1].set_title('Tasa de churn por fuente de datos')
axes[1].set_ylabel('Tasa de churn (%)'); axes[1].set_ylim(0, 40)
[axes[1].text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=13, fontweight='bold')
 for i, v in enumerate(cr.values)]
axes[1].axhline(df['churn'].mean()*100, color='red', linestyle='--', linewidth=1.5,
                label=f'Media global ({df["churn"].mean()*100:.1f}%)')
axes[1].legend()
plt.suptitle('Figura 2. Variable objetivo: distribución y comparativa por fuente', y=1.01)
plt.tight_layout(); plt.savefig('figures/fig02_churn_dist.png', bbox_inches='tight'); plt.close()
print("  → figures/fig02_churn_dist.png guardada")


# =============================================================================
# SECCIÓN 3: LIMPIEZA Y TRATAMIENTO DE VALORES NULOS
# =============================================================================
print("\n[3/8] Limpieza y tratamiento de valores nulos...")

# 3.1 Imputación por MODA (variables categóricas nominales)
for col in ['device_type', 'main_genre']:
    moda = df[col].mode()[0]
    df[col] = df[col].fillna(moda)
    print(f"  '{col}' → imputado con moda: '{moda}'")

# 3.2 Imputación por MEDIANA (variables discretas asimétricas)
for col in ['active_profiles', 'months_paused']:
    mediana = df[col].median()
    df[col] = df[col].fillna(mediana)
    print(f"  '{col}' → imputado con mediana: {mediana}")

# 3.3 Imputación por INTERPOLACIÓN LINEAL (sessions_per_week)
# Se ordena por tenure_months para aprovechar la relación temporal
df_sorted = df.sort_values('tenure_months')
df_sorted['sessions_per_week'] = df_sorted['sessions_per_week'].interpolate(method='linear')
df['sessions_per_week'] = df_sorted['sessions_per_week'].values
print(f"  'sessions_per_week' → imputado por interpolación lineal (nulos restantes: {df['sessions_per_week'].isnull().sum()})")

# 3.4 Imputación con MEDIA + RUIDO GAUSSIANO (avg_session_min)
# Preserva la distribución sin generar valores idénticos
media_s = df['avg_session_min'].mean()
std_s   = df['avg_session_min'].std()
n_nulos = df['avg_session_min'].isnull().sum()
df.loc[df['avg_session_min'].isnull(), 'avg_session_min'] = \
    np.random.normal(media_s, std_s*0.3, n_nulos).clip(5, 120).round(1)
print(f"  'avg_session_min' → imputado con media ({media_s:.1f} min) + ruido gaussiano (σ=0.3×σ_orig)")

# 3.5 Imputación por DISTRIBUCIÓN PROPORCIONAL (variables binarias y región)
for col in ['used_trial', 'has_discount']:
    dist = df[col].value_counts(normalize=True)
    n    = df[col].isnull().sum()
    df.loc[df[col].isnull(), col] = np.random.choice(dist.index, size=n, p=dist.values)
    print(f"  '{col}' → imputado por distribución proporcional: {dist.to_dict()}")

dist_r = df['region'].value_counts(normalize=True)
n_r    = df['region'].isnull().sum()
df.loc[df['region'].isnull(), 'region'] = np.random.choice(dist_r.index, size=n_r, p=dist_r.values)
print(f"  'region' → imputado por distribución proporcional ({len(dist_r)} categorías)")

# Conversión de tipos tras imputación
for col in ['active_profiles', 'months_paused', 'used_trial', 'has_discount']:
    df[col] = df[col].astype(int)

print(f"\n  Nulos restantes tras limpieza: {df.isnull().sum().sum()}")
print("  ✓ Dataset completamente limpio")

# Figura 3 — Antes/Después de la imputación
vars_imp  = ['sessions_per_week', 'avg_session_min', 'active_profiles', 'months_paused']
nombres3  = ['Sesiones/semana', 'Duración sesión (min)', 'Perfiles activos', 'Meses pausado']
colores3  = ['#4878CF', '#E07B54', '#6ACC65', '#9370DB']
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, (var, nom, col) in enumerate(zip(vars_imp, nombres3, colores3)):
    da = df_before[var].dropna()
    dd = df[var]
    for ax, datos, title in [(axes[0,i], da, f'ANTES\n(n={len(da):,})'),
                              (axes[1,i], dd, f'DESPUÉS\n(n={len(dd):,})')]:
        ax.hist(datos, bins=30, color=col, alpha=0.85, edgecolor='white')
        ax.set_title(f'{nom}\n{title}', fontsize=10)
        ax.axvline(datos.mean(), color='red', linestyle='--', linewidth=1.8, label=f'μ={datos.mean():.1f}')
        ax.legend(fontsize=9)
        ax.set_ylabel('Frecuencia' if i == 0 else '')
fig.suptitle('Figura 3. Distribuciones ANTES y DESPUÉS de la imputación', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig('figures/fig03_antes_despues.png', bbox_inches='tight'); plt.close()
print("  → figures/fig03_antes_despues.png guardada")


# =============================================================================
# SECCIÓN 4: TRATAMIENTO DE OUTLIERS
# =============================================================================
print("\n[4/8] Tratamiento de outliers...")

num_vars = ['total_watch_hours', 'hbo_original_share', 'support_tickets',
            'monthly_fee', 'total_revenue', 'sessions_per_week', 'avg_session_min']

# Figura 4 — Boxplots ANTES
fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.flatten()
for i, var in enumerate(num_vars):
    q1, q3 = df[var].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[var] < q1-1.5*iqr) | (df[var] > q3+1.5*iqr)).sum()
    axes[i].boxplot(df[var].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#AEC6CF', color='#336699'),
                    medianprops=dict(color='#E07B54', linewidth=2.5),
                    whiskerprops=dict(color='#336699'), capprops=dict(color='#336699'),
                    flierprops=dict(marker='o', markerfacecolor='#E07B54', markersize=4, alpha=0.5))
    axes[i].set_title(var, fontsize=11)
    axes[i].set_xlabel(f'Outliers: {n_out}', fontsize=9, color='#E07B54')
axes[-1].set_visible(False)
fig.suptitle('Figura 4. Boxplots ANTES del tratamiento de outliers', fontsize=14)
plt.tight_layout(); plt.savefig('figures/fig04_boxplots_antes.png', bbox_inches='tight'); plt.close()

# Winsorización al percentil 1-99 en 4 variables con outliers relevantes
# (total_revenue y monthly_fee se conservan: variabilidad legítima del negocio)
vars_winsorizadas = ['total_watch_hours', 'support_tickets', 'sessions_per_week', 'avg_session_min']
for var in vars_winsorizadas:
    p01, p99 = df[var].quantile(0.01), df[var].quantile(0.99)
    df[var]  = df[var].clip(p01, p99)
    print(f"  '{var}' → Winsorizado [p01={p01:.2f}, p99={p99:.2f}]")

print("  'total_revenue' y 'monthly_fee' → Sin winsorizar (variabilidad legítima del negocio)")

# Figura 5 — Boxplots DESPUÉS
fig, axes = plt.subplots(2, 4, figsize=(18, 9)); axes = axes.flatten()
for i, var in enumerate(num_vars):
    axes[i].boxplot(df[var].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='#B5EAD7', color='#2D6A4F'),
                    medianprops=dict(color='#E07B54', linewidth=2.5),
                    whiskerprops=dict(color='#2D6A4F'), capprops=dict(color='#2D6A4F'),
                    flierprops=dict(marker='o', markerfacecolor='#E07B54', markersize=4, alpha=0.5))
    axes[i].set_title(var, fontsize=11)
axes[-1].set_visible(False)
fig.suptitle('Figura 5. Boxplots DESPUES del tratamiento (winsoriz. p1-p99)', fontsize=14)
plt.tight_layout(); plt.savefig('figures/fig05_boxplots_despues.png', bbox_inches='tight'); plt.close()
print("  → figures/fig04 y fig05 guardadas")


# =============================================================================
# SECCIÓN 5: FEATURE ENGINEERING (VARIABLES DERIVADAS)
# =============================================================================
print("\n[5/8] Feature Engineering...")

# Eliminar variables sin valor predictivo
df_model = df.drop(columns=['subscriber_id'])

# Variable 1: Segmento de antigüedad (ciclo de vida del cliente)
df_model['tenure_segment'] = pd.cut(
    df_model['tenure_months'], bins=[0, 6, 12, 24, 36],
    labels=['Nuevo (1-6m)', 'En desarrollo (7-12m)', 'Consolidado (13-24m)', 'Fiel (25-36m)'])
print("  + tenure_segment: segmento de antigüedad creado")

# Variable 2: Indicador de bajo uso (< 10h/mes → señal de desenganche)
df_model['low_usage'] = (df_model['total_watch_hours'] < 10).astype(int)
print(f"  + low_usage: {df_model['low_usage'].sum():,} suscriptores con bajo uso ({df_model['low_usage'].mean()*100:.1f}%)")

# Variable 3: Alto consumo de contenido original HBO (> 50%)
df_model['high_original'] = (df_model['hbo_original_share'] > 0.5).astype(int)
print(f"  + high_original: {df_model['high_original'].sum():,} suscriptores con alto consumo original")

# Variable 4: Ingresos normalizados por antigüedad
df_model['revenue_per_month'] = (df_model['total_revenue'] / df_model['tenure_months']).round(2)
print("  + revenue_per_month: valor del cliente normalizado creado")

# Variable 5: Ratio de engagement por sesión
df_model['engagement_ratio'] = (
    df_model['total_watch_hours'] / (df_model['sessions_per_week'] * 4.3)).round(3)
print("  + engagement_ratio: intensidad de uso por sesión creado")

# Variable 6: Flag de cliente en riesgo por calidad de servicio
df_model['at_risk_service'] = (
    (df_model['support_tickets'] >= 3) & (df_model['satisfaction_score'] <= 2)).astype(int)
print(f"  + at_risk_service: {df_model['at_risk_service'].sum():,} suscriptores en riesgo por servicio")

print(f"\n  Dataset tras feature engineering: {df_model.shape}")


# =============================================================================
# SECCIÓN 6: CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# =============================================================================
print("\n[6/8] Codificación de variables categóricas...")

# One-Hot Encoding para variables nominales
cat_cols = ['plan_type', 'payment_method', 'device_type', 'main_genre', 'region']
df_encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=False, dtype=int)

# Label Encoding ordinal para tenure_segment
tenure_map = {
    'Nuevo (1-6m)': 1, 'En desarrollo (7-12m)': 2,
    'Consolidado (13-24m)': 3, 'Fiel (25-36m)': 4
}
df_encoded['tenure_segment'] = df_encoded['tenure_segment'].map(tenure_map)

print(f"  One-Hot Encoding aplicado a: {cat_cols}")
print(f"  Label Encoding ordinal aplicado a: tenure_segment")
print(f"  Variables tras codificación: {df_encoded.shape[1]}")


# =============================================================================
# SECCIÓN 7: ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n[7/8] Análisis Exploratorio de Datos (EDA)...")

mg = df_model['churn'].mean() * 100  # Media global de churn

# Figura 6 — Distribuciones por clase de churn
nc = ['tenure_months', 'total_watch_hours', 'hbo_original_share', 'support_tickets',
      'satisfaction_score', 'monthly_fee', 'total_revenue', 'sessions_per_week', 'avg_session_min']
nm = ['Antiguedad (meses)', 'Horas visionado/mes', 'Proporcion contenido original',
      'Tickets soporte', 'Satisfaccion (1-5)', 'Cuota mensual (EUR)',
      'Ingresos totales (EUR)', 'Sesiones/semana', 'Duracion sesion (min)']
fig, axes = plt.subplots(3, 3, figsize=(18, 14)); axes = axes.flatten()
for i, (col, nom) in enumerate(zip(nc, nm)):
    d0 = df_model[df_model['churn']==0][col]
    d1 = df_model[df_model['churn']==1][col]
    axes[i].hist(d0, bins=35, alpha=0.65, color='#4878CF', label=f'Activo (mu={d0.mean():.1f})', density=True)
    axes[i].hist(d1, bins=35, alpha=0.65, color='#E07B54', label=f'Churn (mu={d1.mean():.1f})', density=True)
    axes[i].axvline(d0.mean(), color='#2255AA', linestyle='--', linewidth=1.8)
    axes[i].axvline(d1.mean(), color='#AA3300', linestyle='--', linewidth=1.8)
    axes[i].set_title(nom, fontsize=12); axes[i].set_ylabel('Densidad'); axes[i].legend(fontsize=9)
fig.suptitle('Figura 6. Distribuciones de variables numericas por clase de churn', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig('figures/fig06_distribuciones.png', bbox_inches='tight'); plt.close()
print("  → figures/fig06_distribuciones.png guardada")

# Figura 7 — Churn rate por variables categóricas
cat_v = ['plan_type', 'payment_method', 'device_type', 'main_genre', 'region', 'tenure_segment']
nom7  = ['Tipo de plan', 'Metodo de pago', 'Dispositivo', 'Genero favorito', 'Region', 'Segmento antiguedad']
fig, axes = plt.subplots(2, 3, figsize=(18, 13))
for ax, (var, nom) in zip(axes.flatten(), zip(cat_v, nom7)):
    cr = df_model.groupby(var)['churn'].mean().sort_values() * 100
    bars = ax.barh(cr.index.astype(str), cr.values,
                   color=[plt.cm.RdYlGn_r(v/100) for v in cr.values], edgecolor='white', height=0.6)
    ax.axvline(mg, color='#336699', linestyle='--', linewidth=1.5, label=f'Media ({mg:.1f}%)')
    ax.set_title(nom, fontsize=12); ax.set_xlabel('Tasa de churn (%)'); ax.legend(fontsize=9)
    [ax.text(v+0.3, b.get_y()+b.get_height()/2, f'{v:.1f}%', va='center', fontsize=10, fontweight='bold')
     for b, v in zip(bars, cr.values)]
    ax.set_xlim(0, cr.max()*1.3)
fig.suptitle('Figura 7. Tasa de churn por variables categoricas', fontsize=14)
plt.tight_layout(); plt.savefig('figures/fig07_churn_cat.png', bbox_inches='tight'); plt.close()
print("  → figures/fig07_churn_cat.png guardada")

# Figura 8 — Matriz de correlación
fig, ax = plt.subplots(figsize=(14, 11))
cc2 = ['tenure_months','total_watch_hours','hbo_original_share','support_tickets',
       'satisfaction_score','monthly_fee','total_revenue','sessions_per_week',
       'avg_session_min','active_profiles','months_paused','engagement_ratio',
       'revenue_per_month','low_usage','high_original','churn']
corr = df_model[cc2].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 9})
ax.set_title('Figura 8. Matriz de correlacion de variables numericas', fontsize=14, pad=12)
plt.tight_layout(); plt.savefig('figures/fig08_correlacion.png', bbox_inches='tight'); plt.close()
print("  → figures/fig08_correlacion.png guardada")

# Figura 9 — Análisis temporal del churn por antigüedad
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ct = df_model.groupby('tenure_months')['churn'].mean().reset_index()
ct.columns = ['tm', 'cr']
axes[0].plot(ct['tm'], ct['cr']*100, color='#E07B54', linewidth=2.5, marker='o', markersize=5)
axes[0].fill_between(ct['tm'], ct['cr']*100, alpha=0.15, color='#E07B54')
axes[0].axhline(mg, color='#336699', linestyle='--', linewidth=2, label=f'Media global ({mg:.1f}%)')
axes[0].axvspan(1, 6, alpha=0.08, color='red', label='Zona alto riesgo (1-6m)')
axes[0].set_xlabel('Antiguedad (meses)'); axes[0].set_ylabel('Tasa de churn (%)')
axes[0].set_title('Evolucion del churn por antiguedad'); axes[0].legend(fontsize=10)
for cv, col, lbl in [(0,'#4878CF','Activo'), (1,'#E07B54','Churn')]:
    d = df_model[df_model['churn']==cv]['tenure_months']
    axes[1].hist(d, bins=36, alpha=0.65, color=col, label=f'{lbl} (mu={d.mean():.1f}m)', density=True)
    axes[1].axvline(d.mean(), color=col, linestyle='--', linewidth=2)
axes[1].set_xlabel('Antiguedad (meses)'); axes[1].set_ylabel('Densidad')
axes[1].set_title('Distribucion de antiguedad por clase'); axes[1].legend(fontsize=10)
fig.suptitle('Figura 9. Analisis temporal del churn por antiguedad del suscriptor', fontsize=14)
plt.tight_layout(); plt.savefig('figures/fig09_temporal.png', bbox_inches='tight'); plt.close()
print("  → figures/fig09_temporal.png guardada")

# Figura 10 — Análisis bivariante
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
for cv, col, lbl in [(0,'#4878CF','Activo'), (1,'#E07B54','Churn')]:
    d = df_model[df_model['churn']==cv]['total_watch_hours']
    axes[0].hist(d, bins=35, alpha=0.65, color=col, label=f'{lbl} (mu={d.mean():.1f}h)', density=True)
    axes[0].axvline(d.mean(), color=col, linestyle='--', linewidth=2)
axes[0].axvline(10, color='black', linestyle=':', linewidth=1.5, label='Umbral bajo uso (10h)')
axes[0].set_xlabel('Horas visionado/mes'); axes[0].set_ylabel('Densidad')
axes[0].set_title('Horas de visionado vs Churn'); axes[0].legend(fontsize=10)
cs2 = df_model.groupby('satisfaction_score')['churn'].mean() * 100
b2  = axes[1].bar(cs2.index, cs2.values,
                  color=['#d73027','#f46d43','#fdae61','#74add1','#313695'], edgecolor='white', width=0.6)
axes[1].set_xlabel('Satisfaction score (1-5)'); axes[1].set_ylabel('Tasa de churn (%)')
axes[1].set_title('Satisfaccion vs Churn')
axes[1].set_xticklabels(['','1\n(Muy bajo)','2\n(Bajo)','3\n(Medio)','4\n(Alto)','5\n(Muy alto)'], fontsize=10)
[axes[1].text(b.get_x()+b.get_width()/2, v+0.3, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
 for b, v in zip(b2, cs2.values)]
ct2 = df_model.groupby('support_tickets')['churn'].mean() * 100
axes[2].bar(ct2.index, ct2.values, color='#E07B54', edgecolor='white', width=0.7)
axes[2].set_xlabel('No tickets soporte'); axes[2].set_ylabel('Tasa de churn (%)')
axes[2].set_title('Tickets de soporte vs Churn')
[axes[2].text(idx, v+0.3, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
 for idx, v in ct2.items()]
fig.suptitle('Figura 10. Analisis bivariante: variables clave vs tasa de churn', fontsize=14)
plt.tight_layout(); plt.savefig('figures/fig10_bivariante.png', bbox_inches='tight'); plt.close()
print("  → figures/fig10_bivariante.png guardada")

# Figura 11 — Variables derivadas
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
configs = [
    ('low_usage',       ['Alto uso (>=10h/mes)', 'Bajo uso (<10h/mes)'],  ['#4878CF','#E07B54'], 'Uso del servicio vs Churn'),
    ('high_original',   ['Bajo consumo original','Alto consumo original'], ['#E07B54','#4878CF'], 'Contenido original vs Churn'),
    ('at_risk_service', ['Sin riesgo servicio',   'En riesgo servicio'],   ['#4878CF','#E07B54'], 'Calidad de servicio vs Churn'),
]
for ax, (var, labels, colors, title) in zip(axes, configs):
    vals = df_model.groupby(var)['churn'].mean() * 100
    bars = ax.bar(labels, vals.values, color=colors, edgecolor='white', width=0.45)
    ax.axhline(mg, color='#336699', linestyle='--', linewidth=1.5, label=f'Media global ({mg:.1f}%)')
    ax.set_ylabel('Tasa de churn (%)'); ax.set_title(title, fontsize=13)
    ax.set_ylim(0, vals.max()*1.3); ax.legend(fontsize=10)
    [ax.text(b.get_x()+b.get_width()/2, v+0.4, f'{v:.1f}%', ha='center', fontsize=14, fontweight='bold')
     for b, v in zip(bars, vals.values)]
fig.suptitle('Figura 11. Variables derivadas y su impacto en la tasa de churn', fontsize=14)
plt.tight_layout(); plt.savefig('figures/fig11_derivadas.png', bbox_inches='tight'); plt.close()
print("  → figures/fig11_derivadas.png guardada")


# =============================================================================
# SECCIÓN 8: EXPORTACIÓN DE DATASETS FINALES
# =============================================================================
print("\n[8/8] Exportación de datasets finales...")

# Dataset limpio con variables derivadas (para análisis descriptivo)
df_model.to_csv('hbo_max_limpio.csv', index=False, sep=';')
print(f"  → hbo_max_limpio.csv guardado: {df_model.shape}")

# Dataset codificado listo para modelado
df_encoded.to_csv('hbo_max_modelado.csv', index=False, sep=';')
print(f"  → hbo_max_modelado.csv guardado: {df_encoded.shape}")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 65)
print("  RESUMEN — INGENIERÍA DEL DATO COMPLETADA")
print("=" * 65)
print(f"  Dataset original:         5.000 registros × 11 variables")
print(f"  Dataset enriquecido:      5.000 registros × 20 variables")
print(f"  Dataset combinado:       10.000 registros × 20 variables")
print(f"  Dataset limpio:          {df_model.shape[0]:,} registros × {df_model.shape[1]} variables")
print(f"  Dataset modelado:        {df_encoded.shape[0]:,} registros × {df_encoded.shape[1]} variables")
print(f"  Tasa de churn final:     {df_model['churn'].mean()*100:.1f}%")
print(f"  Nulos restantes:         {df_model.isnull().sum().sum()}")
print(f"  Figuras generadas:       11 (en carpeta figures/)")
print(f"  Datasets exportados:     3 CSV")
print("=" * 65)
print("  ✅ Pipeline de ingeniería del dato completado con éxito")
print("=" * 65)
