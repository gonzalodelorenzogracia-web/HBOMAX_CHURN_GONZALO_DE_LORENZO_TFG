import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Número de suscriptores HBO Max
n_subscribers = 5000

# ID de suscriptores
subscriber_id = np.arange(1, n_subscribers + 1)

# Antigüedad en meses (1-36, CORREGIDO y LIMITADO)
tenure_months = np.random.randint(1, 37, size=n_subscribers)
tenure_months = np.clip(tenure_months, 1, 36) # Garantía explícita máx 36 <-- ESPACIO CORREGIDO

# Tipo de plan HBO Max (precios España 2025)
plan_type = np.random.choice(
    ['Mensual (14.99€)', 'Anual (12.99€/mes)', 'Familia (19.99€/mes)'],
    size=n_subscribers,
    p=[0.65, 0.20, 0.15]
)

# Método de pago
payment_method = np.random.choice(
    ['Tarjeta', 'PayPal', 'Apple Pay/Google Pay'],
    size=n_subscribers,
    p=[0.55, 0.25, 0.20]
)

# Horas de visionado total al mes
total_watch_hours = np.clip(
    np.random.normal(loc=25, scale=12, size=n_subscribers),
    a_min=0, a_max=None
)

# % de contenido HBO original visto (Beta sesgada baja ~40%)
hbo_original_share = np.clip(
    np.random.beta(2, 3, size=n_subscribers), 0, 1
)

# Tickets de soporte (Poisson λ=0.8)
support_tickets = np.random.poisson(lam=0.8, size=n_subscribers)

# Puntuación satisfacción (1-5, correlacionada con uso/soporte)
satisfaction_score = np.clip(
    np.round(
        4.0 + 0.3 * hbo_original_share - 0.2 * support_tickets 
        + 0.01 * tenure_months + np.random.normal(0, 0.5, size=n_subscribers)
    ),
    1, 5
).astype(int)

# Cuota mensual CORREGIDA (ruido mayor, más realista)
base_fee = []
for plan in plan_type:
    if 'Mensual' in plan:
        base_fee.append(14.99)
    elif 'Anual' in plan:
        base_fee.append(12.99)
    else:  # Familia
        base_fee.append(19.99)
base_fee = np.array(base_fee)

monthly_fee = base_fee + np.random.uniform(-1.5, 1.5, size=n_subscribers)
monthly_fee = np.round(np.clip(monthly_fee, 9.99, 24.99), 2)

# Ingresos totales generados
total_revenue = np.round(monthly_fee * tenure_months, 2)

# --- INICIO DE LA CORRECCIÓN DE LÓGICA (AttributeError) ---

# Convertir plan_type a Series de Pandas para usar .str.contains() vectorizado
plan_type_series = pd.Series(plan_type)

# Probabilidad churn (lógica realista HBO Max)
prob_churn = (
    # CORREGIDO: Usar .str.contains() y multiplicar por 1.0 para convertir True/False a 1.0/0.0
    0.40 * plan_type_series.str.contains('Mensual', regex=False) * 1.0
    + 0.30 * (hbo_original_share < 0.3) * 1.0
    + 0.20 * (satisfaction_score <= 2) * 1.0
    + 0.15 * (support_tickets >= 2) * 1.0
    + 0.10 * (total_watch_hours < 10) * 1.0
    - 0.25 * (tenure_months > 12) * 1.0
)

# --- FIN DE LA CORRECCIÓN DE LÓGICA ---

# Normalizar [0.01, 0.75]
prob_churn = 0.01 + 0.74 * (prob_churn - prob_churn.min()) / (prob_churn.max() - prob_churn.min())

# Churn (1=baja, 0=renueva)
churn = np.random.binomial(1, prob_churn)

# DataFrame HBO Max FINAL
df = pd.DataFrame({
    'subscriber_id': subscriber_id,
    'tenure_months': tenure_months,
    'plan_type': plan_type,
    'payment_method': payment_method,
    'total_watch_hours': np.round(total_watch_hours, 1),
    'hbo_original_share': np.round(hbo_original_share, 3),
    'support_tickets': support_tickets,
    'satisfaction_score': satisfaction_score,
    'monthly_fee': monthly_fee,
    'total_revenue': total_revenue,
    'churn': churn
})

# GUARDAR y VERIFICACIONES
df.to_csv('hbo_max_churn_dataset.csv', index=False)

print("✅ Dataset HBO Max generado correctamente")
print("📊 Primeras 5 filas:")
print(df.head())
print("\n🔍 VERIFICACIONES:")
print(f"• tenure_months: {df['tenure_months'].min()}-{df['tenure_months'].max()} (máx 36 OK)")
print(f"• monthly_fee únicos: {df['monthly_fee'].nunique()}/5000")
print(f"• Porcentaje churn: {round(df['churn'].mean() * 100, 1)}%")
print(f"• Valores tenure >36: {(df['tenure_months'] > 36).sum()} (0 OK)")
