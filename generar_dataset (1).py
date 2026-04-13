import numpy as np
import pandas as pd

# Fijar semilla para reproducibilidad
np.random.seed(42)

# Número de clientes
n_customers = 5000

# Variables básicas
customer_id = np.arange(1, n_customers + 1)

# Antigüedad en meses (0–60)
tenure_months = np.random.randint(1, 61, size=n_customers)

# Tipo de contrato
contract_type = np.random.choice(
    ['Mensual', 'Trimestral', 'Anual'],
    size=n_customers,
    p=[0.6, 0.25, 0.15]
)

# Método de pago
payment_method = np.random.choice(
    ['Tarjeta', 'Domiciliación', 'PayPal'],
    size=n_customers,
    p=[0.5, 0.35, 0.15]
)

# Uso del servicio (por ejemplo, horas al mes)
usage_hours = np.clip(
    np.random.normal(loc=30, scale=10, size=n_customers),
    a_min=0,
    a_max=None
)

# Número de incidencias con soporte al cliente
support_tickets = np.random.poisson(lam=1.2, size=n_customers)

# Satisfacción (1–5)
satisfaction_score = np.clip(
    np.round(
        4.2
        - 0.15 * support_tickets
        + 0.01 * tenure_months
        + np.random.normal(0, 0.6, size=n_customers)
    ),
    1,
    5
).astype(int)

# Precio base mensual según contrato
base_price = []
for c in contract_type:
    if c == 'Mensual':
        base_price.append(25)
    elif c == 'Trimestral':
        base_price.append(22)
    else:  # Anual
        base_price.append(20)
base_price = np.array(base_price, dtype=float)

# Descuentos según método de pago
discount = np.where(payment_method == 'Domiciliación', 2,
            np.where(payment_method == 'PayPal', 1, 0)).astype(float)

monthly_fee = base_price - discount + np.random.normal(0, 1.5, size=n_customers)
monthly_fee = np.round(np.clip(monthly_fee, 10, None), 2)

# Ingresos totales generados por cliente
total_revenue = np.round(monthly_fee * tenure_months, 2)

# Probabilidad de churn (modelo “verdadero” que tú defines)
# Factores: contratos mensuales, menor antigüedad, baja satisfacción,
# muchos tickets y cuota más alta aumentan probabilidad de churn.
prob_churn = (
    0.35 * (contract_type == 'Mensual').astype(float)
    + 0.15 * (contract_type == 'Trimestral').astype(float)
    + 0.20 * (monthly_fee > 27).astype(float)
    + 0.25 * (satisfaction_score <= 2).astype(float)
    + 0.15 * (support_tickets >= 3).astype(float)
    - 0.20 * (tenure_months > 24).astype(float)
)

# Normalizar a rango [0.02, 0.8]
prob_churn = 0.02 + 0.78 * (prob_churn - prob_churn.min()) / (
    prob_churn.max() - prob_churn.min()
)

# Variable objetivo churn (1 si se da de baja, 0 si permanece)
churn = np.random.binomial(1, prob_churn)

# Construir DataFrame final
df = pd.DataFrame({
    'customer_id': customer_id,
    'tenure_months': tenure_months,
    'contract_type': contract_type,
    'payment_method': payment_method,
    'usage_hours': np.round(usage_hours, 2),
    'support_tickets': support_tickets,
    'satisfaction_score': satisfaction_score,
    'monthly_fee': monthly_fee,
    'total_revenue': total_revenue,
    'churn': churn
})

# Guardar a CSV
df.to_csv('customer_churn_dataset.csv', index=False)

print(df.head())
print("Porcentaje de churn:", df['churn'].mean())
