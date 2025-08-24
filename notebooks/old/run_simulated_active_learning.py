import os
import pandas as pd
from sklearn.model_selection import train_test_split

from active_learning_with_simulator import run_active_learning

# Naloži podatke (pool brez oznak)
df = pd.read_csv("data/simulation_security_labels_n-1.csv")

# Loči značilke in ciljno spremenljivko
X = df.drop(columns=["timestamp", "status"])  # status namenoma odstranjena, ker simuliramo on-demand
# y tu ni potreben, a ustvarimo dummy zaradi signatur
import numpy as np
y = np.zeros(len(X), dtype=int)

# Naključno razdeli v pool in validation set
X_pool, X_val, y_pool, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Zaženi aktivno učenje s simulacijo
metrics, duration = run_active_learning(
    X_pool=X_pool,
    y_pool=y_pool,  # ne uporablja se če simulate_on_demand=True
    X_val=X_val,
    y_val=y_val,
    strategy="entropy",
    initial_size=10,
    batch_size=10,
    iterations=20,
    simulate_on_demand=True,
)

# Shrani metrike v CSV
os.makedirs("tables", exist_ok=True)
pd.DataFrame(metrics).to_csv("tables/metrics_simulated_entropy.csv", index=False)

print(f"Končano. Čas izvajanja: {duration:.2f} s")