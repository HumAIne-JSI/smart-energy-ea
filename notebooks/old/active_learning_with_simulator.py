import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from simulator_interface import query_simulator_cached


def compute_query_scores(proba: np.ndarray, strategy: str) -> np.ndarray:
    if strategy == "uncertainty":
        return 1.0 - proba.max(axis=1)
    elif strategy == "entropy":
        logp = np.log(proba + 1e-12)
        return -np.sum(proba * logp, axis=1)
    elif strategy == "margin":
        sorted_proba = np.sort(proba, axis=1)
        return -(sorted_proba[:, -1] - sorted_proba[:, -2])
    elif strategy == "random":
        return np.zeros(proba.shape[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def sample_to_dict(row):
    """Pretvori vrstico DataFrame-a v slovar za simulator."""
    return row.to_dict()


def run_active_learning(
    X_pool,
    y_pool,
    X_val,
    y_val,
    strategy,
    initial_size,
    batch_size,
    iterations,
    random_state=42,
    simulate_on_demand=False,
):
    rng = np.random.default_rng(random_state)

    X_pool = X_pool.reset_index(drop=True)
    y_pool = y_pool.reset_index(drop=True)

    n_samples = len(X_pool)
    labeled_mask = np.zeros(n_samples, dtype=bool)
    labeled_mask[rng.choice(n_samples, size=min(initial_size, n_samples), replace=False)] = True

    metrics_per_iteration = []

    # hranimo znane labele, da simulatorja ne kličemo večkrat
    if simulate_on_demand:
        y_labels = np.full(n_samples, np.nan)
    else:
        y_labels = y_pool.values if hasattr(y_pool, "values") else y_pool

    start_time = time.perf_counter()

    # inicialne labele pridobi enkrat (ne vsako iteracijo)
    if simulate_on_demand:
        init_idx = np.where(labeled_mask)[0]
        if len(init_idx):
            for i in init_idx:
                if np.isnan(y_labels[i]):
                    y_labels[i] = 1 if query_simulator_cached(sample_to_dict(X_pool.iloc[i])) == "secure" else 0

    for it in range(iterations):
        labeled_indices = np.where(labeled_mask)[0]

        # 🔄 Trening set (labele so v y_labels)
        y_train = y_labels[labeled_indices].astype(int)
        X_train = X_pool.iloc[labeled_indices]

        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics_per_iteration.append({
            "iteration": it + 1,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_val, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_val, y_pred, average="macro", zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob),
        })

        if not (~labeled_mask).any():
            break

        unlabeled_indices = np.where(~labeled_mask)[0]
        if strategy == "random":
            query_indices = rng.choice(unlabeled_indices, size=min(batch_size, len(unlabeled_indices)), replace=False)
        else:
            # ⚡ računaj probo samo na neoznačenih
            proba_unl = model.predict_proba(X_pool.iloc[unlabeled_indices])
            scores = compute_query_scores(proba_unl, strategy)
            top_local = np.argsort(scores)[::-1][:batch_size]
            query_indices = unlabeled_indices[top_local]

        labeled_mask[query_indices] = True

        # pridobi nove labele samo za NOVO izbrane
        if simulate_on_demand:
            for i in query_indices:
                if np.isnan(y_labels[i]):
                    y_labels[i] = 1 if query_simulator_cached(sample_to_dict(X_pool.iloc[i])) == "secure" else 0

    duration = time.perf_counter() - start_time
    return metrics_per_iteration, duration