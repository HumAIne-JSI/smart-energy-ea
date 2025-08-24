import os
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str):
    data = pd.read_csv(csv_path)
    data["status_binary"] = data["status"].map({"secure": 1, "insecure": 0})
    columns_to_drop = [
        "timestamp",
        "status",
        "status_binary",
        "max_line_loading_percent_basecase",
        "min_bus_voltage_pu_basecase",
        "max_bus_voltage_pu_basecase",
        "max_line_loading_percent_contingency",
        "min_bus_voltage_pu_contingency",
        "max_bus_voltage_pu_contingency",
    ]
    X = data.drop(columns=columns_to_drop, errors="ignore")
    y = data["status_binary"]
    return X, y


def split_dataset(X, y, test_size=0.1, random_state=42):
    X_pool_random, X_val_random, y_pool_random, y_val_random = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    split_index = int((1 - test_size) * len(X))
    X_pool_seq = X.iloc[:split_index]
    y_pool_seq = y.iloc[:split_index]
    X_val_seq = X.iloc[split_index:]
    y_val_seq = y.iloc[split_index:]
    return {
        "random": (
            X_pool_random.reset_index(drop=True),
            X_val_random.reset_index(drop=True),
            y_pool_random.reset_index(drop=True),
            y_val_random.reset_index(drop=True),
        ),
        "sequential": (
            X_pool_seq.reset_index(drop=True),
            X_val_seq.reset_index(drop=True),
            y_pool_seq.reset_index(drop=True),
            y_val_seq.reset_index(drop=True),
        ),
    }


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


def run_active_learning(X_pool, y_pool, X_val, y_val, strategy, initial_size, batch_size, iterations, random_state=42):
    rng = np.random.default_rng(random_state)
    n_samples = len(X_pool)
    labeled_mask = np.zeros(n_samples, dtype=bool)
    labeled_mask[rng.choice(n_samples, size=min(initial_size, n_samples), replace=False)] = True

    metrics_per_iteration = []
    start_time = time.perf_counter()

    for it in range(iterations):
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        X_train = X_pool[labeled_mask]
        y_train = y_pool[labeled_mask]
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

    duration = time.perf_counter() - start_time
    return metrics_per_iteration, duration


def ensure_directories(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def run_experiment_grid(
    csv_path,
    strategies,
    initial_sizes,
    batch_sizes,
    iteration_counts,
    test_size=0.1,
    random_state=42,
    figures_dir="figures",
    tables_dir="tables",
):
    ensure_directories(figures_dir, tables_dir)
    X, y = load_dataset(csv_path)
    splits = split_dataset(X, y, test_size=test_size, random_state=random_state)

    timestamp_all = datetime.now().strftime("%Y%m%d_%H%M%S")
    combinations = list(product(initial_sizes, batch_sizes, iteration_counts, splits.items(), strategies))

    results_rows = []
    all_iteration_metrics = []
    start_all = time.perf_counter()

    for init_sz, batch_sz, iters, (split_name, (X_pool, X_val, y_pool, y_val)), strategy in combinations:
        metrics_iter, duration = run_active_learning(
            X_pool, y_pool, X_val, y_val,
            strategy=strategy,
            initial_size=init_sz,
            batch_size=batch_sz,
            iterations=iters,
            random_state=random_state,
        )

        for m in metrics_iter:
            m.update({
                "timestamp": timestamp_all,
                "strategy_type": strategy,
                "split_type": split_name,
                "initial_size": init_sz,
                "batch_size": batch_sz,
                "iterations": iters,
                "iteration_id": m["iteration"],
            })
        all_iteration_metrics.extend(metrics_iter)

        accs = [m["accuracy"] for m in metrics_iter]
        precisions = [m["precision"] for m in metrics_iter]
        recalls = [m["recall"] for m in metrics_iter]
        f1s = [m["f1"] for m in metrics_iter]
        aucs = [m["roc_auc"] for m in metrics_iter]

        total_labeled = init_sz + iters * batch_sz
        results_rows.append({
            "timestamp": timestamp_all,
            "strategy_type": strategy,
            "split_type": split_name,
            "iterations": iters,
            "test_size": test_size,
            "initial_size": init_sz,
            "batch_size": batch_sz,
            "total_labeled_samples": total_labeled,
            "accuracy_final": accs[-1] if accs else None,
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "precision_mean": np.mean(precisions),
            "recall_mean": np.mean(recalls),
            "f1_mean": np.mean(f1s),
            "roc_auc_mean": np.mean(aucs),
            "duration_sec": duration,
        })

    duration_all = time.perf_counter() - start_all

    df_results = pd.DataFrame(results_rows)
    df_detailed = pd.DataFrame(all_iteration_metrics)

    csv_name = os.path.join(tables_dir, f"active_learning_results_{timestamp_all}.csv")
    xlsx_name = csv_name.replace(".csv", ".xlsx")
    df_results.to_csv(csv_name, index=False)
    try:
        df_results.to_excel(xlsx_name, index=False)
    except Exception:
        # excel ni obvezen
        pass

    detailed_csv_name = os.path.join(tables_dir, f"al_metrics_per_iteration_{timestamp_all}.csv")
    df_detailed.to_csv(detailed_csv_name, index=False)

    print(f"Rezultati shranjeni v: {csv_name}")
    print(f"Iteracije shranjene v: {detailed_csv_name}")
    print(f"Skupni čas izvajanja: {duration_all:.2f} s")
    return df_results


if __name__ == "__main__":
    INITIAL_SIZES = [50, 100, 200]
    BATCH_SIZES = [10, 25, 50]
    ITERATION_COUNTS = [20, 40, 60]

    STRATEGIES = ["uncertainty", "entropy", "margin", "random"]

    DATA_PATH = "data/simulation_security_labels_n-1.csv"
    FIGURES_DIR = "figures"
    TABLES_DIR = "tables"

    run_experiment_grid(
        csv_path=DATA_PATH,
        strategies=STRATEGIES,
        initial_sizes=INITIAL_SIZES,
        batch_sizes=BATCH_SIZES,
        iteration_counts=ITERATION_COUNTS,
        test_size=0.1,
        random_state=42,
        figures_dir=FIGURES_DIR,
        tables_dir=TABLES_DIR,
    )