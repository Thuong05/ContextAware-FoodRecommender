from __future__ import annotations

import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ORDER_ITEM_PATH = Path("order_item_final_best.csv")
ORDER_PATH = Path("order.csv")
DEFAULT_OUTPUT_DIR = Path("models") / "category_recommender_v2"
PRIOR_BLEND_WEIGHT = 0.50

warnings.simplefilter("ignore", PerformanceWarning)


def get_time_slot(hour: int) -> int:
    if 5 <= hour < 11:
        return 0
    if 11 <= hour < 14:
        return 1
    if 14 <= hour < 18:
        return 2
    if 18 <= hour < 23:
        return 3
    return 4


def customer_segment(order_count_before_current: float) -> int:
    if order_count_before_current <= 0:
        return 0
    if order_count_before_current <= 4:
        return 1
    if order_count_before_current <= 19:
        return 2
    return 3


def cyclical_features(values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2.0 * math.pi * values.astype(float) / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        },
        index=values.index,
    )


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(-y_score, axis=1)[:, :k]
    recalls: List[float] = []
    for row_idx in range(y_true.shape[0]):
        true_idx = np.flatnonzero(y_true[row_idx])
        if len(true_idx) == 0:
            continue
        hit_count = len(set(true_idx) & set(top_k_idx[row_idx]))
        recalls.append(hit_count / len(true_idx))
    return float(np.mean(recalls)) if recalls else 0.0


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(-y_score, axis=1)[:, :k]
    precisions: List[float] = []
    for row_idx in range(y_true.shape[0]):
        true_idx = set(np.flatnonzero(y_true[row_idx]))
        if not true_idx:
            continue
        hit_count = len(true_idx & set(top_k_idx[row_idx]))
        precisions.append(hit_count / k)
    return float(np.mean(precisions)) if precisions else 0.0


def map_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(-y_score, axis=1)[:, :k]
    ap_scores: List[float] = []
    for row_idx in range(y_true.shape[0]):
        true_idx = set(np.flatnonzero(y_true[row_idx]))
        if not true_idx:
            continue
        hits = 0
        score = 0.0
        for rank, pred_idx in enumerate(top_k_idx[row_idx], start=1):
            if pred_idx in true_idx:
                hits += 1
                score += hits / rank
        ap_scores.append(score / min(len(true_idx), k))
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, ks: Sequence[int] = (3, 5, 10)) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(y_true, y_score, k)
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_score, k)
        metrics[f"map@{k}"] = map_at_k(y_true, y_score, k)
    return metrics


@dataclass
class PreparedData:
    order_base_all: pd.DataFrame
    order_history_df: pd.DataFrame
    order_df: pd.DataFrame
    all_categories: List[str]
    target_cols: List[str]
    baseline_feature_cols: List[str]
    improved_feature_cols: List[str]
    customer_top_item_map: Dict[Tuple[float, str], str]
    context_top_item_map: Dict[Tuple[str, str, int], str]
    global_top_item_map: Dict[str, str]


def build_top_item_maps(df: pd.DataFrame) -> Tuple[Dict[Tuple[float, str], str], Dict[Tuple[str, str, int], str], Dict[str, str]]:
    customer_map = (
        df.dropna(subset=["customer_id"])
        .groupby(["customer_id", "category", "item_name"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(["customer_id", "category", "quantity", "item_name"], ascending=[True, True, False, True])
        .drop_duplicates(["customer_id", "category"])
    )
    customer_top_item_map = {
        (float(row.customer_id), row.category): row.item_name
        for row in customer_map.itertuples(index=False)
    }

    context_map = (
        df.groupby(["category", "order_type", "time_slot", "item_name"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(["category", "order_type", "time_slot", "quantity", "item_name"], ascending=[True, True, True, False, True])
        .drop_duplicates(["category", "order_type", "time_slot"])
    )
    context_top_item_map = {
        (row.category, row.order_type, int(row.time_slot)): row.item_name
        for row in context_map.itertuples(index=False)
    }

    global_map = (
        df.groupby(["category", "item_name"])["quantity"]
        .sum()
        .reset_index()
        .sort_values(["category", "quantity", "item_name"], ascending=[True, False, True])
        .drop_duplicates(["category"])
    )
    global_top_item_map = {row.category: row.item_name for row in global_map.itertuples(index=False)}
    return customer_top_item_map, context_top_item_map, global_top_item_map


def load_and_prepare_data() -> PreparedData:
    items = pd.read_csv(ORDER_ITEM_PATH)
    orders = pd.read_csv(
        ORDER_PATH,
        low_memory=False,
        usecols=["order_id", "customer_id", "order_type", "created_at", "order_total", "total_items"],
    )

    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(0).clip(lower=0)
    items["price"] = pd.to_numeric(items["price"], errors="coerce").fillna(0).clip(lower=0)
    orders["order_total"] = pd.to_numeric(orders["order_total"], errors="coerce")
    orders["total_items"] = pd.to_numeric(orders["total_items"], errors="coerce")
    orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce")

    order_base_all = (
        orders.dropna(subset=["order_id", "created_at", "order_total"])
        .drop_duplicates(subset=["order_id"])
        .copy()
    )
    order_base_all["order_id"] = order_base_all["order_id"].astype(int)
    order_base_all = order_base_all.sort_values(["customer_id", "created_at", "order_id"]).reset_index(drop=True)

    order_history_df = order_base_all[order_base_all["order_type"].isin(["collection", "delivery"])].copy()
    order_history_df = order_history_df.sort_values(["customer_id", "created_at", "order_id"]).reset_index(drop=True)

    merged = pd.merge(
        items,
        order_history_df[["order_id", "customer_id", "order_type", "created_at", "order_total", "total_items"]],
        on="order_id",
        how="inner",
        validate="m:1",
    )
    merged["line_revenue"] = merged["quantity"] * merged["price"]

    order_df = (
        merged.groupby("order_id")
        .agg(
            customer_id=("customer_id", "first"),
            order_type=("order_type", "first"),
            created_at=("created_at", "first"),
            order_total=("order_total", "first"),
            total_items=("total_items", "first"),
            categories=("category", lambda x: list(pd.unique(x))),
            n_categories=("category", "nunique"),
        )
        .reset_index()
    )
    order_df = order_df.sort_values(["customer_id", "created_at", "order_id"]).reset_index(drop=True)

    all_categories = sorted(merged["category"].unique())
    target_cols = [f"target__{cat}" for cat in all_categories]
    revenue_share_cols = [f"revshare__{cat}" for cat in all_categories]

    for cat, target_col in zip(all_categories, target_cols):
        order_df[target_col] = order_df["categories"].apply(lambda values, category=cat: int(category in values))

    revenue_by_cat = (
        merged.groupby(["order_id", "category"])["line_revenue"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=all_categories, fill_value=0)
    )
    revenue_share = revenue_by_cat.div(revenue_by_cat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    revenue_share.columns = revenue_share_cols
    order_df = order_df.merge(revenue_share.reset_index(), on="order_id", how="left", validate="1:1")

    order_history_df["cum_orders"] = order_history_df.groupby("customer_id").cumcount()
    order_history_df["is_new"] = (order_history_df["cum_orders"] == 0).astype(int)
    order_history_df["cust_seg"] = order_history_df["cum_orders"].apply(customer_segment)
    order_history_df["cum_avg_spend"] = order_history_df.groupby("customer_id")["order_total"].transform(
        lambda series: series.expanding().mean().shift(1)
    )
    order_history_df["prev_total"] = order_history_df.groupby("customer_id")["order_total"].shift(1)
    order_history_df["prev_items"] = order_history_df.groupby("customer_id")["total_items"].shift(1)
    order_history_df["avg_spend_last3"] = order_history_df.groupby("customer_id")["order_total"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    order_history_df["avg_items_last3"] = order_history_df.groupby("customer_id")["total_items"].transform(
        lambda series: series.shift(1).rolling(3, min_periods=1).mean()
    )
    order_history_df["prev_created_at"] = order_history_df.groupby("customer_id")["created_at"].shift(1)
    order_history_df["days_since_prev"] = (
        (order_history_df["created_at"] - order_history_df["prev_created_at"]).dt.total_seconds() / 86400.0
    )

    behavior_cols = [
        "order_id",
        "cum_orders",
        "is_new",
        "cust_seg",
        "cum_avg_spend",
        "prev_total",
        "prev_items",
        "avg_spend_last3",
        "avg_items_last3",
        "days_since_prev",
    ]
    order_df = order_df.merge(order_history_df[behavior_cols], on="order_id", how="left", validate="1:1")

    order_df["hour"] = order_df["created_at"].dt.hour
    order_df["day_of_week"] = order_df["created_at"].dt.dayofweek
    order_df["month"] = order_df["created_at"].dt.month
    order_df["is_weekend"] = (order_df["day_of_week"] >= 5).astype(int)
    order_df["time_slot"] = order_df["hour"].apply(get_time_slot)
    order_df["order_type_enc"] = (order_df["order_type"] == "delivery").astype(int)

    for cyc_df in (
        cyclical_features(order_df["hour"], 24, "hour"),
        cyclical_features(order_df["day_of_week"], 7, "dow"),
        cyclical_features(order_df["month"], 12, "month"),
    ):
        order_df = pd.concat([order_df, cyc_df], axis=1)

    for seg_value in range(4):
        order_df[f"cust_seg_{seg_value}"] = (order_df["cust_seg"] == seg_value).astype(int)

    order_df = pd.concat(
        [
            order_df,
            pd.get_dummies(order_df["time_slot"], prefix="time_slot"),
            pd.get_dummies(order_df["day_of_week"], prefix="dow_onehot"),
        ],
        axis=1,
    )

    history_feature_cols: List[str] = []
    improved_history_feature_cols: List[str] = []

    for cat, target_col, share_col in zip(all_categories, target_cols, revenue_share_cols):
        hist_col = f"hist__{cat}"
        recent_col = f"recent3__{cat}"
        ewm_col = f"ewm__{cat}"
        same_type_col = f"same_type__{cat}"
        share_ewm_col = f"share_ewm__{cat}"
        weighted_col = f"weighted__{cat}"

        order_df[hist_col] = (
            order_df.groupby("customer_id")[target_col]
            .transform(lambda series: series.shift(1).expanding().mean())
            .fillna(0.0)
        )
        order_df[recent_col] = (
            order_df.groupby("customer_id")[target_col]
            .transform(lambda series: series.shift(1).rolling(3, min_periods=1).mean())
            .fillna(0.0)
        )
        order_df[ewm_col] = (
            order_df.groupby("customer_id")[target_col]
            .transform(lambda series: series.shift(1).ewm(alpha=0.5, adjust=False).mean())
            .fillna(0.0)
        )
        order_df[same_type_col] = (
            order_df.groupby(["customer_id", "order_type"])[target_col]
            .transform(lambda series: series.shift(1).expanding().mean())
            .fillna(0.0)
        )
        order_df[share_ewm_col] = (
            order_df.groupby("customer_id")[share_col]
            .transform(lambda series: series.shift(1).ewm(alpha=0.4, adjust=False).mean())
            .fillna(0.0)
        )
        order_df[weighted_col] = (
            0.35 * order_df[hist_col]
            + 0.25 * order_df[recent_col]
            + 0.20 * order_df[ewm_col]
            + 0.10 * order_df[same_type_col]
            + 0.10 * order_df[share_ewm_col]
        )

        history_feature_cols.append(weighted_col)
        improved_history_feature_cols.extend([hist_col, recent_col, ewm_col, same_type_col, share_ewm_col, weighted_col])

    baseline_feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "time_slot",
        "order_type_enc",
        "cum_orders",
        "is_new",
        "cust_seg",
        "cum_avg_spend",
        "prev_total",
    ] + history_feature_cols

    improved_feature_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "time_slot",
        "order_type_enc",
        "cum_orders",
        "is_new",
        "cust_seg",
        "cum_avg_spend",
        "prev_total",
        "prev_items",
        "avg_spend_last3",
        "avg_items_last3",
        "days_since_prev",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ] + [f"cust_seg_{seg_value}" for seg_value in range(4)] + [
        col for col in order_df.columns if col.startswith("time_slot_") or col.startswith("dow_onehot_")
    ] + improved_history_feature_cols

    merged["time_slot"] = merged["created_at"].dt.hour.apply(get_time_slot)
    customer_top_item_map, context_top_item_map, global_top_item_map = build_top_item_maps(merged)

    return PreparedData(
        order_base_all=order_base_all,
        order_history_df=order_history_df,
        order_df=order_df,
        all_categories=all_categories,
        target_cols=target_cols,
        baseline_feature_cols=baseline_feature_cols,
        improved_feature_cols=improved_feature_cols,
        customer_top_item_map=customer_top_item_map,
        context_top_item_map=context_top_item_map,
        global_top_item_map=global_top_item_map,
    )


def split_train_test(order_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ordered = order_df.sort_values(["created_at", "order_id"]).reset_index(drop=True)
    split_idx = int(len(ordered) * 0.8)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def add_context_priors(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_cols: Sequence[str],
    context_cols: Sequence[str] = ("order_type", "time_slot", "day_of_week"),
    smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, object]]:
    global_rates = train_df[target_cols].mean()
    mean_col_map = {col: f"ctx_mean__{col}" for col in target_cols}
    context_mean = (
        train_df.groupby(list(context_cols))[list(target_cols)]
        .mean()
        .reset_index()
        .rename(columns=mean_col_map)
    )
    context_count = train_df.groupby(list(context_cols)).size().reset_index(name="ctx_count")
    lookup = context_mean.merge(context_count, on=list(context_cols), how="left")

    prior_cols = [f"prior__{col}" for col in target_cols]

    def attach(df: pd.DataFrame) -> pd.DataFrame:
        part = df.merge(lookup, on=list(context_cols), how="left")
        count = part["ctx_count"].fillna(0.0).to_numpy().reshape(-1, 1)
        mean_matrix = np.column_stack(
            [part[mean_col_map[col]].fillna(global_rates[col]).to_numpy() for col in target_cols]
        )
        global_matrix = np.tile(global_rates.to_numpy(), (len(part), 1))
        smoothed = (mean_matrix * count + global_matrix * smoothing) / (count + smoothing)
        prior_df = pd.DataFrame(smoothed, columns=prior_cols, index=part.index)
        base_cols = [col for col in df.columns]
        return pd.concat([part[base_cols].reset_index(drop=True), prior_df.reset_index(drop=True)], axis=1)

    metadata = {
        "context_cols": list(context_cols),
        "smoothing": smoothing,
        "global_rates": global_rates.to_dict(),
        "lookup": lookup,
        "mean_col_map": mean_col_map,
        "prior_cols": prior_cols,
        "target_cols": list(target_cols),
    }
    return attach(train_df), attach(test_df), prior_cols, metadata


def fill_numeric_nans(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    fill_values: Dict[str, float] = {}
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(train_df[col])]
    for col in numeric_cols:
        median = float(train_df[col].median()) if train_df[col].notna().any() else 0.0
        fill_values[col] = median
        train_df[col] = train_df[col].fillna(median)
        test_df[col] = test_df[col].fillna(median)
    return train_df, test_df, fill_values


def train_baseline_random_forest(train_df: pd.DataFrame, target_cols: Sequence[str], feature_cols: Sequence[str]) -> MultiOutputClassifier:
    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=220,
            max_depth=20,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        n_jobs=-1,
    )
    model.fit(train_df[list(feature_cols)], train_df[list(target_cols)])
    return model


def train_improved_logistic(train_df: pd.DataFrame, target_cols: Sequence[str], feature_cols: Sequence[str]) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        C=1.5,
                        max_iter=1000,
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_df[list(feature_cols)], train_df[list(target_cols)])
    return model


def train_improved_extra_trees(train_df: pd.DataFrame, target_cols: Sequence[str], feature_cols: Sequence[str]) -> MultiOutputClassifier:
    model = MultiOutputClassifier(
        ExtraTreesClassifier(
            n_estimators=320,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        n_jobs=-1,
    )
    model.fit(train_df[list(feature_cols)], train_df[list(target_cols)])
    return model


def predict_proba_matrix(model: object, X: pd.DataFrame) -> np.ndarray:
    if isinstance(model, Pipeline):
        return model.predict_proba(X)
    if hasattr(model, "estimators_"):
        return np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def choose_top_item(
    customer_id: float | None,
    category: str,
    order_type: str,
    time_slot: int,
    customer_top_item_map: Dict[Tuple[float, str], str],
    context_top_item_map: Dict[Tuple[str, str, int], str],
    global_top_item_map: Dict[str, str],
) -> str | None:
    if customer_id is not None:
        customer_key = (float(customer_id), category)
        if customer_key in customer_top_item_map:
            return customer_top_item_map[customer_key]
    context_key = (category, order_type, time_slot)
    if context_key in context_top_item_map:
        return context_top_item_map[context_key]
    return global_top_item_map.get(category)


def fit_context_prior_from_full_data(order_df: pd.DataFrame, target_cols: Sequence[str], smoothing: float = 20.0) -> Dict[str, object]:
    full_df, _, prior_cols, metadata = add_context_priors(order_df.copy(), order_df.copy(), target_cols, smoothing=smoothing)
    metadata["lookup"] = metadata["lookup"]
    metadata["prior_cols"] = prior_cols
    metadata["full_df_columns"] = full_df.columns.tolist()
    return metadata


def apply_context_prior_for_inference(base_df: pd.DataFrame, metadata: Dict[str, object]) -> pd.DataFrame:
    target_cols = metadata["target_cols"]
    context_cols = metadata["context_cols"]
    global_rates = metadata["global_rates"]
    smoothing = metadata["smoothing"]
    lookup = metadata["lookup"]
    mean_col_map = metadata["mean_col_map"]
    prior_cols = metadata["prior_cols"]

    global_rate_series = pd.Series(global_rates)
    part = base_df.merge(lookup, on=context_cols, how="left")
    count = part["ctx_count"].fillna(0.0).to_numpy().reshape(-1, 1)
    mean_matrix = np.column_stack(
        [part[mean_col_map[col]].fillna(global_rate_series[col]).to_numpy() for col in target_cols]
    )
    global_matrix = np.tile(global_rate_series.to_numpy(), (len(part), 1))
    smoothed = (mean_matrix * count + global_matrix * smoothing) / (count + smoothing)
    prior_df = pd.DataFrame(smoothed, columns=prior_cols, index=part.index)
    base_cols = list(base_df.columns)
    return pd.concat([part[base_cols].reset_index(drop=True), prior_df.reset_index(drop=True)], axis=1)


def build_single_inference_row(
    prepared: PreparedData,
    fill_values: Dict[str, float],
    context_prior_metadata: Dict[str, object],
    month: int,
    hour: int,
    day_of_week: int,
    order_type: str,
    customer_id: float | None,
) -> pd.DataFrame:
    if not 0 <= hour <= 23:
        raise ValueError("hour must be in [0, 23]")
    if not 0 <= day_of_week <= 6:
        raise ValueError("day_of_week must be in [0, 6]")
    if not 1 <= month <= 12:
        raise ValueError("month must be in [1, 12]")
    if order_type not in {"collection", "delivery"}:
        raise ValueError("order_type must be 'collection' or 'delivery'")

    cid = None if customer_id is None or pd.isna(customer_id) else float(customer_id)
    total_orders_all = 0
    eligible_hist = prepared.order_history_df.iloc[0:0].copy()
    category_hist = prepared.order_df.iloc[0:0].copy()

    if cid is not None:
        total_orders_all = int((prepared.order_base_all["customer_id"] == cid).sum())
        eligible_hist = prepared.order_history_df[prepared.order_history_df["customer_id"] == cid].sort_values("created_at")
        category_hist = prepared.order_df[prepared.order_df["customer_id"] == cid].sort_values("created_at")

    cum_orders = float(len(eligible_hist))
    cum_avg_spend = float(eligible_hist["order_total"].mean()) if len(eligible_hist) > 0 else fill_values.get("cum_avg_spend", 0.0)
    prev_total = float(eligible_hist["order_total"].iloc[-1]) if len(eligible_hist) > 0 else fill_values.get("prev_total", 0.0)
    prev_items = float(eligible_hist["total_items"].iloc[-1]) if len(eligible_hist) > 0 else fill_values.get("prev_items", 0.0)
    avg_spend_last3 = (
        float(eligible_hist["order_total"].tail(3).mean()) if len(eligible_hist) > 0 else fill_values.get("avg_spend_last3", 0.0)
    )
    avg_items_last3 = (
        float(eligible_hist["total_items"].tail(3).mean()) if len(eligible_hist) > 0 else fill_values.get("avg_items_last3", 0.0)
    )
    days_since_prev = float(fill_values.get("days_since_prev", 0.0))

    time_slot = get_time_slot(hour)
    row: Dict[str, float | int | str] = {
        "order_type": order_type,
        "time_slot": time_slot,
        "day_of_week": day_of_week,
        "hour": hour,
        "month": month,
        "is_weekend": int(day_of_week >= 5),
        "order_type_enc": int(order_type == "delivery"),
        "cum_orders": cum_orders,
        "is_new": int(cum_orders == 0),
        "cust_seg": customer_segment(cum_orders),
        "cum_avg_spend": cum_avg_spend,
        "prev_total": prev_total,
        "prev_items": prev_items,
        "avg_spend_last3": avg_spend_last3,
        "avg_items_last3": avg_items_last3,
        "days_since_prev": days_since_prev,
        "total_orders_all": total_orders_all,
        "eligible_orders_count": len(eligible_hist),
        "category_orders_count": len(category_hist),
    }

    row.update(cyclical_features(pd.Series([hour]), 24, "hour").iloc[0].to_dict())
    row.update(cyclical_features(pd.Series([day_of_week]), 7, "dow").iloc[0].to_dict())
    row.update(cyclical_features(pd.Series([month]), 12, "month").iloc[0].to_dict())

    for seg_value in range(4):
        row[f"cust_seg_{seg_value}"] = int(row["cust_seg"] == seg_value)
    for slot_value in range(5):
        row[f"time_slot_{slot_value}"] = int(time_slot == slot_value)
    for dow_value in range(7):
        row[f"dow_onehot_{dow_value}"] = int(day_of_week == dow_value)

    cat_matrix = None
    share_matrix = None
    if len(category_hist) > 0:
        target_cols = [f"target__{cat}" for cat in prepared.all_categories]
        share_cols = [f"revshare__{cat}" for cat in prepared.all_categories]
        cat_matrix = category_hist[target_cols]
        share_matrix = category_hist[share_cols]

    for cat in prepared.all_categories:
        target_col = f"target__{cat}"
        share_col = f"revshare__{cat}"
        hist_col = f"hist__{cat}"
        recent_col = f"recent3__{cat}"
        ewm_col = f"ewm__{cat}"
        same_type_col = f"same_type__{cat}"
        share_ewm_col = f"share_ewm__{cat}"
        weighted_col = f"weighted__{cat}"

        if cat_matrix is None:
            hist_value = recent_value = ewm_value = same_type_value = share_ewm_value = 0.0
        else:
            hist_series = cat_matrix[target_col]
            hist_value = float(hist_series.mean())
            recent_value = float(hist_series.tail(3).mean())
            ewm_value = float(hist_series.ewm(alpha=0.5, adjust=False).mean().iloc[-1])
            same_type_hist = category_hist.loc[category_hist["order_type"] == order_type, target_col]
            same_type_value = float(same_type_hist.mean()) if len(same_type_hist) > 0 else 0.0
            share_ewm_value = float(share_matrix[share_col].ewm(alpha=0.4, adjust=False).mean().iloc[-1])

        row[hist_col] = hist_value
        row[recent_col] = recent_value
        row[ewm_col] = ewm_value
        row[same_type_col] = same_type_value
        row[share_ewm_col] = share_ewm_value
        row[weighted_col] = (
            0.35 * hist_value
            + 0.25 * recent_value
            + 0.20 * ewm_value
            + 0.10 * same_type_value
            + 0.10 * share_ewm_value
        )

    base_df = pd.DataFrame([row])
    with_priors = apply_context_prior_for_inference(base_df, context_prior_metadata)
    for feature_name, fill_value in fill_values.items():
        if feature_name not in with_priors.columns:
            continue
        with_priors[feature_name] = with_priors[feature_name].fillna(fill_value)
    return with_priors


def train_and_evaluate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = load_and_prepare_data()

    train_df, test_df = split_train_test(prepared.order_df)

    baseline_train, baseline_test = train_df.copy(), test_df.copy()
    baseline_train, baseline_test, baseline_fill_values = fill_numeric_nans(
        baseline_train,
        baseline_test,
        prepared.baseline_feature_cols,
    )
    baseline_model = train_baseline_random_forest(baseline_train, prepared.target_cols, prepared.baseline_feature_cols)
    baseline_scores = predict_proba_matrix(baseline_model, baseline_test[prepared.baseline_feature_cols])
    baseline_metrics = evaluate_scores(baseline_test[prepared.target_cols].to_numpy(), baseline_scores)

    _, test_with_prior, prior_cols, _ = add_context_priors(
        train_df.copy(),
        test_df.copy(),
        prepared.target_cols,
    )
    context_prior_scores = test_with_prior[prior_cols].to_numpy()
    blended_scores = ((1.0 - PRIOR_BLEND_WEIGHT) * baseline_scores) + (PRIOR_BLEND_WEIGHT * context_prior_scores)
    context_prior_metrics = evaluate_scores(baseline_test[prepared.target_cols].to_numpy(), context_prior_scores)
    blended_metrics = evaluate_scores(baseline_test[prepared.target_cols].to_numpy(), blended_scores)

    results = {
        "baseline_rf": baseline_metrics,
        "context_prior_only": context_prior_metrics,
        f"baseline_rf_plus_context_prior_{PRIOR_BLEND_WEIGHT:.2f}": blended_metrics,
    }

    final_context_prior_metadata = fit_context_prior_from_full_data(prepared.order_df, prepared.target_cols)
    full_train = prepared.order_df.copy()
    final_feature_cols = prepared.baseline_feature_cols
    full_train, _, final_fill_values = fill_numeric_nans(full_train, full_train.copy(), final_feature_cols)
    final_rf_model = train_baseline_random_forest(full_train, prepared.target_cols, final_feature_cols)

    artifact = {
        "all_categories": prepared.all_categories,
        "target_cols": prepared.target_cols,
        "feature_cols": final_feature_cols,
        "default_month": int(train_df["month"].mode().iloc[0]),
        "fill_values": final_fill_values,
        "context_prior_metadata": final_context_prior_metadata,
        "rf_model": final_rf_model,
        "prior_blend_weight": PRIOR_BLEND_WEIGHT,
        "customer_top_item_map": prepared.customer_top_item_map,
        "context_top_item_map": prepared.context_top_item_map,
        "global_top_item_map": prepared.global_top_item_map,
        "order_base_all": prepared.order_base_all[["order_id", "customer_id", "order_type", "created_at", "order_total", "total_items"]].copy(),
        "order_history_df": prepared.order_history_df[["order_id", "customer_id", "order_type", "created_at", "order_total", "total_items"]].copy(),
        "order_df_history": prepared.order_df[["order_id", "customer_id", "order_type", "created_at", "order_total", "total_items", "categories"] + prepared.target_cols + [f"revshare__{cat}" for cat in prepared.all_categories]].copy(),
        "results": results,
    }

    artifact_path = output_dir / "category_recommender_v2.joblib"
    results_path = output_dir / "category_recommender_v2_results.joblib"
    joblib.dump(artifact, artifact_path)
    joblib.dump(results, results_path)

    return {
        "artifact_path": str(artifact_path),
        "results_path": str(results_path),
        "results": results,
    }


def load_artifact(path: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR / "category_recommender_v2.joblib") -> Dict[str, object]:
    return joblib.load(path)


def summarize_customer_history(artifact: Dict[str, object], customer_id: float | None) -> Dict[str, float]:
    if customer_id is None or pd.isna(customer_id):
        return {
            "customer_found": False,
            "total_orders_all": 0,
            "eligible_orders_count": 0,
            "category_orders_count": 0,
            "coverage": 0.0,
        }

    cid = float(customer_id)
    all_hist = artifact["order_base_all"][artifact["order_base_all"]["customer_id"] == cid]
    eligible_hist = artifact["order_history_df"][artifact["order_history_df"]["customer_id"] == cid]
    category_hist = artifact["order_df_history"][artifact["order_df_history"]["customer_id"] == cid]
    eligible_orders_count = int(len(eligible_hist))
    category_orders_count = int(len(category_hist))
    return {
        "customer_found": bool(len(all_hist) > 0),
        "total_orders_all": int(len(all_hist)),
        "eligible_orders_count": eligible_orders_count,
        "category_orders_count": category_orders_count,
        "coverage": (category_orders_count / eligible_orders_count) if eligible_orders_count > 0 else 0.0,
    }


def recommend(
    artifact: Dict[str, object],
    hour: int,
    day_of_week: int,
    order_type: str,
    customer_id: float | None = None,
    top_n: int = 5,
    month: int | None = None,
) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    if month is None:
        month = int(artifact["default_month"])

    prepared_like = PreparedData(
        order_base_all=artifact["order_base_all"],
        order_history_df=artifact["order_history_df"],
        order_df=artifact["order_df_history"],
        all_categories=artifact["all_categories"],
        target_cols=artifact["target_cols"],
        baseline_feature_cols=[],
        improved_feature_cols=[],
        customer_top_item_map=artifact["customer_top_item_map"],
        context_top_item_map=artifact["context_top_item_map"],
        global_top_item_map=artifact["global_top_item_map"],
    )

    row = build_single_inference_row(
        prepared=prepared_like,
        fill_values=artifact["fill_values"],
        context_prior_metadata=artifact["context_prior_metadata"],
        month=int(month),
        hour=hour,
        day_of_week=day_of_week,
        order_type=order_type,
        customer_id=customer_id,
    )
    feature_cols = artifact["feature_cols"]
    rf_scores = predict_proba_matrix(artifact["rf_model"], row[feature_cols])
    prior_cols = artifact["context_prior_metadata"]["prior_cols"]
    prior_scores = row[prior_cols].to_numpy()
    prior_weight = float(artifact["prior_blend_weight"])
    final_scores = ((1.0 - prior_weight) * rf_scores) + (prior_weight * prior_scores)

    categories = artifact["all_categories"]
    time_slot = get_time_slot(hour)
    result = pd.DataFrame(
        {
            "category": categories,
            "prob": final_scores.flatten(),
        }
    ).sort_values("prob", ascending=False).head(top_n).reset_index(drop=True)
    result.index += 1
    result["top_item"] = result["category"].map(
        lambda category: choose_top_item(
            customer_id=customer_id,
            category=category,
            order_type=order_type,
            time_slot=time_slot,
            customer_top_item_map=artifact["customer_top_item_map"],
            context_top_item_map=artifact["context_top_item_map"],
            global_top_item_map=artifact["global_top_item_map"],
        )
    )
    result["prob_pct"] = (result["prob"] * 100).round(1).astype(str) + "%"
    return result[["category", "top_item", "prob_pct", "prob"]]


def print_metrics(results: Dict[str, Dict[str, float]]) -> None:
    for model_name, metrics in results.items():
        print("=" * 90)
        print(model_name)
        for key, value in metrics.items():
            print(f"{key:<14} {value:.4f}")


from pathlib import Path

BASE_DIR = Path.cwd()
ORDER_ITEM_PATH = BASE_DIR / 'order_item_final_best.csv'
ORDER_PATH = BASE_DIR / 'order.csv'
DEFAULT_OUTPUT_DIR = BASE_DIR / 'models' / 'category_recommender_v2'
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('BASE_DIR          =', BASE_DIR)
print('ORDER_ITEM_PATH   =', ORDER_ITEM_PATH)
print('ORDER_PATH        =', ORDER_PATH)
print('DEFAULT_OUTPUT_DIR=', DEFAULT_OUTPUT_DIR)
print('PRIOR_BLEND_WEIGHT=', PRIOR_BLEND_WEIGHT)


prepared = load_and_prepare_data()

print('Tổng đơn thực tế                 :', prepared.order_base_all['order_id'].nunique())
print('Đơn hợp lệ bài toán              :', prepared.order_history_df['order_id'].nunique())
print('Đơn có item/category             :', prepared.order_df['order_id'].nunique())
print('Số category                      :', len(prepared.all_categories))
print('Coverage category history        :', f"{prepared.order_df['order_id'].nunique() / prepared.order_history_df['order_id'].nunique():.1%}")
print('Số target categories             :', len(prepared.target_cols))


summary = train_and_evaluate(output_dir=DEFAULT_OUTPUT_DIR)
print_metrics(summary['results'])

artifact_path = Path(summary['artifact_path'])
results_path = Path(summary['results_path'])

print('=' * 90)
print('artifact_path =', artifact_path)
print('results_path  =', results_path)


artifact = load_artifact(DEFAULT_OUTPUT_DIR / 'category_recommender_v2.joblib')
metrics_df = pd.DataFrame(artifact['results']).T
metrics_df


DAY_LABELS = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']

def recommend_category(
    hour: int,
    day_of_week: int,
    order_type: str,
    customer_id: float | None = None,
    top_n: int = 5,
    month: int | None = None,
):
    if month is None:
        month = artifact['default_month']

    history = summarize_customer_history(artifact, customer_id)
    result = recommend(
        artifact=artifact,
        hour=hour,
        day_of_week=day_of_week,
        order_type=order_type,
        customer_id=customer_id,
        top_n=top_n,
        month=month,
    )

    customer_label = f'ID {int(customer_id)}' if history['customer_found'] else 'Mới/Vãng lai'
    print(f'👤 Khách       : {customer_label}')
    print(f'🕐 Thời gian   : {hour}h | {DAY_LABELS[day_of_week]} | {order_type} | tháng {month}')
    print(f"📦 Đơn thực tế : {history['total_orders_all']} | Hợp lệ bài toán: {history['eligible_orders_count']} | Có category: {history['category_orders_count']}")
    print(f"🧩 Coverage    : {history['coverage']:.1%} đơn hợp lệ có category history")
    print()
    return result


print('=' * 55)
print('TEST 1: Khách loyal, bữa tối, delivery')
print('=' * 55)
sample_cid = (
    artifact['order_history_df']
    .dropna(subset=['customer_id'])
    .groupby('customer_id')
    .size()
    .idxmax()
)
result1 = recommend_category(hour=19, day_of_week=4, order_type='delivery', customer_id=sample_cid, top_n=5)
print(result1.to_string())


print()
print('=' * 55)
print('TEST 2: Khách mới, buổi trưa, collection')
print('=' * 55)
result2 = recommend_category(hour=12, day_of_week=1, order_type='collection', customer_id=None, top_n=5)
print(result2.to_string())


print()
print('=' * 55)
print('TEST 3: Khách cụ thể (ID=248), bữa tối, delivery')
print('=' * 55)
result3 = recommend_category(hour=19, day_of_week=4, order_type='delivery', customer_id=248.0, top_n=5)
print(result3.to_string())


