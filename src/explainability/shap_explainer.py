import shap
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.logger import get_logger

logger = get_logger("shap_explainer")

ROOT_DIR    = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT_DIR / "outputs" / "figures"

# Task-specific label mapping
TASK_LABELS = {
    "fraud"  : {"positive": "Fraud",   "negative": "Legit"},
    "credit" : {"positive": "Default", "negative": "No-Default"},
}


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"   Saved: {path}")


def _get_tree_model(model) -> object:
    """Extract underlying tree model from Pipeline or return as-is."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, LogisticRegression):
                raise ValueError(
                    "Stacking meta-learner (LogisticRegression) is not supported "
                    "by TreeExplainer. Pass a base tree model (tuned_xgb/lgb/cat) instead."
                )
    return model


def get_explainer(model) -> shap.TreeExplainer:
    """Create TreeExplainer. Raises if model is not tree-based."""
    model = _get_tree_model(model)
    logger.info(f"Creating TreeExplainer for {type(model).__name__}...")
    return shap.TreeExplainer(model)


def get_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
    n_samples: int = 2000,
) -> tuple:
    """Sample X and compute SHAP values."""
    X_sample = X.sample(n=min(n_samples, len(X)), random_state=42)
    logger.info(f"Computing SHAP values on {len(X_sample):,} samples...")
    shap_values = explainer(X_sample, check_additivity=False)
    logger.info(f"   SHAP values shape: {shap_values.values.shape}")
    logger.info(f"   Base value       : {float(explainer.expected_value):.4f}")
    return shap_values, X_sample


# ── Global SHAP ───────────────────────────────────────────────────────────────

def plot_global_bar(
    shap_values: shap.Explanation,
    task: str = "fraud",
    max_display: int = 20,
) -> None:
    """Mean absolute SHAP — overall feature importance ranking."""
    logger.info("Global SHAP — Bar plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=max_display, show=False, ax=ax)
    ax.set_title(
        f"Global SHAP Feature Importance — {task.upper()} Model",
        fontsize=13, fontweight="bold", pad=12,
    )
    _save(fig, FIGURES_DIR / task / "shap" / "01_global_bar.png")


def plot_global_beeswarm(
    shap_values: shap.Explanation,
    task: str = "fraud",
    max_display: int = 20,
) -> None:
    """Beeswarm — feature importance + direction + distribution."""
    logger.info("Global SHAP — Beeswarm plot...")
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(
        f"Global SHAP Summary (Beeswarm) — {task.upper()} Model",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig = plt.gcf()
    _save(fig, FIGURES_DIR / task / "shap" / "02_global_beeswarm.png")


def plot_global_heatmap(
    shap_values: shap.Explanation,
    task: str = "fraud",
    max_display: int = 20,
) -> None:
    """Heatmap — sample × feature SHAP matrix."""
    logger.info("Global SHAP — Heatmap...")
    shap.plots.heatmap(shap_values, max_display=max_display, show=False)
    plt.title(
        f"Global SHAP Heatmap — {task.upper()} Model",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig = plt.gcf()
    _save(fig, FIGURES_DIR / task / "shap" / "03_global_heatmap.png")


def plot_mean_abs_shap(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    task: str = "fraud",
    top_n: int = 20,
) -> None:
    """Custom mean |SHAP| bar — cleaner than shap.plots.bar for reports."""
    logger.info("Global SHAP — Mean |SHAP| custom bar...")
    mean_abs = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=X_sample.columns,
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#E24B4A" if "FE_" in c else "#378ADD" for c in mean_abs.index]
    ax.barh(mean_abs.index, mean_abs.values, color=colors, edgecolor="none")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(
        f"Top {top_n} Features — Mean |SHAP| | {task.upper()}",
        fontsize=13, fontweight="bold",
    )
    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#E24B4A", label="Engineered (FE_)"),
        Patch(color="#378ADD", label="Original"),
    ]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    _save(fig, FIGURES_DIR / task / "shap" / "07_mean_abs_shap_custom.png")


# ── Local SHAP ────────────────────────────────────────────────────────────────

def plot_local_waterfall(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    idx: int,
    task: str = "fraud",
    label: str = "",
    max_display: int = 15,
) -> None:
    """Waterfall plot — single sample explanation."""
    logger.info(f"Local SHAP — Waterfall (idx={idx}, {label})...")
    shap.plots.waterfall(shap_values[idx], max_display=max_display, show=False)
    plt.title(
        f"Local SHAP Waterfall — Sample #{idx} ({label}) | {task.upper()}",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig = plt.gcf()
    fname = f"04_local_waterfall_{label.lower().replace(' ', '_')}_idx{idx}.png"
    _save(fig, FIGURES_DIR / task / "shap" / fname)


def plot_local_force(
    explainer: shap.TreeExplainer,
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    idx: int,
    task: str = "fraud",
    label: str = "",
) -> None:
    """Force plot — single sample push/pull visualization."""
    logger.info(f"Local SHAP — Force plot (idx={idx}, {label})...")
    shap.initjs()
    shap.force_plot(
        explainer.expected_value,
        shap_values.values[idx],
        X_sample.iloc[idx],
        show=False,
        matplotlib=True,
        figsize=(18, 4),
    )
    fig = plt.gcf()
    plt.title(
        f"Local SHAP Force — Sample #{idx} ({label}) | {task.upper()}",
        fontsize=12, fontweight="bold",
    )
    fname = f"05_local_force_{label.lower().replace(' ', '_')}_idx{idx}.png"
    _save(fig, FIGURES_DIR / task / "shap" / fname)


# ── Interaction SHAP ──────────────────────────────────────────────────────────

def plot_dependence(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    feature: str,
    interaction_feature: str = None,
    task: str = "fraud",
) -> None:
    """Dependence plot — feature value vs SHAP value, colored by interaction."""
    logger.info(f"SHAP Dependence — {feature} × {interaction_feature}...")

    if feature not in X_sample.columns:
        logger.info(f"   Feature {feature} not found, skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_idx  = list(X_sample.columns).index(feature)
    shap_feat = shap_values.values[:, feat_idx]

    if interaction_feature and interaction_feature in X_sample.columns:
        inter_idx  = list(X_sample.columns).index(interaction_feature)
        color_vals = X_sample[interaction_feature].values
        sc = ax.scatter(
            X_sample[feature], shap_feat,
            c=color_vals, cmap="coolwarm",
            alpha=0.6, edgecolors="none", s=20,
        )
        plt.colorbar(sc, ax=ax, label=interaction_feature)
    else:
        ax.scatter(
            X_sample[feature], shap_feat,
            alpha=0.5, edgecolors="none", s=20, color="steelblue",
        )

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(
        f"SHAP Dependence: {feature}"
        + (f" × {interaction_feature}" if interaction_feature else ""),
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fname = f"06_dependence_{feature}.png"
    _save(fig, FIGURES_DIR / task / "shap" / fname)


def plot_positive_negative_shap(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    task: str = "fraud",
    top_n: int = 15,
) -> None:
    """Split mean SHAP into risk-increasing vs risk-decreasing features."""
    logger.info("Global SHAP — Positive vs Negative split...")

    labels    = TASK_LABELS.get(task, {"positive": "Positive", "negative": "Negative"})
    mean_shap = pd.Series(
        shap_values.values.mean(axis=0),
        index=X_sample.columns,
    ).sort_values()

    pos = mean_shap[mean_shap > 0].tail(top_n)
    neg = mean_shap[mean_shap < 0].head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].barh(neg.index, neg.values, color="#378ADD", edgecolor="none")
    axes[0].set_title(
        f"Risk-Decreasing Features\n(↓ {labels['positive']} probability)",
        fontweight="bold"
    )
    axes[0].set_xlabel("Mean SHAP value")
    axes[0].axvline(0, color="black", lw=0.8)

    axes[1].barh(pos.index, pos.values, color="#E24B4A", edgecolor="none")
    axes[1].set_title(
        f"Risk-Increasing Features\n(↑ {labels['positive']} probability)",
        fontweight="bold"
    )
    axes[1].set_xlabel("Mean SHAP value")
    axes[1].axvline(0, color="black", lw=0.8)

    plt.suptitle(
        f"SHAP Direction Analysis — {task.upper()} Model",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, FIGURES_DIR / task / "shap" / "08_positive_negative_shap.png")


def plot_fe_vs_raw_contribution(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    task: str = "fraud",
) -> None:
    """Compare total SHAP contribution: engineered features vs raw features."""
    logger.info("SHAP — FE vs Raw contribution comparison...")

    mean_abs = pd.Series(
        np.abs(shap_values.values).mean(axis=0),
        index=X_sample.columns,
    )
    fe_total  = mean_abs[[c for c in mean_abs.index if c.startswith("FE_")]].sum()
    raw_total = mean_abs[[c for c in mean_abs.index if not c.startswith("FE_")]].sum()
    total     = fe_total + raw_total

    logger.info(f"   FE features contribution  : {fe_total:.4f} ({fe_total/total*100:.1f}%)")
    logger.info(f"   Raw features contribution : {raw_total:.4f} ({raw_total/total*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].pie(
        [fe_total, raw_total],
        labels=[
            f"Engineered (FE_)\n{fe_total/total*100:.1f}%",
            f"Original\n{raw_total/total*100:.1f}%",
        ],
        colors=["#E24B4A", "#378ADD"],
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11},
    )
    axes[0].set_title(
        "Total SHAP Contribution\nEngineered vs Original Features",
        fontweight="bold",
    )

    # Top FE features
    fe_features = mean_abs[
        [c for c in mean_abs.index if c.startswith("FE_")]
    ].sort_values(ascending=True).tail(15)
    axes[1].barh(fe_features.index, fe_features.values,
                 color="#E24B4A", edgecolor="none")
    axes[1].set_xlabel("Mean |SHAP value|")
    axes[1].set_title("Top Engineered Features — SHAP Importance", fontweight="bold")

    plt.suptitle(
        f"Feature Engineering Impact — {task.upper()} Model",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, FIGURES_DIR / task / "shap" / "09_fe_vs_raw_contribution.png")


# ── Full Analysis ─────────────────────────────────────────────────────────────

def run_full_shap_analysis(
    model,
    X: pd.DataFrame,
    y: pd.Series = None,
    task: str = "fraud",
    n_samples: int = 2000,
    top_k_dependence: int = 3,
) -> None:
    """Run complete SHAP analysis and save all figures for the given task."""
    logger.info("=" * 60)
    logger.info(f"SHAP FULL ANALYSIS — {task.upper()}")
    logger.info("=" * 60)

    labels    = TASK_LABELS.get(task, {"positive": "Positive", "negative": "Negative"})
    explainer = get_explainer(model)
    shap_values, X_sample = get_shap_values(explainer, X, n_samples)

    # ── Global ────────────────────────────────────────────────
    plot_global_bar(shap_values, task)
    plot_global_beeswarm(shap_values, task)
    plot_global_heatmap(shap_values, task)
    plot_mean_abs_shap(shap_values, X_sample, task)
    plot_positive_negative_shap(shap_values, X_sample, task)
    plot_fe_vs_raw_contribution(shap_values, X_sample, task)

    # ── Local — task-aware labels ─────────────────────────────
    if y is not None:
        y_sample = y.loc[X_sample.index]

        # MODIFIED: Find sample with highest probability for positive class
        pos_mask = y_sample == 1
        if pos_mask.any():
            pos_probs = model.predict_proba(X_sample[pos_mask])[:, 1]
            # Get index of maximum probability within the positive samples
            pos_idx = int(np.where(pos_mask.values)[0][pos_probs.argmax()])
            
            plot_local_waterfall(
                shap_values, X_sample, pos_idx, task,
                label=f"High-Risk {labels['positive']}"
            )
            try:
                plot_local_force(
                    explainer, shap_values, X_sample, pos_idx, task,
                    label=f"High-Risk {labels['positive']}"
                )
            except Exception as e:
                logger.info(f"   Force plot skipped: {e}")

        # Find sample with lowest probability for positive class (most negative)
        neg_mask = y_sample == 0
        if neg_mask.any():
            neg_probs = model.predict_proba(X_sample[neg_mask])[:, 1]
            neg_idx   = int(np.where(neg_mask.values)[0][neg_probs.argmin()])
            
            plot_local_waterfall(
                shap_values, X_sample, neg_idx, task,
                label=f"Clear {labels['negative']}"
            )
    else:
        plot_local_waterfall(shap_values, X_sample, 0, task, label="Sample")

    # ── Dependence — top K features ───────────────────────────
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features  = X_sample.columns[
        np.argsort(mean_abs_shap)[-top_k_dependence:][::-1]
    ].tolist()

    logger.info(f"Top {top_k_dependence} features for dependence: {top_features}")
    for i, feat in enumerate(top_features):
        inter = top_features[i + 1] if i + 1 < len(top_features) else None
        plot_dependence(shap_values, X_sample, feat, inter, task)

    logger.info("=" * 60)
    logger.info(f"SHAP COMPLETE — figures: outputs/figures/{task}/shap/")
    logger.info("=" * 60)


# ── Quick single-sample explanation ──────────────────────────────────────────

def explain_single(
    model,
    X_single: pd.DataFrame,
    task: str = "fraud",
    label: str = "single",
) -> dict:
    """Explain one prediction with SHAP — returns contribution dict."""
    labels    = TASK_LABELS.get(task, {"positive": "Positive", "negative": "Negative"})
    explainer = get_explainer(model)
    shap_values = explainer(X_single, check_additivity=False)

    contributions = pd.Series(
        shap_values.values[0],
        index=X_single.columns,
    ).sort_values(key=abs, ascending=False)

    base_value = float(explainer.expected_value)
    pred_shap  = base_value + shap_values.values[0].sum()

    logger.info(f"\nSingle explanation ({label}):")
    logger.info(f"   Base value (avg prediction) : {base_value:.4f}")
    logger.info(f"   SHAP prediction             : {pred_shap:.4f}")
    logger.info(f"\n   Top 10 contributors:")
    for feat, val in contributions.head(10).items():
        direction = f"↑ {labels['positive']}" if val > 0 else f"↓ {labels['positive']}"
        logger.info(f"    {feat:35s} {val:+.4f}  {direction}")

    plot_local_waterfall(shap_values, X_single, 0, task, label=label)

    return {
        "base_value"   : base_value,
        "shap_pred"    : pred_shap,
        "contributions": contributions.to_dict(),
    }