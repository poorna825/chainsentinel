"""
Rolling-baseline drift vs real performance. PRE-REGISTERED before running; do
not tune the window or threshold after seeing results.

Question: the fixed-baseline PSI (src/report.py) was above the "major" cutoff
at every step 35-49 and did not flag the step-43 collapse. Does a rolling
baseline do better?

Rule:
- Baseline for step t = steps t-3..t-1 (window=3).
- Metrics: median_psi, weighted_psi, prediction_psi.
- Per metric, alarm threshold = mean + 2*std (sample std, ddof=1) of that
  metric over steps 35-42. Alarm at step t if value > threshold.
- Success (per metric) = alarm at step 43 AND no alarm at any of steps 35-42.
  Overall pass = at least one metric succeeds; every metric is reported
  either way. Steps 44-49 are shown but not part of the success rule.

Known limitations (stated up front, they weaken the result):
1. The threshold is fit on steps 35-42 and then checked on those same steps,
   so "no false alarms at 35-42" is in-sample and weak - the threshold is
   built from exactly the values it is tested against.
2. Baselines for steps 35-37 include train steps 32-34, where the model's
   scores are in-sample (overconfident, since it was fit on those rows).
   That can inflate prediction_psi at 35-37 relative to later steps.

Run from the repo root:  python src/rolling_report.py
Writes results/rolling_report.csv and results/rolling_psi.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import REPO_ROOT, TEST_STEPS, feature_columns, load_elliptic, time_split
from drift import rolling_psi_over_time
from metrics import pr_auc_over_time
from model import train_random_forest

RESULTS_DIR = REPO_ROOT / "results"
WINDOW = 3
CALM_STEPS = range(35, 43)   # threshold is fit on these
BREAK_STEP = 43
METRICS = ["median_psi", "weighted_psi", "prediction_psi"]


def build_report():
    df, _ = load_elliptic()
    feats = feature_columns(df)
    train, _ = time_split(df)
    model = train_random_forest(train, feats)
    importances = dict(zip(feats, model.feature_importances_))
    steps = list(TEST_STEPS)

    roll = pd.DataFrame(rolling_psi_over_time(model, df, feats, steps, importances, window=WINDOW))
    perf = pd.DataFrame(pr_auc_over_time(model, df, feats, steps))
    return roll.merge(perf, on="time_step")


def check_rule(report):
    """Apply the pre-registered rule. Returns {metric: dict(threshold, alarms, success)}."""
    calm = report[report["time_step"].isin(CALM_STEPS)]
    out = {}
    for m in METRICS:
        threshold = calm[m].mean() + 2 * calm[m].std(ddof=1)
        alarm_steps = report.loc[report[m] > threshold, "time_step"].tolist()
        false_alarms = [s for s in alarm_steps if s in CALM_STEPS]
        out[m] = {
            "threshold": threshold,
            "alarm_steps": alarm_steps,
            "success": BREAK_STEP in alarm_steps and not false_alarms,
        }
    return out


def plot(report, verdict, path):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                         gridspec_kw={"height_ratios": [1, 1.4]})
    steps = report["time_step"]
    ink, muted = "#222222", "#777777"
    colors = {"median_psi": "#2a6fb0", "weighted_psi": "#c0612b", "prediction_psi": "#4d8f5a"}
    markers = {"median_psi": "o", "weighted_psi": "s", "prediction_psi": "^"}

    ax_top.plot(steps, report["pr_auc"], color=ink, lw=2, marker="o", ms=5)
    ax_top.set_ylabel("PR-AUC (illicit)")
    ax_top.set_title("Real performance vs rolling-baseline drift (window=3)", loc="left")

    for m in METRICS:
        ax_bot.plot(steps, report[m], color=colors[m], lw=2, marker=markers[m], ms=5, label=m)
        ax_bot.axhline(verdict[m]["threshold"], color=colors[m], lw=1, ls="--", alpha=0.6)
    ax_bot.set_ylabel("PSI vs previous 3 steps")
    ax_bot.set_xlabel("time step")
    ax_bot.legend(frameon=False, loc="center right")

    for ax in (ax_top, ax_bot):
        ax.axvline(BREAK_STEP, color="#b22222", lw=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        ax.set_axisbelow(True)
    ax_top.annotate("step 43: shutdown", (BREAK_STEP, 0.55), xytext=(6, 0),
                    textcoords="offset points", color="#b22222")
    ax_bot.text(0.995, 0.22, "dashed = alarm threshold\n(mean+2std over steps 35-42)",
                transform=ax_bot.transAxes, ha="right", va="top", fontsize=8, color=muted)
    ax_bot.set_xticks(list(steps))

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    report = build_report()
    verdict = check_rule(report)

    RESULTS_DIR.mkdir(exist_ok=True)
    report.to_csv(RESULTS_DIR / "rolling_report.csv", index=False)
    plot(report, verdict, RESULTS_DIR / "rolling_psi.png")

    print(report[["time_step", "pr_auc", "n_illicit"] + METRICS].round(3).to_string(index=False))
    print()
    for m, v in verdict.items():
        print(f"{m:15s} threshold={v['threshold']:.3f}  alarms at {v['alarm_steps']}  "
              f"-> {'PASS' if v['success'] else 'FAIL'}")
    print("\nOVERALL:", "PASS" if any(v["success"] for v in verdict.values()) else "FAIL")


if __name__ == "__main__":
    main()
