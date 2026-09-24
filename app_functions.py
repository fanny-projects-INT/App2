import ast
import colorsys
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap


# =============================================================================
# STYLE / COLORS
# =============================================================================
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "grid.alpha": 0.16,
    "legend.frameon": False,
})

# Protocol colors
YELLOW = "#F4D35E"
BLUE = "#8EC5FF"
RED = "#F29A8E"

# Dashboard palette
BLUE_MAIN = "#2563EB"
BLUE_FILL = "#60A5FA"
GREEN_MAIN = "#22C55E"
ORANGE_MAIN = "#F59E0B"
GRAY_MAIN = "#94A3B8"
RED_MAIN = "#EF4444"
PURPLE_MAIN = "#7C3AED"

COLORS = {
    "navy": "#1E293B",
    "gray": "#64748B",
    "grid": "#E2E8F0",
    "axis": "#CBD5E1",
    "blue_main": BLUE_MAIN,
    "blue_fill": BLUE_FILL,
    "green_main": GREEN_MAIN,
    "orange_main": ORANGE_MAIN,
    "gray_main": GRAY_MAIN,
    "red_main": RED_MAIN,
    "purple_main": PURPLE_MAIN,
}

PROTOCOL_COLORS = {
    1: (0.96, 0.88, 0.42),
    2: (0.56, 0.77, 1.00),
    3: (0.95, 0.63, 0.58),
    4: (0.50, 0.39, 0.60),
}

PROTOCOL_HEX = {
    1: YELLOW,
    2: BLUE,
    3: RED,
    4: "#80639A",
}

PROTOCOL_LABELS = {
    1: "Training 1",
    2: "Training 2",
    3: "Task",
    4: "Flexifliping",
}


# =============================================================================
# BASIC HELPERS
# =============================================================================
def ensure_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []


def ensure_array(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        try:
            return np.asarray(x)
        except Exception:
            return np.array([], dtype=float)
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return np.asarray(val)
        except Exception:
            return np.array([], dtype=float)
    return np.array([], dtype=float)


def flatten_nested_times(x):
    arr = ensure_array(x)

    if arr.size == 0:
        return []

    if len(arr) > 0 and isinstance(arr[0], (list, np.ndarray)):
        out = []
        for block in arr:
            if isinstance(block, np.ndarray):
                out.extend(block.tolist())
            else:
                out.extend(list(block))
        return out

    return arr.tolist()


def smooth_discrete_curve_fixed_range(
    x_vals,
    y_vals,
    x_min=1,
    x_max=25,
    sigma=1.0,
    points=900
):
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    x_grid = np.linspace(x_min, x_max, points)
    y_smooth = np.zeros_like(x_grid)

    for xi, yi in zip(x_vals, y_vals):
        y_smooth += yi * np.exp(-0.5 * ((x_grid - xi) / sigma) ** 2)

    dx = x_grid[1] - x_grid[0]
    area = y_smooth.sum() * dx

    if area > 0:
        y_smooth /= area

    return x_grid, y_smooth


def shade_color(base_rgb, p):
    if p is None or pd.isna(p):
        return (*base_rgb, 0.26)

    x = p / 0.30
    r, g, b = base_rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    new_l = 0.975 - 0.11 * x
    new_s = min(1, s + (0.10 + 0.10 * x))
    nr, ng, nb = colorsys.hls_to_rgb(h, new_l, new_s)
    return (nr, ng, nb, 0.24)


def parse_proba(p):
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    p = str(p)
    if "/" not in p:
        return None
    try:
        return float(p.split("/")[1])
    except Exception:
        return None


def protocol_name(proto):
    try:
        return PROTOCOL_LABELS.get(int(proto), f"Protocol {int(proto)}")
    except Exception:
        return "-"


def get_protocol_blocks(df_session):
    blocks = []
    start = None
    curr_p = None
    curr_proto = None

    for _, r in df_session.iterrows():
        s = r["SessionIndex"]
        p = r["Proba_val"]
        proto = r["Protocol"]

        if start is None:
            start, curr_p, curr_proto = s, p, proto
            continue

        if p != curr_p or proto != curr_proto:
            blocks.append((start, s - 1, curr_p, curr_proto))
            start, curr_p, curr_proto = s, p, proto

    if start is not None:
        blocks.append((start, df_session["SessionIndex"].max(), curr_p, curr_proto))

    return blocks


def style_axes(ax):
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])
    ax.tick_params(colors=COLORS["gray"])
    ax.yaxis.label.set_color(COLORS["navy"])
    ax.xaxis.label.set_color(COLORS["navy"])
    ax.title.set_color(COLORS["navy"])
    ax.grid(alpha=0.22, axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def add_proba_labels(ax, blocks, y_text, color=None, fontsize=8.3):
    if color is None:
        color = COLORS["gray"]

    for start, end, p, proto in blocks:
        if p is None:
            continue
        center = (start + end) / 2
        ax.text(
            center,
            y_text,
            f"{p:.2f}",
            ha="center",
            va="top",
            fontsize=fontsize,
            color=color,
            clip_on=False,
        )


def valid_bout_mask_from_row(row, target_len=None):
    mask = ensure_array(row.get("Correct Bouts")).astype(bool)
    if target_len is None:
        return mask
    if len(mask) == 0:
        return np.zeros(target_len, dtype=bool)
    if len(mask) >= target_len:
        return mask[:target_len]
    out = np.zeros(target_len, dtype=bool)
    out[:len(mask)] = mask
    return out


def count_valid_bouts(row):
    valid_bouts = ensure_array(row.get("Correct Bouts")).astype(bool)
    if len(valid_bouts) == 0:
        return 0
    return int(np.sum(valid_bouts))


def fit_exponential_trend_with_band(y_values):
    y = np.asarray(y_values, dtype=float)
    x = np.arange(len(y), dtype=float)

    mask = np.isfinite(y) & (y > 0)
    x_fit = x[mask]
    y_fit = y[mask]

    if len(y_fit) < 2:
        return None

    log_y = np.log(y_fit)

    model = LinearRegression()
    model.fit(x_fit.reshape(-1, 1), log_y)

    log_pred = model.predict(x_fit.reshape(-1, 1))
    residuals = log_y - log_pred
    sigma_log = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    a = float(np.exp(model.intercept_))
    b = float(model.coef_[0])

    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = a * np.exp(b * x_smooth)

    y_lower = y_smooth * np.exp(-sigma_log)
    y_upper = y_smooth * np.exp(+sigma_log)

    return {
        "x_smooth": x_smooth,
        "y_smooth": y_smooth,
        "y_lower": y_lower,
        "y_upper": y_upper,
    }


def prepare_mouse_dataframe(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Date_norm"] = df["Date"].dt.normalize()
    df["Mouse_ID"] = df["Mouse_ID"].astype(str)
    df["Version"] = df["Version"].astype(str)
    df["Protocol"] = pd.to_numeric(df["Protocol"], errors="coerce").astype("Int64")
    df["Proba_val"] = df["Probas"].apply(parse_proba)
    df = df.sort_values(["Mouse_ID", "Date", "Version"]).reset_index(drop=True)

    session_cmap = LinearSegmentedColormap.from_list(
        "session_cmap",
        [BLUE_MAIN, RED_MAIN]
    )

    return df, session_cmap


# =============================================================================
# CORE ANALYSIS
# =============================================================================
def compute_failures(row):
    timestamps = ensure_array(row.get("Timestamps"))
    bouts = ensure_array(row.get("Bout for Timestamps"))

    if len(timestamps) == 0 or len(bouts) == 0:
        return []

    rewarded = flatten_nested_times(row.get("Times Rewarded Licks", []))
    non_rewarded = flatten_nested_times(row.get("Times Non Rewarded Licks", []))

    failures = []
    for b in np.unique(bouts):
        mask = bouts == b
        t_in = timestamps[mask]

        if len(t_in) == 0:
            failures.append(0)
            continue

        r_in = [t for t in rewarded if t_in[0] <= t <= t_in[-1]]
        nr_in = [t for t in non_rewarded if t_in[0] <= t <= t_in[-1]]

        if len(nr_in) == 0:
            failures.append(0)
            continue

        if len(r_in) == 0:
            failures.append(len(nr_in))
        else:
            last_r = max(r_in)
            count = sum(1 for t in nr_in if t > last_r)
            failures.append(count)

    return failures


def count_reward_per_bout(row):
    timestamps = ensure_array(row.get("Timestamps"))
    bouts = ensure_array(row.get("Bout for Timestamps"))

    if len(timestamps) == 0 or len(bouts) == 0:
        return []

    rewarded = flatten_nested_times(row.get("Times Rewarded Licks", []))

    counts = []
    for b in np.unique(bouts):
        mask = bouts == b
        t_in = timestamps[mask]

        if len(t_in) == 0:
            counts.append(0)
            continue

        r_in = [t for t in rewarded if t_in[0] <= t <= t_in[-1]]
        counts.append(len(r_in))

    return counts


def count_licks(row, lick_type):
    column_map = {
        "rewarded": "Times Rewarded Licks",
        "non_rewarded": "Times Non Rewarded Licks",
        "invalid": "Times Invalid Licks",
    }
    column_name = column_map.get(lick_type)

    if not column_name:
        return 0

    data = ensure_array(row.get(column_name))
    if len(data) == 0:
        return 0

    if len(data) > 0 and isinstance(data[0], (list, np.ndarray)):
        return sum(len(x) for x in data)

    return len(data)


# =============================================================================
# OVERVIEW PLOTS
# =============================================================================
def plot_protocol_strip(df, mouse):
    df_mouse = df[df["Mouse_ID"] == mouse].copy()
    if df_mouse.empty:
        return None

    counts = (
        df_mouse["Protocol"]
        .dropna()
        .astype(int)
        .value_counts()
        .reindex([1, 2, 3, 4], fill_value=0)
    )

    total = counts.sum()
    if total == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 1.25))
    left = 0

    for proto in [1, 2, 3, 4]:
        width = counts[proto]
        if width <= 0:
            continue

        ax.barh([0], [width], left=left, color=PROTOCOL_HEX[proto], height=0.32)

        center = left + width / 2
        ax.text(
            center,
            0,
            f"{PROTOCOL_LABELS[proto]} ({int(width)})",
            ha="center",
            va="center",
            fontsize=9.5,
            color=COLORS["navy"],
        )
        left += width

    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(f"{mouse} - Session types", pad=8, color=COLORS["navy"])

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


def plot_bout_count_rewards(df, mouse):
    df_session = df[df["Mouse_ID"] == mouse].copy()
    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    df_session["SessionIndex"] = range(1, len(df_session) + 1)
    df_session["ValidBoutCount"] = df_session.apply(count_valid_bouts, axis=1)
    df_session["RewardedCount"] = df_session["Number of Rewarded Licks"].fillna(0)

    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for _, row in df_session.iterrows():
        s = row["SessionIndex"]
        proto = row["Protocol"]
        p = row["Proba_val"]
        base = PROTOCOL_COLORS.get(proto, (0.93, 0.93, 0.93))
        ax.axvspan(s - 0.5, s + 0.5, color=shade_color(base, p), zorder=1)

    blocks = get_protocol_blocks(df_session)

    ax.plot(
        df_session["SessionIndex"],
        df_session["ValidBoutCount"],
        marker="o",
        markersize=4.8,
        linewidth=2.25,
        color=COLORS["navy"],
        zorder=20,
        label="Valid bouts",
    )

    ax.plot(
        df_session["SessionIndex"],
        df_session["RewardedCount"],
        marker="s",
        markersize=4.6,
        linestyle="--",
        linewidth=2.1,
        color=GREEN_MAIN,
        zorder=25,
        label="Rewarded licks",
    )

    ymin_data = min(
        np.nanmin(df_session["ValidBoutCount"].to_numpy(dtype=float)),
        np.nanmin(df_session["RewardedCount"].to_numpy(dtype=float)),
    )
    ymax_data = max(
        np.nanmax(df_session["ValidBoutCount"].to_numpy(dtype=float)),
        np.nanmax(df_session["RewardedCount"].to_numpy(dtype=float)),
    )

    yrange = max(1.0, ymax_data - ymin_data)
    bottom_margin = 0.11 * yrange
    top_margin = 0.12 * yrange
    y_text = ymin_data - 0.048 * yrange

    ax.set_ylim(ymin_data - bottom_margin, ymax_data + top_margin)

    for _, row in df_session.iterrows():
        ax.text(
            row["SessionIndex"],
            row["RewardedCount"] + 0.016 * (ymax_data + top_margin),
            str(int(row["RewardedCount"])),
            ha="center",
            fontsize=8,
            color=GREEN_MAIN,
            zorder=50,
        )

    add_proba_labels(ax, blocks, y_text=y_text, color=COLORS["gray"], fontsize=8.3)

    ax.set_xlabel("Training sessions", color=COLORS["navy"])
    ax.set_ylabel("Counts", color=COLORS["navy"])
    ax.set_title(f"Mouse {mouse} - Valid bout count + rewarded licks", color=COLORS["navy"])
    ax.set_xticks(df_session["SessionIndex"])
    style_axes(ax)

    ax.legend(
        handles=[
            mpatches.Patch(color=PROTOCOL_HEX[1], label="Training 1"),
            mpatches.Patch(color=PROTOCOL_HEX[2], label="Training 2"),
            mpatches.Patch(color=PROTOCOL_HEX[3], label="Task"),
            mlines.Line2D([], [], color=COLORS["navy"], marker="o", label="Valid bouts"),
            mlines.Line2D([], [], color=GREEN_MAIN, marker="s", linestyle="--", label="Rewarded licks"),
        ],
        loc="upper left",
    )

    fig.tight_layout()
    return fig


def plot_stacked_lick_counts(df, mouse):
    df_session = df[df["Mouse_ID"] == mouse].copy()
    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    df_session["SessionIndex"] = range(1, len(df_session) + 1)

    reward_total = [count_licks(row, "rewarded") for _, row in df_session.iterrows()]
    nonreward_total = [count_licks(row, "non_rewarded") for _, row in df_session.iterrows()]
    invalid_total = [count_licks(row, "invalid") for _, row in df_session.iterrows()]

    fig, ax = plt.subplots(figsize=(14, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for _, row in df_session.iterrows():
        s = row["SessionIndex"]
        proto = row["Protocol"]
        p = row["Proba_val"]
        base = PROTOCOL_COLORS.get(proto, (0.93, 0.93, 0.93))
        ax.axvspan(s - 0.5, s + 0.5, color=shade_color(base, p), zorder=1)

    blocks = get_protocol_blocks(df_session)
    x = df_session["SessionIndex"].to_numpy()

    ax.bar(
        x,
        reward_total,
        width=0.62,
        color=GREEN_MAIN,
        alpha=0.98,
        edgecolor=GREEN_MAIN,
        linewidth=0.8,
        label="Rewarded",
        zorder=40,
    )
    ax.bar(
        x,
        nonreward_total,
        bottom=reward_total,
        width=0.62,
        color=ORANGE_MAIN,
        alpha=0.98,
        edgecolor=ORANGE_MAIN,
        linewidth=0.8,
        label="Non-rewarded",
        zorder=40,
    )
    ax.bar(
        x,
        invalid_total,
        bottom=np.array(reward_total) + np.array(nonreward_total),
        width=0.62,
        color=GRAY_MAIN,
        alpha=0.98,
        edgecolor=GRAY_MAIN,
        linewidth=0.8,
        label="Invalid",
        zorder=40,
    )

    stacked_totals = np.array(reward_total) + np.array(nonreward_total) + np.array(invalid_total)
    ymax_data = float(np.nanmax(stacked_totals)) if len(stacked_totals) else 1.0
    yrange = max(1.0, ymax_data)
    bottom_margin = 0.11 * yrange
    y_text = -0.045 * yrange

    ax.set_ylim(-bottom_margin, ymax_data * 1.04)
    add_proba_labels(ax, blocks, y_text=y_text, color=COLORS["gray"], fontsize=8.3)

    ax.set_title(f"Mouse {mouse} - Lick counts per session", color=COLORS["navy"])
    ax.set_xlabel("Training sessions", color=COLORS["navy"])
    ax.set_ylabel("Total licks", color=COLORS["navy"])
    ax.set_xticks(x)
    style_axes(ax)

    lick_legend = [
        mpatches.Patch(color=GREEN_MAIN, label="Rewarded"),
        mpatches.Patch(color=ORANGE_MAIN, label="Non-rewarded"),
        mpatches.Patch(color=GRAY_MAIN, label="Invalid"),
    ]
    protocol_legend = [
        mpatches.Patch(color=PROTOCOL_HEX[1], label="Training 1"),
        mpatches.Patch(color=PROTOCOL_HEX[2], label="Training 2"),
        mpatches.Patch(color=PROTOCOL_HEX[3], label="Task"),
    ]
    ax.legend(handles=lick_legend + protocol_legend, loc="upper left")

    fig.tight_layout()
    return fig


def plot_histogram_kde_failures(df, mouse):
    df_session = df[df["Mouse_ID"] == mouse].copy()
    if df_session.empty:
        return None

    valid_failures = []

    for _, row in df_session.iterrows():
        failures = np.asarray(compute_failures(row), dtype=float)
        valid_mask = valid_bout_mask_from_row(row, target_len=len(failures))
        failures = failures[valid_mask]
        failures = failures[np.isfinite(failures)]
        failures = failures[failures > 0]
        if len(failures) > 0:
            valid_failures.append(failures)

    if len(valid_failures) == 0:
        return None

    all_failures = np.concatenate(valid_failures)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    max_x = min(50, int(np.nanmax(all_failures))) if len(all_failures) else 50
    max_x = max(max_x, 5)

    failures_int = np.rint(all_failures).astype(int)
    failures_int = failures_int[(failures_int >= 1) & (failures_int <= max_x)]

    all_x = np.arange(1, max_x + 1)
    counts_full = np.zeros_like(all_x, dtype=float)

    if len(failures_int) > 0:
        vals, cnts = np.unique(failures_int, return_counts=True)
        for v, c in zip(vals, cnts):
            if v in all_x:
                counts_full[all_x == v] = c

    p_emp = counts_full / counts_full.sum() if counts_full.sum() > 0 else counts_full

    ax.bar(
        all_x,
        p_emp,
        width=0.8,
        color=BLUE_FILL,
        edgecolor=BLUE_FILL,
        alpha=0.30,
        linewidth=1.0,
        zorder=10,
    )

    x_smooth, y_smooth = smooth_discrete_curve_fixed_range(
        all_x,
        p_emp,
        x_min=1,
        x_max=max_x,
        sigma=1.0,
        points=900,
    )

    ax.plot(x_smooth, y_smooth, color=BLUE_MAIN, linewidth=2.6, zorder=20)
    ax.fill_between(x_smooth, y_smooth, color=BLUE_MAIN, alpha=0.14, zorder=15)

    ax.set_title(f"Mouse {mouse} - Distribution of consecutive failures", color=COLORS["navy"])
    ax.set_xlabel("Consecutive failures", color=COLORS["navy"])
    ax.set_ylabel("Probability", color=COLORS["navy"])
    ax.set_xlim(1, max_x)
    style_axes(ax)

    fig.tight_layout()
    return fig


def plot_kde_failures_by_session(df, mouse, session_cmap, bandwidth_factor=0.8):
    df_session = df[
        (df["Mouse_ID"] == mouse) &
        (df["Protocol"] == 3) &
        (df["Proba_val"] == 0.30)
    ].copy()

    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    valid_sessions = []
    for _, row in df_session.iterrows():
        failures = np.array(compute_failures(row), dtype=float)
        valid_mask = valid_bout_mask_from_row(row, target_len=len(failures))
        failures = failures[valid_mask]
        failures = failures[np.isfinite(failures)]
        failures = failures[failures > 0]

        if len(failures) < 100:
            continue

        valid_sessions.append((row["Date"], failures))

    if len(valid_sessions) == 0:
        plt.close(fig)
        return None

    xs = np.linspace(0, 30, 400)
    n_sessions = len(valid_sessions)

    for idx, (date_val, failures) in enumerate(valid_sessions):
        kde = gaussian_kde(failures)
        kde.set_bandwidth(kde.factor * bandwidth_factor)
        ys = kde(xs)
        color = session_cmap(idx / max(1, n_sessions - 1))
        ax.plot(
            xs,
            ys,
            linewidth=2.5,
            color=color,
            alpha=0.98,
            label=f"{date_val.strftime('%Y-%m-%d')} (n={len(failures)})",
        )

    ax.set_title(f"Mouse {mouse} - KDE of consecutive failures by task session", color=COLORS["navy"])
    ax.set_xlabel("Consecutive failures", color=COLORS["navy"])
    ax.set_ylabel("Density", color=COLORS["navy"])
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    style_axes(ax)
    ax.legend(title="Session", fontsize=8)

    fig.tight_layout()
    return fig


def plot_regression_rewards_failures_and_slope(
    df,
    mouse,
    session_cmap,
    max_reward=7,
    max_failure=30,
    min_valid_bouts=100
):
    df_session = df[
        (df["Mouse_ID"] == mouse) &
        (df["Protocol"] == 3) &
        (df["Proba_val"] == 0.30)
    ].copy()

    if df_session.empty:
        return None

    df_session = df_session.sort_values("Date").reset_index(drop=True)
    valid_sessions = []

    for _, row in df_session.iterrows():
        failures = np.asarray(compute_failures(row), dtype=float)
        rewards = np.asarray(count_reward_per_bout(row), dtype=float)

        n0 = min(len(failures), len(rewards))
        failures = failures[:n0]
        rewards = rewards[:n0]

        valid_mask = valid_bout_mask_from_row(row, target_len=n0)

        failures = failures[valid_mask]
        rewards = rewards[valid_mask]

        mask = np.isfinite(failures) & np.isfinite(rewards)
        failures = failures[mask]
        rewards = rewards[mask]

        mask = failures > 0
        failures = failures[mask]
        rewards = rewards[mask]

        n_valid = len(failures)
        if n_valid < min_valid_bouts:
            continue

        mask = rewards <= max_reward
        failures_cut = failures[mask]
        rewards_cut = rewards[mask]

        mask = failures_cut <= max_failure
        failures_cut = failures_cut[mask]
        rewards_cut = rewards_cut[mask]

        if len(failures_cut) < 2 or len(np.unique(rewards_cut)) < 2:
            continue

        model = LinearRegression()
        model.fit(rewards_cut.reshape(-1, 1), failures_cut)

        valid_sessions.append({
            "date": pd.to_datetime(row["Date"]),
            "n_valid": n_valid,
            "slope": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "mean_failures": float(np.mean(failures_cut)),
        })

    if len(valid_sessions) == 0:
        return None

    fig, axs = plt.subplots(1, 3, figsize=(20, 5.4), gridspec_kw={"wspace": 0.30})
    fig.patch.set_facecolor("white")
    ax_left, ax_mid, ax_right = axs

    n_sessions = len(valid_sessions)
    x_line = np.linspace(1, max_reward, 200)

    for idx, sess in enumerate(valid_sessions):
        color = session_cmap(idx / max(1, n_sessions - 1))
        y_line = sess["slope"] * x_line + sess["intercept"]
        ax_left.plot(
            x_line,
            y_line,
            color=color,
            linewidth=2.25,
            alpha=0.98,
            label=f"{sess['date'].strftime('%Y-%m-%d')} (n={sess['n_valid']})",
        )

    ax_left.set_title(f"Mouse {mouse} - Reward vs failures", color=COLORS["navy"])
    ax_left.set_xlabel("Reward count", color=COLORS["navy"])
    ax_left.set_ylabel("Consecutive failures", color=COLORS["navy"])
    ax_left.set_xlim(1, max_reward)
    ax_left.set_ylim(1, max_failure)
    ax_left.set_xticks(np.arange(1, max_reward + 1))
    style_axes(ax_left)
    ax_left.legend(title="Session", fontsize=8)

    dates = [sess["date"] for sess in valid_sessions]
    slopes = np.array([sess["slope"] for sess in valid_sessions], dtype=float)
    x_idx = np.arange(len(valid_sessions))

    ax_mid.scatter(x_idx, slopes, color="black", s=34, zorder=20)

    exp_fit_slopes = fit_exponential_trend_with_band(slopes)
    if exp_fit_slopes is not None:
        ax_mid.fill_between(
            exp_fit_slopes["x_smooth"],
            exp_fit_slopes["y_lower"],
            exp_fit_slopes["y_upper"],
            color=PURPLE_MAIN,
            alpha=0.16,
            zorder=10,
        )
        ax_mid.plot(
            exp_fit_slopes["x_smooth"],
            exp_fit_slopes["y_smooth"],
            color=PURPLE_MAIN,
            linewidth=2.5,
            zorder=15,
        )

    ax_mid.set_title("Slope over time", color=COLORS["navy"])
    ax_mid.set_xlabel("Session date", color=COLORS["navy"])
    ax_mid.set_ylabel("Linear slope", color=COLORS["navy"])
    ax_mid.set_xticks(x_idx)
    ax_mid.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    style_axes(ax_mid)

    mean_failures = np.array([sess["mean_failures"] for sess in valid_sessions], dtype=float)
    ax_right.scatter(x_idx, mean_failures, color="black", s=34, zorder=20)

    exp_fit_mean = fit_exponential_trend_with_band(mean_failures)
    if exp_fit_mean is not None:
        ax_right.fill_between(
            exp_fit_mean["x_smooth"],
            exp_fit_mean["y_lower"],
            exp_fit_mean["y_upper"],
            color=PURPLE_MAIN,
            alpha=0.16,
            zorder=10,
        )
        ax_right.plot(
            exp_fit_mean["x_smooth"],
            exp_fit_mean["y_smooth"],
            color=PURPLE_MAIN,
            linewidth=2.5,
            zorder=15,
        )

    ax_right.set_title("Mean failures over time", color=COLORS["navy"])
    ax_right.set_xlabel("Session date", color=COLORS["navy"])
    ax_right.set_ylabel("Mean consecutive failures", color=COLORS["navy"])
    ax_right.set_xticks(x_idx)
    ax_right.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=45, ha="right")
    style_axes(ax_right)

    fig.tight_layout()
    return fig


# =============================================================================
# SESSION FOCUS PLOTS
# =============================================================================
def prepare_session_arrays(session, reward_cut=7):
    rewards = np.asarray(ensure_list(session["Rewards"]), dtype=float)
    failures = np.asarray(ensure_list(session["Licks After"]), dtype=float)
    valid_bouts = np.asarray(ensure_list(session["Correct Bouts"]), dtype=bool)

    n = min(len(rewards), len(failures), len(valid_bouts))
    rewards = rewards[:n]
    failures = failures[:n]
    valid_bouts = valid_bouts[:n]

    rewards = rewards[valid_bouts]
    failures = failures[valid_bouts]

    mask = np.isfinite(rewards) & np.isfinite(failures)
    rewards = rewards[mask]
    failures = failures[mask]

    mask = failures > 0
    rewards = rewards[mask]
    failures = failures[mask]

    mask = rewards <= reward_cut
    rewards = rewards[mask]
    failures = failures[mask]

    return rewards, failures, int(np.sum(valid_bouts))


def build_session_plot_rewards_vs_failures(session, mouse_id, date_str, reward_cut=7):
    rewards, failures, n_valid = prepare_session_arrays(session, reward_cut=reward_cut)

    title_suffix = f" - {session['Probas']}" if "Probas" in session.index else ""
    title = f"{mouse_id} - {date_str}{title_suffix}"

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if len(rewards) > 0:
        reward_bins = np.sort(np.unique(rewards.astype(int)))
        means = np.array([failures[rewards == r].mean() for r in reward_bins])
        stds = np.array([failures[rewards == r].std() for r in reward_bins])
        counts = np.array([np.sum(rewards == r) for r in reward_bins])

        ax.fill_between(
            reward_bins,
            means - stds,
            means + stds,
            color=BLUE_FILL,
            alpha=0.22,
        )
        ax.plot(
            reward_bins,
            means,
            color=BLUE_MAIN,
            linewidth=2.45,
            marker="o",
            markersize=4.8,
        )

        for x, y, c in zip(reward_bins, means, counts):
            ax.text(
                x,
                y + 0.25,
                str(int(c)),
                ha="center",
                va="bottom",
                fontsize=8,
                color=COLORS["gray"],
            )

    ax.set_title(title, color=COLORS["navy"])
    ax.set_xlabel("Consecutive rewards", color=COLORS["navy"])
    ax.set_ylabel("Consecutive failures", color=COLORS["navy"])
    ax.set_xlim(0.8, reward_cut + 0.2)
    ax.set_ylim(bottom=1)
    ax.set_xticks(np.arange(1, reward_cut + 1))
    style_axes(ax)

    ax.text(
        0.99,
        0.98,
        f"n valid bouts = {n_valid}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLORS["gray"],
    )

    fig.tight_layout()
    return fig


def build_session_plot_failure_distribution(session, failure_xlim=(0, 25), reward_cut=7):
    rewards, failures, n_valid = prepare_session_arrays(session, reward_cut=reward_cut)
    _ = rewards

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    failures_dist = failures.astype(int) if len(failures) > 0 else np.array([], dtype=int)
    failures_dist = failures_dist[(failures_dist >= 1) & (failures_dist <= failure_xlim[1])]

    all_x = np.arange(1, failure_xlim[1] + 1)
    counts_full = np.zeros_like(all_x, dtype=float)

    if len(failures_dist) > 0:
        vals, cnts = np.unique(failures_dist, return_counts=True)
        for v, c in zip(vals, cnts):
            if v in all_x:
                counts_full[all_x == v] = c

    p_emp = counts_full / counts_full.sum() if counts_full.sum() > 0 else counts_full

    ax.bar(
        all_x,
        p_emp,
        width=0.8,
        color=BLUE_FILL,
        edgecolor=BLUE_FILL,
        alpha=0.28,
        linewidth=1.0,
        zorder=10,
    )

    x_smooth, y_smooth = smooth_discrete_curve_fixed_range(
        all_x,
        p_emp,
        x_min=1,
        x_max=failure_xlim[1],
        sigma=1.0,
        points=900,
    )

    ax.plot(x_smooth, y_smooth, color=BLUE_MAIN, linewidth=2.5, zorder=20)
    ax.fill_between(x_smooth, y_smooth, color=BLUE_MAIN, alpha=0.14, zorder=15)

    ax.set_xlabel("Consecutive failures", color=COLORS["navy"])
    ax.set_ylabel("Probability", color=COLORS["navy"])
    ax.set_xlim(1, failure_xlim[1])
    ax.set_xticks(np.arange(1, failure_xlim[1] + 1, 5))
    style_axes(ax)

    ax.text(
        0.99,
        0.98,
        f"n valid bouts = {n_valid}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=COLORS["gray"],
    )

    fig.tight_layout()
    return fig


# =============================================================================
# CALENDAR HELPERS
# =============================================================================
def get_month_options(valid_dates):
    return sorted({(d.year, d.month) for d in valid_dates})


def month_to_label(year, month):
    return f"{year}-{month:02d}"


def build_month_day_list(year, month):
    return [
        d
        for d in calendar.Calendar(firstweekday=0).itermonthdates(year, month)
        if d.month == month
    ]


def get_session_date_to_protocol(df_sessions):
    out = {}
    for _, row in df_sessions.iterrows():
        d = pd.Timestamp(row["Date_norm"]).date()
        proto = row["Protocol"]
        out[d] = int(proto) if pd.notna(proto) else None
    return out


def save_figure(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SESSION FOCUS - BOUT TIMELINE
# =============================================================================
def extract_bout_timeline_data(session):
    timestamps = ensure_array(session.get("Timestamps")).astype(float)
    bouts = ensure_array(session.get("Bout for Timestamps"))

    rewarded = flatten_nested_times(session.get("Times Rewarded Licks", []))
    non_rewarded = flatten_nested_times(session.get("Times Non Rewarded Licks", []))

    if "Manual Reward Bouts" in session.index:
        manual_bouts = set(ensure_array(session.get("Manual Reward Bouts")).tolist())
    else:
        manual_bouts = set()

    if "Correct Bouts" in session.index:
        correct_bouts = ensure_array(session.get("Correct Bouts")).astype(bool)
    else:
        correct_bouts = np.array([], dtype=bool)

    if len(timestamps) == 0 or len(bouts) == 0:
        return None

    n = min(len(timestamps), len(bouts))
    timestamps = timestamps[:n]
    bouts = bouts[:n]

    unique_bouts = np.unique(bouts)
    unique_bouts_sorted = np.sort(unique_bouts)

    rows = []
    for i, b in enumerate(unique_bouts_sorted):
        mask = bouts == b
        t = timestamps[mask]

        if len(t) == 0:
            continue

        t_start = float(np.min(t))
        t_end = float(np.max(t))
        duration = max(t_end - t_start, 1e-6)

        r = sum(t_start <= x <= t_end for x in rewarded)
        nr = sum(t_start <= x <= t_end for x in non_rewarded)
        m = 1 if b in manual_bouts else 0

        if len(correct_bouts) > i:
            is_valid = bool(correct_bouts[i])
        else:
            is_valid = True

        rows.append((b, t_start, t_end, duration, r, nr, m, is_valid))

    if not rows:
        return None

    arr = np.array(rows, dtype=object)
    return {
        "bout_id": arr[:, 0],
        "t_start": arr[:, 1].astype(float),
        "t_end": arr[:, 2].astype(float),
        "duration": arr[:, 3].astype(float),
        "rewarded": arr[:, 4].astype(float),
        "non_rewarded": arr[:, 5].astype(float),
        "manual": arr[:, 6].astype(float),
        "is_valid": arr[:, 7].astype(bool),
    }


def build_session_plot_bout_timeline(
    session,
    title="Bout events over raw time",
    y_max=30,
):
    data = extract_bout_timeline_data(session)

    if data is None:
        return None

    t_start = data["t_start"]
    t_end = data["t_end"]
    duration = data["duration"]

    rewarded = data["rewarded"]
    non_rewarded = data["non_rewarded"]
    manual = data["manual"]
    valid = data["is_valid"]
    invalid = ~valid
    manual_mask = manual > 0

    n_bouts = len(data["bout_id"])
    fig_width = min(34, max(18, n_bouts * 0.03))
    fig_height = 6.2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    manual_navy = "#0B3D91"

    ax.set_ylim(0, y_max)

    # Invalid bouts
    if np.any(invalid):
        first_invalid = True
        for xs, xe in zip(t_start[invalid], t_end[invalid]):
            ax.axvspan(
                xs,
                xe,
                color=GRAY_MAIN,
                alpha=0.28,
                lw=0,
                zorder=1,
                label="Invalid bout" if first_invalid else None,
            )
            first_invalid = False

    # Manual bouts
    if np.any(manual_mask):
        first_manual = True
        for xs, xe in zip(t_start[manual_mask], t_end[manual_mask]):
            ax.axvspan(
                xs,
                xe,
                color=manual_navy,
                alpha=0.60,
                lw=0,
                zorder=2,
                label="Manual bout" if first_manual else None,
            )
            first_manual = False

    # Valid bout bars
    if np.any(valid):
        ax.bar(
            t_start[valid],
            rewarded[valid],
            width=duration[valid],
            align="edge",
            color=GREEN_MAIN,
            edgecolor="none",
            label="Rewarded",
            zorder=3,
        )

        ax.bar(
            t_start[valid],
            non_rewarded[valid],
            width=duration[valid],
            align="edge",
            bottom=rewarded[valid],
            color=RED_MAIN,
            edgecolor="none",
            label="Non-rewarded",
            zorder=3,
        )

        if np.any(manual[valid] > 0):
            ax.bar(
                t_start[valid],
                manual[valid],
                width=duration[valid],
                align="edge",
                bottom=rewarded[valid] + non_rewarded[valid],
                color=BLUE_MAIN,
                edgecolor="none",
                label="Manual reward",
                zorder=4,
            )

    ax.set_xlim(t_start.min(), t_end.max())

    ax.set_xlabel("Raw time", fontsize=14, color=COLORS["navy"])
    ax.set_ylabel("Count per bout", fontsize=14, color=COLORS["navy"])
    ax.set_title(title, fontsize=16, color=COLORS["navy"], pad=12)

    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["axis"])
    ax.spines["bottom"].set_color(COLORS["axis"])

    ax.tick_params(axis="both", labelsize=12, colors=COLORS["gray"])

    handles, labels = ax.get_legend_handles_labels()
    order = ["Rewarded", "Non-rewarded", "Manual reward", "Manual bout", "Invalid bout"]
    ordered = [(h, l) for name in order for h, l in zip(handles, labels) if l == name]

    if ordered:
        legend = ax.legend(
            [x[0] for x in ordered],
            [x[1] for x in ordered],
            loc="upper right",
            ncol=3,
            fontsize=14,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            borderpad=0.8,
            labelspacing=0.8,
            handlelength=1.8,
            handletextpad=0.6,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor(COLORS["axis"])
        legend.get_frame().set_linewidth(1.0)

    fig.tight_layout()
    return fig
