"""Arena session metrics, using the existing visualisation_metriques notebook formulas.

OFT: 16 pixels/cm, center [300,700], likelihood >= .5, speed cleanup <=100 cm/s.
LDT: dark box inferred from visibility of neck/torso/tail_base at likelihood .5.
"""
import json
import os
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ARENA_METRICS_VERSION = 2
ALGORITHM_VERSION = 1


def source_inputs(session):
    folder = Path(session['session_dir'])
    tracking = sorted(folder.glob('remapped_cropped_*_clean.csv'))
    behavior = sorted(folder.glob('*_behavior.csv'))
    videos = sorted(folder.glob('*_clean.mp4'))
    if not tracking or (session['assay'] == 'OFT' and not behavior):
        return None
    return dict(tracking_csv=str(tracking[0]),
                behavior_csv=str(behavior[0]) if behavior else None,
                clean_video=str(videos[0]) if videos else None)


def inputs_signature(inputs):
    return {'algorithm': ALGORITHM_VERSION, 'files': [
        [name, str(Path(path).resolve()), Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        for name, path in inputs.items() if path
    ]}


def recording_fps(video):
    if not video:
        # Same fallback as the reference notebook, recorded in the saved result.
        return 50., 'fallback'
    import imageio_ffmpeg
    import subprocess
    import re
    from fractions import Fraction
    probe = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-i', video,
         '-vf', 'showinfo', '-frames:v', '1', '-an', '-f', 'null', '-'],
        capture_output=True, text=True, timeout=30,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    match = re.search(r'config in time_base: .*?frame_rate: (\d+/\d+)', probe.stderr)
    if probe.returncode != 0 or match is None:
        raise ValueError('Video frame rate could not be read.')
    fps = float(Fraction(match.group(1)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video frame rate is invalid.')
    return fps, 'video'


def save_result(path, payload):
    handle, temporary = tempfile.mkstemp(prefix='.arena_metrics-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, allow_nan=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def session_metrics(session):
    """Return fresh per-session metrics or calculate/save them once from predictions."""
    if session['assay'] not in ('OFT', 'LDT'):
        return None
    inputs = source_inputs(session)
    if inputs is None:
        return None
    signature = inputs_signature(inputs)
    target = Path(session['session_dir']) / 'arena_metrics.json'
    if target.exists():
        try:
            saved = json.loads(target.read_text(encoding='utf-8'))
            if saved['signature'] == signature and saved['assay'] == session['assay']:
                return saved['metrics']
        except (OSError, ValueError, KeyError):
            pass
    fps, fps_source = recording_fps(inputs['clean_video'])
    row = {**inputs, 'mouse': session['mouse'], 'fps': fps,
           'date_folder': session['date'].replace('-', '_'),
           'session_id': f"{session['mouse']}_{session['date'].replace('-', '_')}_{session['assay']}"}
    result = compute_session_metrics(row) if session['assay'] == 'OFT' else compute_ldt_box_metrics(row)
    if result['n_frames'] <= 0:
        raise ValueError('No analyzed frames are available.')
    result['fps_source'] = fps_source
    # Standard JSON null for unavailable values; keep columns numeric when loaded.
    result = {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value)
              else value for key, value in result.items()}
    if inputs_signature(inputs) != signature:
        raise ValueError('Analysis files changed during calculation; retry on the next visit.')
    save_result(target, {'signature': signature, 'assay': session['assay'], 'metrics': result})
    return result


def collect_metrics(inventory, legacy=None):
    """Local results override the legacy export; keep legacy-only sessions available."""
    legacy = legacy or {}
    frames, errors = {}, []
    for assay in ('OFT', 'LDT', 'SIT'):
        frame = legacy.get(assay, pd.DataFrame()).copy()
        date_column = f'{assay.lower()}_date'
        for session in inventory.loc[inventory['assay'].eq(assay)].to_dict('records'):
            try:
                metrics = session_metrics(session)
            except (OSError, ValueError, KeyError, IndexError, RuntimeError, subprocess.SubprocessError) as exc:
                errors.append(f"{session['mouse']} {assay} {session['date']}: {exc}")
                continue
            if metrics is None:
                continue
            if not frame.empty and 'mouse' in frame and date_column in frame:
                dates = pd.to_datetime(frame[date_column].astype(str).str.replace('_', '-', regex=False), format='mixed', errors='coerce')
                frame = frame.loc[~(frame['mouse'].astype(str).eq(str(session['mouse']))
                                     & dates.dt.strftime('%Y-%m-%d').eq(session['date']))]
            local_row = pd.DataFrame([metrics])
            local_row[date_column] = pd.to_datetime(local_row[date_column].str.replace('_', '-', regex=False), errors='raise')
            frame = pd.concat([frame, local_row], ignore_index=True)
        if date_column in frame:
            frame[date_column] = pd.to_datetime(frame[date_column].astype(str).str.replace('_', '-', regex=False), format='mixed', errors='coerce')
        for name in frame.columns:
            if name.startswith(f'{assay}_'):
                frame[name] = pd.to_numeric(frame[name], errors='coerce')
        frames[assay] = frame
    return frames, errors

PX_PER_CM = 16

CENTER_MIN = 300.0

CENTER_MAX = 700.0

BEHAVIOR_LABELS = ('Rearing', 'wall_rearing', 'Grooming')

BACKGROUND_CONTEXT_LABELS = ('background_center_static', 'background_center_walking', 'background_center_running', 'background_periphery_static', 'background_periphery_walking', 'background_periphery_running')

CENTER_BODYPOINTS = ('neck', 'torso', 'tail_base')

def read_tracking_csv(path):
    return pd.read_csv(path, header=[0, 1, 2], index_col=0)

def read_behavior_csv(path):
    df = pd.read_csv(path)
    if 'context_label' not in df.columns:
        df['context_label'] = df['label']
    return df

def get_bp(df, bp):
    x = df.xs((bp, 'x'), level=('bodyparts', 'coords'), axis=1).iloc[:, 0].to_numpy(float)
    y = df.xs((bp, 'y'), level=('bodyparts', 'coords'), axis=1).iloc[:, 0].to_numpy(float)
    try:
        l = df.xs((bp, 'likelihood'), level=('bodyparts', 'coords'), axis=1).iloc[:, 0].to_numpy(float)
    except KeyError:
        l = np.ones(len(df), dtype=float)
    return (x, y, l)

def mean_position_from_bps(df, bodyparts=CENTER_BODYPOINTS, likelihood_threshold=0.5):
    xs, ys = ([], [])
    for bp in bodyparts:
        x, y, l = get_bp(df, bp)
        valid = np.isfinite(x) & np.isfinite(y) & (l >= likelihood_threshold)
        xs.append(np.where(valid, x, np.nan))
        ys.append(np.where(valid, y, np.nan))
    x_stack = np.vstack(xs)
    y_stack = np.vstack(ys)
    x_count = np.sum(np.isfinite(x_stack), axis=0)
    y_count = np.sum(np.isfinite(y_stack), axis=0)
    x_mean = np.divide(np.nansum(x_stack, axis=0), x_count, out=np.full(x_stack.shape[1], np.nan), where=x_count > 0)
    y_mean = np.divide(np.nansum(y_stack, axis=0), y_count, out=np.full(y_stack.shape[1], np.nan), where=y_count > 0)
    return (x_mean, y_mean)

def rolling_median(values, window_frames):
    if window_frames <= 1:
        return np.asarray(values, dtype=float)
    if window_frames % 2 == 0:
        window_frames += 1
    return pd.Series(values, dtype=float).rolling(window=window_frames, center=True, min_periods=1).median().to_numpy()

def clean_position_for_speed(x_px, y_px, fps, max_speed_cm_s=100, interpolate_gap_s=0.5, smooth_s=0.2):
    x_px = np.asarray(x_px, dtype=float).copy()
    y_px = np.asarray(y_px, dtype=float).copy()
    x_cm = (x_px - 500) / PX_PER_CM
    y_cm = (y_px - 500) / PX_PER_CM
    speed = np.sqrt(np.diff(x_cm) ** 2 + np.diff(y_cm) ** 2) * fps
    bad_target = np.concatenate(([False], speed > max_speed_cm_s))
    x_px[bad_target] = np.nan
    y_px[bad_target] = np.nan
    gap_frames = max(1, int(round(interpolate_gap_s * fps))) if fps and np.isfinite(fps) else 1
    x_px = pd.Series(x_px).interpolate(limit=gap_frames, limit_area='inside').to_numpy()
    y_px = pd.Series(y_px).interpolate(limit=gap_frames, limit_area='inside').to_numpy()
    smooth_frames = max(1, int(round(smooth_s * fps))) if fps and np.isfinite(fps) else 1
    return (rolling_median(x_px, smooth_frames), rolling_median(y_px, smooth_frames))

def count_bouts(mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return 0
    previous = np.concatenate(([False], mask[:-1]))
    return int(np.sum(mask & ~previous))

def circular_rolling_median_angle(angle, window_frames):
    if window_frames <= 1:
        return angle
    if window_frames % 2 == 0:
        window_frames += 1
    sin_s = pd.Series(np.sin(angle)).rolling(window=window_frames, center=True, min_periods=1).median()
    cos_s = pd.Series(np.cos(angle)).rolling(window=window_frames, center=True, min_periods=1).median()
    return np.arctan2(sin_s.to_numpy(), cos_s.to_numpy())

def angular_speed_from_tracking(df, fps, head_bp='nose', tail_bp='tail_base', likelihood_threshold=0.5, max_angular_speed_deg_s=720):
    try:
        head_x, head_y, head_l = get_bp(df, head_bp)
    except KeyError:
        head_x, head_y, head_l = get_bp(df, 'neck')
    tail_x, tail_y, tail_l = get_bp(df, tail_bp)
    dx = head_x - tail_x
    dy = head_y - tail_y
    length_cm = np.sqrt(dx ** 2 + dy ** 2) / PX_PER_CM
    valid = np.isfinite(dx) & np.isfinite(dy) & (head_l >= likelihood_threshold) & (tail_l >= likelihood_threshold) & (length_cm >= 2.0)
    angle = np.full(len(dx), np.nan)
    angle[valid] = np.arctan2(dy[valid], dx[valid])
    angle = pd.Series(angle).interpolate(limit=max(1, int(round(0.2 * fps))), limit_area='inside').to_numpy()
    angle = circular_rolling_median_angle(angle, max(1, int(round(0.2 * fps))))
    valid_angle = np.isfinite(angle)
    unwrapped = np.full_like(angle, np.nan, dtype=float)
    unwrapped[valid_angle] = np.unwrap(angle[valid_angle])
    angular_speed = np.abs(np.diff(unwrapped)) * fps * 180 / np.pi
    angular_speed[angular_speed > max_angular_speed_deg_s] = np.nan
    return angular_speed

def compute_session_metrics(row):
    behavior = read_behavior_csv(row['behavior_csv'])
    tracking = read_tracking_csv(row['tracking_csv'])
    fps = float(row['fps'])
    n_frames = min(len(behavior), len(tracking))
    behavior = behavior.iloc[:n_frames].copy()
    tracking = tracking.iloc[:n_frames]
    if n_frames <= 0:
        raise ValueError("No analyzed frames are available.")
    duration_s = n_frames / fps
    duration_min = duration_s / 60
    x_px, y_px = mean_position_from_bps(tracking, likelihood_threshold=0.5)
    x_px, y_px = clean_position_for_speed(x_px, y_px, fps)
    x_cm = (x_px - 500) / PX_PER_CM
    y_cm = (y_px - 500) / PX_PER_CM
    valid = np.isfinite(x_px) & np.isfinite(y_px)
    in_center = valid & (CENTER_MIN <= x_px) & (x_px <= CENTER_MAX) & (CENTER_MIN <= y_px) & (y_px <= CENTER_MAX)
    in_periphery = valid & ~in_center
    dist_cm = np.sqrt(np.diff(x_cm) ** 2 + np.diff(y_cm) ** 2)
    dist_cm = dist_cm[np.isfinite(dist_cm)]
    speed_cm_s = dist_cm * fps
    speed_cm_s = speed_cm_s[np.isfinite(speed_cm_s) & (speed_cm_s <= 100)]
    angular_speed = angular_speed_from_tracking(tracking, fps)
    labels = behavior['label'].fillna('background').astype(str).to_numpy()
    context_labels = behavior['context_label'].fillna(behavior['label']).astype(str).to_numpy()
    out = {'mouse': row['mouse'], 'oft_date': row['date_folder'], 'date_folder': row['date_folder'], 'session_id': row['session_id'], 'duration_min': duration_min, 'n_frames': n_frames, 'fps': fps, 'OFT_ratio_time_center': float(np.nanmean(in_center)), 'OFT_distance_cm_per_min': float(np.nansum(dist_cm) / duration_min), 'OFT_entries_center_per_min': float(count_bouts(in_center) / duration_min), 'OFT_median_speed_cm_s': float(np.nanmedian(speed_cm_s)) if len(speed_cm_s) else np.nan, 'OFT_speed_cv': float(np.nanstd(speed_cm_s) / np.nanmean(speed_cm_s)) if len(speed_cm_s) and np.nanmean(speed_cm_s) > 0 else np.nan}
    for label in BEHAVIOR_LABELS:
        clean_name = label.lower().replace('wall_rearing', 'wall_rearing')
        mask = labels == label
        out[f'OFT_{clean_name}_per_min'] = float(count_bouts(mask) / duration_min)
        out[f'OFT_{clean_name}_time_ratio'] = float(np.mean(mask))
        out[f'OFT_{clean_name}_mean_score'] = float(behavior.loc[mask, 'score'].mean()) if mask.any() and 'score' in behavior else np.nan
    for label in BACKGROUND_CONTEXT_LABELS:
        mask = context_labels == label
        out[f'OFT_{label}_time_ratio'] = float(np.mean(mask))
        out[f'OFT_{label}_per_min'] = float(count_bouts(mask) / duration_min)
    out['behavior_csv'] = row['behavior_csv']
    out['tracking_csv'] = row['tracking_csv']
    return out

LDT_BOX_BODYPOINTS = ('neck', 'torso', 'tail_base')

LDT_BOX_LIKELIHOOD_THRESHOLD = 0.5

def keypoint_visibility_count(df, bodyparts=LDT_BOX_BODYPOINTS, likelihood_threshold=LDT_BOX_LIKELIHOOD_THRESHOLD):
    present = []
    for bp in bodyparts:
        x, y, likelihood = get_bp(df, bp)
        present.append(np.isfinite(x) & np.isfinite(y) & (likelihood >= likelihood_threshold))
    return np.sum(np.vstack(present), axis=0)

def compute_ldt_box_metrics(row):
    tracking = read_tracking_csv(row['tracking_csv'])
    fps = float(row['fps'])
    n_frames = len(tracking)
    if n_frames <= 0:
        raise ValueError("No analyzed frames are available.")
    duration_s = n_frames / fps
    duration_min = duration_s / 60
    visible_count = keypoint_visibility_count(tracking)
    in_box = visible_count == 0
    out_box = visible_count > 0
    all_core_visible = visible_count == len(LDT_BOX_BODYPOINTS)
    partial_visible = (visible_count > 0) & (visible_count < len(LDT_BOX_BODYPOINTS))
    dark_entries = count_bouts(in_box)
    dark_exits = count_bouts(out_box)
    return {'mouse': row['mouse'], 'ldt_date': row['date_folder'], 'date_folder': row['date_folder'], 'session_id': row['session_id'], 'duration_min': duration_min, 'n_frames': n_frames, 'fps': fps, 'LDT_dark_time_ratio': float(np.mean(in_box)), 'LDT_dark_entries_per_min': float(dark_entries / duration_min), 'LDT_in_box_time_ratio': float(np.mean(in_box)), 'LDT_out_box_time_ratio': float(np.mean(out_box)), 'LDT_all_core_keypoints_visible_time_ratio': float(np.mean(all_core_visible)), 'LDT_partial_core_keypoints_visible_time_ratio': float(np.mean(partial_visible)), 'LDT_box_entries_per_min': float(dark_entries / duration_min), 'LDT_box_exits_per_min': float(dark_exits / duration_min), 'tracking_csv': row['tracking_csv'], 'clean_video': row['clean_video']}
