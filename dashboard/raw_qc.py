"""Bounded, read-only inspection of SpikeGLX recordings (no full decompression)."""
RAW_QC_VERSION = 6

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.signal import welch
import streamlit as st


def find_recordings(session_dir, probe):
    folder = Path(session_dir) / 'rec' / probe
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob('*')
                  if p.suffix.lower() in ('.bin', '.cbin')
                  and '.ap.' in p.name.lower()
                  and p.with_suffix('.meta').exists())


def file_signature(path):
    path = Path(path)
    return tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns)
                 for p in (path, path.with_suffix('.meta'), path.with_suffix('.ch')) if p.exists())


def compute_maps(reader, windows=64, window_s=.25, include_rms=True):
    """Sample evenly spaced windows; DC-removed RMS and mean Welch PSD."""
    fs = float(reader.fs)
    n = min(int(round(window_s * fs)), int(reader.ns))
    if n < 8 or reader.nc - reader.nsync < 1:
        raise ValueError('Recording is too short to inspect.')
    channels = int(reader.nc - reader.nsync)
    starts = np.unique(np.linspace(0, reader.ns - n, windows).astype(int))
    rms, spectrum_sum = [], None
    for start in starts:
        data = np.asarray(reader.read(slice(start, start + n), slice(0, channels), sync=False), dtype=np.float64) * 1e6
        data -= data.mean(axis=0, keepdims=True)
        if include_rms:
            rms.append(np.sqrt(np.mean(data * data, axis=0)))
        frequencies, psd = welch(data, fs=fs, nperseg=min(n, max(8, int(fs / 10))), axis=0)
        if spectrum_sum is None:
            spectrum_sum = np.zeros_like(psd)
        spectrum_sum += psd
    return dict(times=(starts + n / 2) / fs, rms=np.asarray(rms).T if include_rms else None,
                frequencies=frequencies, psd=(spectrum_sum / len(starts)).T,
                duration=reader.ns / fs, window_s=n / fs, channels=channels)


@st.cache_data(show_spinner=False, max_entries=8)
def load_maps(path, signature, windows=64, include_rms=True):
    import spikeglx
    with spikeglx.Reader(path, sort=False) as reader:
        return compute_maps(reader, windows=windows, include_rms=include_rms)


@st.cache_data(show_spinner=False, max_entries=16)
def load_saved_rms(folder, signature):
    folder = Path(folder)
    values = np.load(folder / '_iblqc_ephysTimeRmsAP.rms.npy', allow_pickle=False)
    times = np.load(folder / '_iblqc_ephysTimeRmsAP.timestamps.npy', allow_pickle=False)
    if times.ndim != 1 or values.ndim != 2 or values.shape[0] != len(times):
        raise ValueError('Saved RMS dimensions do not match timestamps.')
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Saved RMS timestamps are invalid.')
    return times, values.T


def prepare_display_rms(times, values, max_bins=600):
    """Pool neighboring saved windows for display; retain every time window."""
    if len(times) <= max_bins:
        return times, np.asarray(values, dtype=np.float32)
    groups = np.array_split(np.arange(len(times)), max_bins)
    display_times = np.array([np.mean(times[group]) for group in groups])
    display_values = np.column_stack([
        np.sqrt(np.nanmean(np.square(values[:, group], dtype=np.float64), axis=1))
        for group in groups
    ]).astype(np.float32)
    return display_times, display_values


def heatmap(x, values, title, x_title, units, symmetric=False):
    limits = np.nanpercentile(np.abs(values) if symmetric else values, [2, 98])
    figure = go.Figure(go.Heatmap(
        x=x, y=np.arange(values.shape[0]), z=values,
        colorscale='RdBu_r' if symmetric else 'Viridis',
        zmin=-limits[1] if symmetric else limits[0], zmax=limits[1],
        colorbar=dict(title=dict(text=units, side="right")),
        hovertemplate='Channel %{y}<br>' + x_title + ': %{x:.3f}<br>%{z:.2f} ' + units + '<extra></extra>',
    ))
    figure.update_layout(title=dict(text=title or ""), height=360, margin=dict(l=55, r=25, t=45, b=45),
                         xaxis_title=x_title, yaxis_title='Acquisition channel',
                         paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#263548'))
    return figure


def render_raw_qc(session_dir, probe):
    st.markdown('<div class="subplot-title">RMS over time</div>', unsafe_allow_html=True)
    folder = Path(session_dir) / 'alf' / probe
    files = [folder / f'_iblqc_ephysTimeRmsAP.{suffix}.npy' for suffix in ('rms', 'timestamps')]
    if not all(p.exists() for p in files):
        st.info('Saved RMS is not available for this probe.')
        return
    try:
        signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files)
        times, values = load_saved_rms(str(folder), signature)
        times, values = prepare_display_rms(times, values)
        figure = heatmap(times, values, '', 'Time (s)', 'uV RMS')
        figure.update_layout(title=dict(text=""), margin=dict(l=65, r=60, t=20, b=45))
        st.plotly_chart(figure, width='stretch', config={'displayModeBar': False})
    except (OSError, ValueError) as exc:
        st.warning(f'Saved RMS could not be read: {exc}')
