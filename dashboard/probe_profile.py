"""Region coverage and good-unit metrics for an individual probe."""

PROBE_PROFILE_VERSION = 4

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROFILE_METRICS = {
    'firing_rate': 'Firing rate (Hz)',
    'amplitude_median': 'Amplitude (µV)',
    'snr': 'Signal-to-noise ratio',
    'presence_ratio': 'Presence ratio',
    'num_spikes': 'Spike count',
    'rp_contamination': 'Refractory-period contamination',
    'good_units': 'Good unit count',
}


def prepare_profile(channels, units):
    channels = channels.copy()
    channels['depth_um'] = pd.to_numeric(channels['depth_um'], errors='coerce')
    channels['brain_region'] = channels['brain_region'].fillna('Unknown').astype(str)
    sites = channels.dropna(subset=['depth_um']).sort_values('depth_um').drop_duplicates('depth_um')
    spacing = sites['depth_um'].diff()
    spacing = spacing[spacing > 0].median()
    spacing = float(spacing) if pd.notna(spacing) else 20.
    sites['segment'] = sites['brain_region'].ne(sites['brain_region'].shift()).cumsum()
    segments = sites.groupby(['segment', 'brain_region'], as_index=False).agg(
        start=('depth_um', 'min'), end=('depth_um', 'max'))
    segments['start'] -= spacing / 2
    segments['end'] += spacing / 2
    segments['span_mm'] = (segments['end'] - segments['start']) / 1000
    good = units.loc[units['bombcell_label'].eq('good')].copy()
    good['brain_region'] = good['brain_region'].fillna('Unknown').astype(str)
    for column in ('depth_um', 'firing_rate', 'amplitude_median'):
        good[column] = pd.to_numeric(good[column], errors='coerce')
    good['amplitude_median'] = good['amplitude_median'].abs()
    metrics = good.groupby('brain_region').agg(
        good_units=('unit_id', 'count'), firing_rate=('firing_rate', 'median'),
        amplitude=('amplitude_median', 'median'))
    coverage = channels.groupby('brain_region').size().rename('channels').to_frame()
    coverage['span_mm'] = segments.groupby('brain_region')['span_mm'].sum()
    summary = coverage.join(metrics, how='outer')
    for column in ('channels', 'good_units'):
        summary[column] = summary[column].fillna(0).astype(int)
    summary['density'] = summary['good_units'] / summary['span_mm'].where(summary['span_mm'] > 0)
    summary = summary[summary['good_units'] > 0].reset_index().sort_values(['good_units', 'brain_region'], ascending=[False, True])
    return segments, good, summary


def build_profile_figure(segments, good, region_color, metric='firing_rate', region_summary=None):
    if segments.empty:
        return None
    figure = make_subplots(rows=1, cols=2, shared_yaxes=True,
                           column_widths=[.24, .76], horizontal_spacing=.07,
                           subplot_titles=('Regions along probe', 'Good units by region' if metric == 'good_units' else 'Good units by depth'))
    counts = good['brain_region'].value_counts()
    summaries = {} if region_summary is None else region_summary.set_index('brain_region').to_dict('index')

    def region_hover(region):
        details = summaries.get(region)
        text = f'<b>{region}</b><br>Good units: {int(counts.get(region, 0))}'
        if details is not None:
            for field, label, precision in (
                ('channels', 'Channels', 0), ('span_mm', 'Span (mm)', 2),
                ('density', 'Good units / mm', 1), ('firing_rate', 'Median rate (Hz)', 2),
                ('amplitude', 'Median amplitude (µV)', 1),
            ):
                value = details.get(field, np.nan)
                formatted = f'{value:.{precision}f}' if pd.notna(value) else 'Unavailable'
                text += f'<br>{label}: {formatted}'
        return text

    for segment in segments.itertuples(index=False):
        midpoint = (segment.start + segment.end) / 2
        figure.add_trace(go.Bar(
            x=[1], y=[midpoint], width=[segment.end - segment.start], orientation='h',
            marker_color=region_color(segment.brain_region) if counts.get(segment.brain_region, 0) else '#D5DCE3', showlegend=False,
            customdata=[[segment.brain_region, segment.start, segment.end]],
            hovertemplate=region_hover(segment.brain_region) + '<br>%{customdata[1]:.0f}–%{customdata[2]:.0f} µm<extra></extra>',
        ), row=1, col=1)
        if segment.end - segment.start >= 180 and counts.get(segment.brain_region, 0):
            figure.add_annotation(x=.5, y=midpoint, text=segment.brain_region,
                                  showarrow=False, font=dict(size=10, color='#172033'), row=1, col=1)
    if metric == 'good_units':
        # One total per region, drawn on its largest contiguous depth segment.
        primary = segments.loc[segments.groupby('brain_region')['span_mm'].idxmax()]
        for segment in primary.itertuples(index=False):
            count = int(counts.get(segment.brain_region, 0))
            if count:
                figure.add_trace(go.Bar(
                    x=[count], y=[(segment.start + segment.end) / 2], orientation='h',
                    width=[(segment.end - segment.start) * .8],
                    marker_color=region_color(segment.brain_region), text=[count], textposition='outside',
                    name=segment.brain_region,
                    hovertemplate=region_hover(segment.brain_region) + '<extra></extra>',
                ), row=1, col=2)
    plotted = good.copy()
    plotted[metric] = pd.to_numeric(plotted.get(metric, np.nan), errors='coerce')
    plotted = plotted.replace([np.inf, -np.inf], np.nan).dropna(subset=['depth_um', metric])
    title = PROFILE_METRICS[metric]
    for region, group in plotted.groupby('brain_region'):
        figure.add_trace(go.Scatter(
            x=group[metric], y=group['depth_um'], mode='markers', name=region,
            marker=dict(size=7, color=region_color(region), opacity=.75),
            customdata=group[['unit_id', 'brain_region']],
            hovertemplate='Unit %{customdata[0]} · %{customdata[1]}<br>' + title + ': %{x:.2f}<br>%{y:.0f} µm<extra></extra>',
        ), row=1, col=2)
        median = float(group[metric].median())
        for segment in segments.loc[segments['brain_region'].eq(region)].itertuples(index=False):
            figure.add_shape(type='line', x0=median, x1=median, y0=segment.start, y1=segment.end,
                             line=dict(color=region_color(region), width=3), row=1, col=2)
    if plotted.empty and (metric != 'good_units' or good.empty):
        figure.add_annotation(x=.5, y=.5, xref='x2 domain', yref='y2 domain',
                              text='No good-unit data for this metric', showarrow=False)
    figure.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    figure.update_xaxes(title=title, rangemode='tozero', gridcolor='#E2E8F0', row=1, col=2)
    figure.update_yaxes(title='Distance from probe tip (µm)', row=1, col=1)
    figure.update_yaxes(range=[float(segments['start'].min()), float(segments['end'].max())], gridcolor='#E2E8F0')
    figure.update_layout(height=640, margin=dict(l=65, r=12, t=45, b=55),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          font=dict(color='#263548', size=11), showlegend=False, bargap=0)
    figure.update_annotations(font_size=11)
    return figure
