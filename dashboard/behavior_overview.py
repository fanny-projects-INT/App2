"""Lightweight Behavior overview using only local session metadata."""

import html
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dashboard.app_functions import PROTOCOL_LABELS

OVERVIEW_UI_VERSION = 20
PROTOCOL_COLORS = {1:"#D2A72C", 2:"#428CB3", 3:"#CF725E", 4:"#80639A"}
STYLES = """
<style>
.behavior-overview-grid {display:flex;flex-wrap:wrap;align-items:flex-start;gap:10px;margin:6px 0 24px;}
.behavior-overview-card {flex:0 0 auto;max-width:100%;padding:12px 16px;border:1px solid #E3E9EE;border-radius:10px;background:white;box-sizing:border-box;}
.behavior-overview-label {font-size:.63rem;font-weight:600;letter-spacing:.04em;color:#748091;margin-bottom:5px;}
.behavior-overview-value {font-size:1.08rem;font-weight:600;line-height:1.25;color:#223248;overflow-wrap:anywhere;}

.behavior-overview-types {display:flex;flex-wrap:wrap;gap:14px;line-height:1.35;color:#536277;font-size:.76rem;}
.behavior-overview-types span {white-space:nowrap;}
.behavior-overview-types i {display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;background:var(--accent);}
.behavior-overview-types strong {color:#223248;font-size:1.08rem;margin-right:4px;}

.behavior-session-scroll {overflow-x:auto;padding:14px 8px 4px;}
.behavior-session-track {display:grid;grid-template-columns:repeat(30,minmax(24px,1fr));list-style:none;margin:0;padding:0;min-width:720px;position:relative;border-radius:8px;overflow:hidden;}
.behavior-session-track::before {content:'';position:absolute;left:0;right:0;top:43px;height:2px;background:#CDD7E2;z-index:1;pointer-events:none;}
.behavior-session-day {position:relative;min-width:0;text-align:center;padding:8px 0;background:var(--week-background);}
.behavior-session-weekday {height:26px;line-height:18px;color:#8492A4;font-size:.65rem;}
.behavior-session-day.week-start .behavior-session-weekday {color:#536277;font-weight:600;}
.behavior-session-markers {position:relative;z-index:2;height:20px;display:flex;justify-content:center;align-items:center;gap:3px;}
.behavior-session-day.week-start::before {content:'';position:absolute;left:0;top:4px;bottom:4px;width:1px;background:#BECAD7;}
.behavior-session-number {font-size:.72rem;line-height:22px;color:#536277;margin-top:5px;}
.behavior-session-day.week-start .behavior-session-number {font-weight:700;}
.behavior-session-day.weekend .behavior-session-number {color:#A4AEBA;}
.behavior-session-dot {display:block;flex:0 1 10px;width:10px;min-width:4px;height:10px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 2px white;}
.behavior-session-dot:hover,.behavior-session-dot:focus {transform:scale(1.3);outline:2px solid var(--accent);outline-offset:2px;z-index:1;}
.behavior-session-range {display:flex;justify-content:space-between;min-width:720px;color:#8A97A5;font-size:.7rem;margin-top:9px;}
@media(max-width:700px) {.behavior-overview-card {padding:10px 12px;}}
</style>
"""


def recent_sessions(frame, limit=24):
    """Keep distinct dated DB rows, most recent first, with numeric version ordering."""
    if frame.empty:
        return []
    local = frame.copy()
    local['_date'] = pd.to_datetime(local['Date'], errors='coerce', utc=True)
    local['_version'] = pd.to_numeric(local['Version'], errors='coerce')
    local = local.dropna(subset=['_date']).sort_values(
        ['_date','_version'], ascending=[False,False], kind='stable', na_position='last')
    sessions = []
    for row in (local if limit is None else local.head(limit)).to_dict('records'):
        protocol = pd.to_numeric(row.get('Protocol'), errors='coerce')
        protocol = int(protocol) if pd.notna(protocol) else None
        version = row.get('Version')
        if pd.isna(version):
            version = ''
        elif pd.notna(row['_version']) and float(row['_version']).is_integer():
            version = str(int(row['_version']))
        else:
            version = str(version)
        sessions.append(dict(date=row['_date'], version=version, protocol=protocol,
                             label=PROTOCOL_LABELS.get(protocol, 'Other')))
    return sessions


def info_cards_markup(values):
    """Shared compact cards for mouse and selected-session information."""
    return ''.join(
        '<div class="behavior-overview-card"><div class="behavior-overview-label">'+html.escape(str(label))+
        '</div><div class="behavior-overview-value">'+html.escape(str(value))+'</div></div>'
        for label, value in values)


def render_session_info(values):
    st.html(STYLES)
    st.html('<div class="behavior-overview-grid">'+info_cards_markup(values)+'</div>')


def summary_markup(mouse_id, frame):
    latest = recent_sessions(frame, 1)
    date = latest[0]['date'].strftime('%d %b %Y') if latest else '—'
    phase = latest[0]['label'] if latest else '—'
    values = [('MOUSE',mouse_id),('SESSIONS',len(frame)),('LATEST SESSION',date),('CURRENT PHASE',phase)]
    cards = info_cards_markup(values)
    protocols = pd.to_numeric(frame.get('Protocol', pd.Series(index=frame.index, dtype=float)), errors='coerce')
    counts = [(protocol, label, int(protocols.eq(protocol).sum()))
              for protocol, label in PROTOCOL_LABELS.items()]
    other = int((~protocols.isin(PROTOCOL_LABELS)).sum())
    if other:
        counts.append((None, 'Other', other))
    breakdown = ''.join(
        '<span style="--accent:'+PROTOCOL_COLORS.get(protocol, '#8A97A5')+'"><i></i><strong>'+
        str(count)+'</strong>'+html.escape(label)+'</span>' for protocol,label,count in counts)
    return ('<div class="behavior-overview-grid">'+cards+
            '<div class="behavior-overview-card"><div class="behavior-overview-label">SESSION TYPES · ALL HISTORY</div>'
            '<div class="behavior-overview-types">'+breakdown+'</div></div></div>')


def rolling_window(end_date=None):
    """Thirty calendar days, including today, rather than a calendar month."""
    end = pd.Timestamp.now().normalize() if end_date is None else pd.Timestamp(end_date).normalize()
    end = end.tz_localize('UTC') if end.tzinfo is None else end.tz_convert('UTC').normalize()
    return end - pd.Timedelta(days=29), end


def timeline_markup(sessions, end_date=None):
    start, end = rolling_window(end_date)
    by_day = {}
    for session in reversed(sessions):
        day = session['date'].normalize()
        if start <= day <= end:
            by_day.setdefault(day, []).append(session)
    nodes = []
    for date in pd.date_range(start, end):
        markers = []
        for session in by_day.get(date, []):
            color = PROTOCOL_COLORS.get(session['protocol'], '#8A97A5')
            version = 'v'+session['version'] if session['version'] else ''
            tooltip = session['date'].strftime('%d %b %Y')+' · '+session['label']+(' · '+version if version else '')
            markers.append('<span class="behavior-session-dot" style="--accent:'+color+
                           '" title="'+html.escape(tooltip, quote=True)+'" aria-label="'+
                           html.escape(tooltip, quote=True)+'" tabindex="0"></span>')
        week_start = date.dayofweek == 0
        classes = (' week-start' if week_start else '') + (' weekend' if date.dayofweek >= 5 else '')
        week_offset = ((date - start).days + start.dayofweek) // 7
        week_background = '#EDF2F7' if week_offset % 2 == 0 else '#FFFFFF'
        weekday = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')[date.dayofweek]
        nodes.append('<li class="behavior-session-day'+classes+'" data-date="'+date.strftime('%Y-%m-%d')+'" style="--week-background:'+week_background+'">'
                     '<div class="behavior-session-weekday" title="'+weekday+'">'+weekday[0]+'</div>'
                     '<div class="behavior-session-markers">'+''.join(markers)+'</div>'
                     '<div class="behavior-session-number">'+str(date.day)+'</div></li>')
    return ('<div class="behavior-session-scroll"><ol class="behavior-session-track" '
            'aria-label="Behavior sessions over the last 30 days, oldest to newest">'+''.join(nodes)+
            '</ol><div class="behavior-session-range"><span>'+start.strftime('%d %b')+
            '</span><span>'+end.strftime('%d %b')+'</span></div></div>')


def render_behavior_overview(mouse_id, mouse_frame):
    st.html(STYLES)
    st.html(summary_markup(mouse_id, mouse_frame))
    st.html(timeline_markup(recent_sessions(mouse_frame, limit=None)))


def combined_failure_figure(plots):
    """Overlay the existing global smooth curve on the per-session curves."""
    sessions = plots.get('kde_failures_by_session')
    distribution = plots.get('histogram_kde_failures')
    if sessions is None and distribution is None:
        return None
    figure = go.Figure(sessions) if sessions is not None else go.Figure()
    if sessions is None:
        figure.update_layout(distribution.layout)
    if distribution is not None:
        for trace in distribution.data:
            if trace.type != 'scatter':
                continue
            curve = go.Scatter(trace)
            curve.update(
                name='All sessions', mode='lines', fill=None,
                line={'color': '#111111', 'dash': 'dash', 'width': 2.2},
                hovertemplate='All sessions<br>%{x:.1f} consecutive failures<br>%{y:.3f} density<extra></extra>',
            )
            figure.add_trace(curve)
    figure.update_xaxes(range=[0, 30], autorange=False)
    return figure


def compact_progression_figure(plots):
    """Reuse Progression's data and fits without changing its cached figures."""
    plots = dict(plots, combined_failures=combined_failure_figure(plots))
    panels = [
        ('training_outcomes', 1, 1, 1, 'Valid bouts'),
        ('training_outcomes', 2, 2, 1, 'Rewarded licks / min'),
        ('combined_failures', 1, 1, 2, 'Consecutive failure distribution'),
        ('regression', 1, 2, 2, 'Rewards vs consecutive failures'),
        ('regression', 2, 2, 3, 'Linear slope'),
        ('regression', 3, 1, 3, 'Mean consecutive failures'),
    ]
    def panel_traces(name, axis_index):
        original = plots.get(name)
        suffix = '' if axis_index == 1 else str(axis_index)
        return [] if original is None else [
            trace for trace in original.data
            if (trace.xaxis or 'x') == 'x'+suffix
            and any(values is not None and len(values) for values in (trace.x, trace.y))]

    titles = [title if panel_traces(name, axis_index) else ''
              for name, axis_index, row, col, title in sorted(panels, key=lambda panel: (panel[2], panel[3]))]
    figure = make_subplots(
        rows=2, cols=3, column_widths=[.46, .27, .27],
        subplot_titles=titles,
        horizontal_spacing=.065, vertical_spacing=.22,
    )
    for name, axis_index, row, col, title in panels:
        original = plots.get(name)
        suffix = '' if axis_index == 1 else str(axis_index)
        traces = panel_traces(name, axis_index)
        for trace in traces:
            payload = trace.to_plotly_json()
            payload.pop('xaxis', None)
            payload.pop('yaxis', None)
            payload['showlegend'] = False
            if payload.get('type') == 'scatter':
                marker = payload.get('marker', {})
                if name == 'training_outcomes' and ('markers' in (payload.get('mode') or '') or 'marker' in payload):
                    marker = payload.setdefault('marker', {})
                    marker['size'] = 10
                elif isinstance(marker.get('size'), (int, float)):
                    marker['size'] = min(marker['size'], 5)
                line = payload.get('line', {})
                if line.get('width', 0) > 0:
                    line['width'] = min(line['width'], 2.2 if name == 'combined_failures' and line.get('dash') == 'dash' else 1.6)
            figure.add_trace(payload, row=row, col=col)
        if original is not None:
            for direction in ('x', 'y'):
                axis = original.layout[direction+'axis'+suffix].to_plotly_json()
                for key in ('domain', 'anchor', 'matches', 'overlaying', 'position', 'scaleanchor'):
                    axis.pop(key, None)
                axis.update(title=None, tickfont={'size': 8}, nticks=4, tickangle=0,
                            automargin=True, ticklen=2, zeroline=False)
                if axis.get('tickmode') == 'array' and len(axis.get('tickvals', [])) > 4:
                    indexes = sorted({round(i * (len(axis['tickvals'])-1) / 3) for i in range(4)})
                    axis['tickvals'] = [axis['tickvals'][i] for i in indexes]
                    if 'ticktext' in axis:
                        axis['ticktext'] = [axis['ticktext'][i] for i in indexes]
                if direction == 'x' and name == 'regression' and axis_index in (2, 3):
                    axis.update(showticklabels=False, ticks='')
                update = figure.update_xaxes if direction == 'x' else figure.update_yaxes
                update(axis, row=row, col=col)
        if not traces:
            figure.update_xaxes(visible=False, row=row, col=col)
            figure.update_yaxes(visible=False, row=row, col=col)
    if panel_traces('combined_failures', 1):
        figure.add_annotation(
            xref='x2 domain', yref='y2 domain', x=.98, y=.82,
            xanchor='right', yanchor='top', showarrow=False,
            text=(
                '<span style="color:#246BCE">Earlier</span> → '
                '<span style="color:#D94A55">Latest</span><br>'
                '<span style="color:#111111">┄ All sessions</span>'
            ),
            align='right', font={'size': 8, 'color': '#536277'},
            bgcolor='rgba(0,0,0,0)', borderpad=0,
        )
    figure.update_annotations(font={
        'family': 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
        'size': 13.6, 'weight': 680, 'color': '#4B5563',
    })
    for name, axis_index, row, col, title in panels:
        left_edge = figure.get_subplot(row, col).xaxis.domain[0]
        for annotation in figure.layout.annotations:
            if annotation.text == title:
                annotation.update(x=left_edge, xanchor='left', align='left', xshift=-30, yshift=18)
    figure.update_layout(
        height=500, margin={'l': 30, 'r': 12, 't': 50, 'b': 40},
        showlegend=False, hovermode='closest',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial, sans-serif', 'size': 9, 'color': '#536277'},
    )
    return figure


def render_progression_preview(plots):
    st.plotly_chart(
        compact_progression_figure(plots), width='stretch',
        key='behavior_overview_progression',
        config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False},
    )
