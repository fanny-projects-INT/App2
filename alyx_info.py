"""Mouse information from Alyx, isolated to the current Streamlit session."""

import html
import re  # Kept for compatibility with Streamlit sessions running the previous view.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alyx_dashboard_readonly.alyx_rest import DEFAULT_LAB, DEFAULT_URL, connect, load_raw_sessions
from alyx_dashboard_readonly.normalize import COLUMNS, normalize_sessions
from app_functions import count_licks
from ui_states import render_no_data


INFO_CACHE_VERSION = 5


ICONS = {
    "calendar": '<svg viewBox="0 0 24 24"><path d="M6 3v3m12-3v3M4 9h16M5 5h14a1 1 0 0 1 1 1v14H4V6a1 1 0 0 1 1-1Z"/></svg>',
    "task": '<svg viewBox="0 0 24 24"><path d="m12 3 8 4-8 4-8-4 8-4Zm-8 9 8 4 8-4M4 17l8 4 8-4"/></svg>',
    "user": '<svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 21c.8-4.5 3.5-7 8-7s7.2 2.5 8 7"/></svg>',
    "bouts": '<svg viewBox="0 0 24 24"><path d="M4 19V9m6 10V5m6 14v-7m4 7V8"/></svg>',
    "result": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    "performance": '<svg viewBox="0 0 24 24"><path d="M4 19V9m6 10V5m6 14v-7m4 7V8"/></svg>',
    "setup": '<svg viewBox="0 0 24 24"><path d="M4 6h16M7 12h10M9 18h6"/><circle cx="8" cy="6" r="2"/><circle cx="14" cy="12" r="2"/><circle cx="12" cy="18" r="2"/></svg>',
    "welfare": '<svg viewBox="0 0 24 24"><path d="M12 21s-8-4.8-8-11a4.5 4.5 0 0 1 8-2.8A4.5 4.5 0 0 1 20 10c0 6.2-8 11-8 11Z"/></svg>',
    "water": '<svg viewBox="0 0 24 24"><path d="M12 3S6 10 6 15a6 6 0 0 0 12 0c0-5-6-12-6-12Z"/></svg>',
    "surgery": '<svg viewBox="0 0 24 24"><path d="m5 19 4-4m6-6 4-4M8 4l12 12-4 4L4 8l4-4Z"/></svg>',
    "probe": '<svg viewBox="0 0 24 24"><path d="M7 3h10v4H7zM10 7h4v10l-2 4-2-4V7Z"/></svg>',
    "lick": '<svg viewBox="0 0 24 24"><path d="M5 13h3l2-5 3 10 2-5h4"/></svg>',
    "rig": '<svg viewBox="0 0 24 24"><path d="M5 5h14v11H5zM9 20h6M12 16v4"/></svg>',
}


INFO_STYLES = """
<style>
.alyx-profile-band {
    display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); align-items:stretch;
    min-height:126px; margin:0 0 28px; border-radius:16px;
    background:rgba(255,255,255,.62); overflow:hidden;
}
.alyx-profile-item {
    min-width:0; min-height:92px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; padding:16px 18px;
    border-left:1px solid rgba(115,137,157,.18); text-align:center;
}
.alyx-profile-item:first-child { border-left:0; }
.alyx-profile-value {
    max-width:100%; color:#223248; font-size:1.42rem; font-weight:650;
    line-height:1.15; overflow-wrap:anywhere;
}
.alyx-profile-label {
    margin-top:9px; color:#718094; font-size:.76rem; font-weight:600;
    letter-spacing:.04em; text-transform:uppercase;
}
[class*="st-key-alyx_session_"] {
    border-left:4px solid #F04F47 !important;
    border-radius:13px !important;
    background:rgba(255,255,255,.68) !important;
    box-shadow:0 2px 8px rgba(34,50,72,.04);
}
[class*="st-key-alyx_session_"] div[data-testid="stButton"] button {
    min-width:42px; min-height:42px; border-radius:50%;
    padding:0; font-size:1.05rem; font-weight:650;
}
.alyx-sessions-title {
    margin:2px 0 14px; color:#223248; font-size:1.28rem; font-weight:680;
}
.alyx-row-info { display:flex; align-items:center; gap:9px; min-width:0; }
.alyx-row-icon {
    width:30px; height:30px; flex:0 0 30px; display:grid; place-items:center;
    color:#F04F47;
}
.alyx-row-icon svg, .alyx-section-icon svg {
    width:16px; height:16px; fill:none; stroke:currentColor; stroke-width:1.8;
    stroke-linecap:round; stroke-linejoin:round;
}
.alyx-row-copy { min-width:0; display:flex; align-items:baseline; gap:8px; }
.alyx-band-value { color:#34465A; font-size:.87rem; font-weight:620; line-height:1.28; overflow-wrap:anywhere; }
.alyx-band-note { color:#66788B; font-size:.78rem; font-weight:580; line-height:1.2; white-space:nowrap; }
.alyx-bout-bar {
    display:flex; width:100%; height:9px; margin:7px 0 6px; overflow:hidden;
    border-radius:999px; background:#DDE4EA;
}
.alyx-bout-correct { background:#3B9372; }
.alyx-bout-other { background:#D99A48; }
.alyx-bout-invalid { background:#C8D1DA; }
.alyx-bout-legend { color:#718094; font-size:.67rem; white-space:nowrap; }
.alyx-bout-legend b { color:#34465A; font-weight:650; }
.alyx-category-grid {
    display:grid; grid-auto-flow:column; grid-auto-columns:minmax(235px,1fr); gap:10px;
    padding:13px 0 22px; overflow-x:auto; overscroll-behavior-inline:contain;
    scrollbar-width:thin;
}
.alyx-category-card {
    min-width:0; padding:14px 15px 16px;
    border:1px solid color-mix(in srgb,var(--category-color) 25%,transparent);
    border-radius:13px; background:var(--category-bg);
    box-shadow:0 3px 12px rgba(43,61,78,.07);
}
.alyx-category-head {
    display:flex; align-items:center; gap:8px; margin-bottom:7px;
    color:var(--category-color); font-size:.91rem; font-weight:760;
    letter-spacing:.025em;
}
.alyx-section-icon { width:24px; height:24px; display:grid; place-items:center; color:inherit; }
.alyx-section-icon svg { width:20px; height:20px; stroke-width:2; }
.alyx-category-fields { display:grid; grid-template-columns:1fr; gap:0; }
.alyx-category-field {
    min-width:0; display:flex; align-items:baseline; justify-content:flex-start;
    gap:5px; padding:3px 0; line-height:1.2;
}
.alyx-category-field span {
    flex:0 0 auto; color:#66788B; font-size:.75rem; font-weight:560;
}
.alyx-category-field span::after { content:":"; }
.alyx-category-field strong {
    min-width:0; color:#2F4054; font-size:.78rem; font-weight:650;
    text-align:left; overflow-wrap:anywhere;
}
.alyx-category-wide { grid-column:1 / -1; }
.alyx-category-record {
    grid-column:1 / -1; padding-top:6px;
    color:#536477; font-size:.78rem; line-height:1.38;
}
.alyx-category-record b { color:#2F4054; font-weight:650; }
.alyx-detail-id { margin-top:7px; color:#9AA6B2; font-size:.61rem; overflow-wrap:anywhere; }
@media (max-width:900px) {
    .alyx-profile-band { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .alyx-profile-item:nth-child(3) { border-left:0; }
}
</style>
"""


def _text(value, fallback="—"):
    if value is None or (isinstance(value, str) and not value.strip()):
        return fallback
    if not isinstance(value, (list, dict)):
        try:
            if bool(pd.isna(value)):
                return fallback
        except (TypeError, ValueError):
            pass
    return str(value)


def _number(value, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    rendered = str(int(number)) if number.is_integer() else f"{number:.1f}"
    return f"{rendered}{suffix}"


def _compact_number(value, suffix="", decimals=1):
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    rendered = str(int(number)) if number.is_integer() else f"{number:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{rendered}{suffix}"


def _friendly_datetime(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return _text(value)
    return parsed.strftime("%d/%m/%Y %H:%M")


def _rig_number(value):
    rendered = _text(value)
    match = re.search(r"(?i)rig\s*[-_]?\s*(.+)", rendered)
    return match.group(1) if match else rendered


def _looks_like_uuid(value):
    return bool(re.fullmatch(r"[0-9a-fA-F-]{32,36}", str(value or "")))


def _display_value(value):
    if isinstance(value, dict):
        value = value.get("name") or value.get("nickname") or value.get("serial")
    if isinstance(value, (list, tuple, set)):
        cleaned = [_display_value(item) for item in value]
        cleaned = [item for item in cleaned if item != "—"]
        return ", ".join(cleaned) if cleaned else "—"
    rendered = _text(value).strip()
    if _looks_like_uuid(rendered) or rendered.lower() in {"none", "nan", "null", "[]", "{}"}:
        return "—"
    return rendered


def _profile_band(mouse_id, subject):
    sex = {"M": "Male", "F": "Female", "U": "Unknown"}.get(
        str(subject.get("sex", "")).upper(), subject.get("sex")
    )
    values = [
        (mouse_id, "Mouse"),
        (sex, "Sex"),
        (subject.get("birth_date"), "Birth date"),
        (subject.get("responsible_user"), "Responsible user"),
    ]
    blocks = "".join(
        '<div class="alyx-profile-item">'
        f'<div class="alyx-profile-value">{html.escape(_text(value))}</div>'
        f'<div class="alyx-profile-label">{label}</div></div>'
        for value, label in values
    )
    return f'<div class="alyx-profile-band">{blocks}</div>'


def _category_card(title, icon, color, items, records=None, footer=None, background="#F3F7F9"):
    fields = "".join(
        '<div class="alyx-category-field">'
        f'<span>{html.escape(label)}</span><strong>{html.escape(_display_value(value))}</strong></div>'
        for label, value in items
    )
    record_html = "".join(
        f'<div class="alyx-category-record">{record}</div>' for record in (records or [])
    )
    footer_html = f'<div class="alyx-detail-id">{html.escape(footer)}</div>' if footer else ""
    return (
        f'<section class="alyx-category-card" style="--category-color:{color};--category-bg:{background}">'
        f'<div class="alyx-category-head"><span class="alyx-section-icon">{ICONS[icon]}</span>'
        f'{html.escape(title)}</div><div class="alyx-category-fields">{fields}{record_html}</div>'
        f'{footer_html}</section>'
    )


def _row_info(icon, value, note=""):
    note_html = (
        f'<div class="alyx-band-note">{html.escape(_text(note))}</div>'
        if note not in (None, "") else ""
    )
    return (
        '<div class="alyx-row-info">'
        f'<span class="alyx-row-icon">{ICONS[icon]}</span><div class="alyx-row-copy">'
        f'<div class="alyx-band-value">{html.escape(_text(value))}</div>'
        f'{note_html}</div></div>'
    )


def _safe_list(client, endpoint, subject):
    if endpoint not in client.rest_schemes.endpoints:
        return []
    try:
        return client.rest(endpoint, "list", subject=str(subject), no_cache=True) or []
    except Exception:
        try:
            return client.rest(endpoint, "list", nickname=str(subject), no_cache=True) or []
        except Exception:
            return []


def _load_mouse_context(client, subject):
    return {
        "water_administrations": _safe_list(client, "water-administrations", subject),
        "water_restrictions": _safe_list(client, "water-restriction", subject),
        "surgeries": _safe_list(client, "surgeries", subject),
        "insertions": _safe_list(client, "insertions", subject),
        "chronic_insertions": _safe_list(client, "chronic-insertions", subject),
        "weighings": _safe_list(client, "weighings", subject),
    }


def _record_id(value):
    if isinstance(value, dict):
        value = value.get("id") or value.get("url") or value.get("name")
    return str(value or "").rstrip("/").rsplit("/", 1)[-1]


def _for_session(records, session_id):
    wanted = _record_id(session_id)
    return [record for record in records if _record_id(record.get("session")) == wanted]


def _enrich_behavior_counts(data, behavior_data, mouse_id):
    """Add lick counts from the local behavior database when Alyx omits them."""
    if behavior_data is None or behavior_data.empty or "Mouse_ID" not in behavior_data:
        return data
    local = behavior_data[behavior_data["Mouse_ID"].astype(str) == str(mouse_id)].copy()
    if local.empty or "Date" not in local:
        return data
    local["_info_date"] = pd.to_datetime(local["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for index, session in data.iterrows():
        matches = local[local["_info_date"] == str(session.get("date"))]
        version = session.get("version")
        if len(matches) > 1 and "Version" in matches and pd.notna(version):
            exact = matches[matches["Version"].astype(str) == str(version)]
            if not exact.empty:
                matches = exact
        if matches.empty:
            continue
        source = matches.iloc[-1]
        if pd.isna(session.get("rewarded_licks")):
            data.at[index, "rewarded_licks"] = count_licks(source, "rewarded")
        if pd.isna(session.get("unrewarded_licks")):
            data.at[index, "unrewarded_licks"] = count_licks(source, "non_rewarded")
    return data


def _bout_distribution(row):
    correct = pd.to_numeric(row.get("correct_bouts"), errors="coerce")
    valid = pd.to_numeric(row.get("valid_bouts"), errors="coerce")
    total = pd.to_numeric(row.get("total_bouts"), errors="coerce")
    if pd.isna(total) or total <= 0:
        return _row_info("bouts", "Bouts unavailable", "No distribution")
    correct = max(0, min(float(correct) if pd.notna(correct) else 0, float(total)))
    valid = max(correct, min(float(valid) if pd.notna(valid) else correct, float(total)))
    other = valid - correct
    invalid = float(total) - valid
    widths = [100 * value / float(total) for value in (correct, other, invalid)]
    return (
        '<div class="alyx-row-info"><span class="alyx-row-icon">'
        f'{ICONS["bouts"]}</span><div class="alyx-row-copy" style="width:100%">'
        '<div class="alyx-bout-bar">'
        f'<i class="alyx-bout-correct" style="width:{widths[0]:.2f}%"></i>'
        f'<i class="alyx-bout-other" style="width:{widths[1]:.2f}%"></i>'
        f'<i class="alyx-bout-invalid" style="width:{widths[2]:.2f}%"></i></div>'
        '<div class="alyx-bout-legend">'
        f'<b>{_number(correct)}</b> correct · <b>{_number(other)}</b> other valid · '
        f'<b>{_number(invalid)}</b> invalid</div></div></div>'
    )


def _render_session_details(row, subject, context):
    cards = [
        _category_card("Performance", "performance", "#3B9372", [
            ("Correct bouts", _compact_number(row.get("correct_bouts"))),
            ("Valid / total", f'{_number(row.get("valid_bouts"))} / {_number(row.get("total_bouts"))}'),
            ("Rewarded licks", _compact_number(row.get("rewarded_licks"))),
            ("Unrewarded licks", _compact_number(row.get("unrewarded_licks"))),
            ("Bout rate / min", _compact_number(row.get("bout_rate_per_min"), decimals=1)),
            ("Reward rate / min", _compact_number(row.get("reward_rate_per_min"), decimals=1)),
            ("Duration", _compact_number(row.get("session_duration_min"), " min", 1)),
            ("Reward", _compact_number(row.get("reward_ul"), " µL", 1)),
            ("Total reward", _compact_number(row.get("total_reward_ul"), " µL", 1)),
        ], background="#E5F4EC"),
        _category_card("Session", "setup", "#4D9FC7", [
            ("Recording", row.get("recording_type")), ("Training", row.get("training_type")),
            ("Probabilities", row.get("session_probabilities")), ("Version", row.get("version")),
            ("Experimentalist", row.get("experimentalist")), ("Rig", _rig_number(row.get("rig_number"))),
            ("Manual reward", _number(row.get("manual_reward_ul"), " µL")),
            ("Protocol", row.get("protocol")),
        ], records=[
            f'<b>Comments</b><br>{html.escape(_text(row.get("session_comments"), "No comments"))}'
        ], background="#E7F3F9"),
        _category_card("Animal welfare", "welfare", "#80639A", [
            ("Alive", subject.get("alive")),
            ("Water restricted", subject.get("water_restricted")),
            ("Actual severity", subject.get("actual_severity")),
            ("Adverse effects", subject.get("adverse_effects")),
            ("Poops", row.get("poops_number")), ("Tube", row.get("tube")),
            ("Headfix", row.get("headfix")),
            ("Treadmill comfort", row.get("comfort_on_treadmill")),
        ], background="#F0EAF6"),
    ]
    session_id = row.get("session_id")
    water = _for_session(context.get("water_administrations", []), session_id)
    restrictions = context.get("water_restrictions", [])
    if water or restrictions:
        water_cards = []
        for item in water:
            amount = html.escape(_text(item.get("water_administered")))
            water_type = html.escape(_text(item.get("water_type"), "Water"))
            date = html.escape(_friendly_datetime(item.get("date_time")))
            water_cards.append(f'<b>{amount} mL · {water_type}</b><br>{date}')
        for item in restrictions[:1]:
            reference = html.escape(_text(item.get("reference_weight")))
            start = html.escape(_text(item.get("start_time"), "Start unavailable"))
            end = html.escape(_text(item.get("end_time"), "Ongoing"))
            water_cards.append(f'<b>Water restriction · reference {reference} g</b><br>{start} → {end}')
        cards.append(_category_card("Water and care", "water", "#328DA8", [
            ("Administrations", len(water)), ("Restriction records", len(restrictions)),
        ], records=water_cards, background="#E5F4F6"))

    surgeries = context.get("surgeries", [])
    if surgeries:
        latest_surgery = sorted(
            surgeries, key=lambda item: str(item.get("start_time") or ""), reverse=True
        )[0]
        procedures = latest_surgery.get("procedures") or latest_surgery.get("procedure") or []
        if isinstance(procedures, list):
            procedures = ", ".join(
                str(value) for value in procedures if not _looks_like_uuid(value)
            )
        if _looks_like_uuid(procedures) or not procedures:
            procedures = "Surgery"
        outcome = {"r": "Recovery", "a": "Acute"}.get(
            str(latest_surgery.get("outcome_type") or "").lower(),
            _text(latest_surgery.get("outcome_type")),
        )
        surgery_card = (
            f'<b>{html.escape(_text(procedures, "Procedure unavailable"))}</b><br>'
            f'{html.escape(_friendly_datetime(latest_surgery.get("start_time")))} · '
            f'Outcome: {html.escape(outcome)}'
        )
        cards.append(_category_card("Surgery", "surgery", "#D1833F", [
            ("Recorded surgeries", len(surgeries)),
            ("Implant weight", latest_surgery.get("implant_weight")),
            ("Location", latest_surgery.get("location")),
        ], records=[surgery_card], background="#FAEEDF"))

    insertions = _for_session(context.get("insertions", []), session_id)
    chronic = _for_session(context.get("chronic_insertions", []), session_id)
    if insertions or chronic:
        probe_cards = []
        for item in [*insertions, *chronic]:
            name = html.escape(_text(item.get("name") or item.get("probe"), "Probe"))
            model = html.escape(_text(item.get("model"), "Model unavailable"))
            serial = html.escape(_text(item.get("serial"), "Serial unavailable"))
            region = html.escape(_text(item.get("atlas_acronym"), "Unavailable"))
            probe_cards.append(f'<b>{name} · {model}</b><br>Serial {serial} · Target {region}')
        cards.append(_category_card("Neuropixel", "probe", "#C45D67", [
            ("Insertions", len(insertions)), ("Chronic insertions", len(chronic)),
            ("Recording type", row.get("recording_type")),
            ("Protocol", row.get("protocol")),
        ], records=probe_cards, background="#F9E8EB"))

    st.markdown(f'<div class="alyx-category-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _toggle_session_details(session_key):
    state_key = "info_alyx_expanded_session"
    current = st.session_state.get(state_key)
    st.session_state[state_key] = None if current == session_key else session_key
    st.session_state["_info_quick_rerun"] = True


def _render_session_list(data, subject, context):
    for index, row in enumerate(data.to_dict("records")):
        session_key = _text(row.get("session_id"), f"session-{index}")
        expanded_key = "info_alyx_expanded_session"
        expanded = st.session_state.get(expanded_key) == session_key
        with st.container(border=True, key=f"alyx_session_{index}"):
            date, protocol, duration, performance, rig, person, details = st.columns(
                [1.0, 1.25, .85, 2.35, .65, 1.05, .38], vertical_alignment="center"
            )
            date.markdown(_row_info("calendar", row.get("date")), unsafe_allow_html=True)
            protocol.markdown(_row_info(
                "task", _text(row.get("protocol"), "Protocol unspecified"),
                row.get("session_probabilities"),
            ), unsafe_allow_html=True)
            duration.markdown(_row_info(
                "result", _number(row.get("session_duration_min"), " min")
            ), unsafe_allow_html=True)
            performance.markdown(_row_info(
                "bouts",
                f'{_number(row.get("valid_bouts"))} valid · '
                f'{_number(row.get("rewarded_licks"))} rewarded · '
                f'{_number(row.get("unrewarded_licks"))} unrewarded',
            ), unsafe_allow_html=True)
            rig.markdown(_row_info(
                "rig", _rig_number(row.get("rig_number"))
            ), unsafe_allow_html=True)
            person.markdown(_row_info("user", row.get("experimentalist")), unsafe_allow_html=True)
            details.button(
                " ",
                key=f"toggle_alyx_session_{index}",
                help="Hide details" if expanded else "Show all session information",
                on_click=_toggle_session_details,
                args=(session_key,),
                type="tertiary",
                icon=":material/expand_less:" if expanded else ":material/expand_more:",
            )
            if expanded:
                _render_session_details(row, subject or {}, context or {})


def _tracking_figure(records, date_fields, value_field, title, unit, color):
    points = []
    for record in records or []:
        date_value = next((record.get(field) for field in date_fields if record.get(field)), None)
        date = pd.to_datetime(date_value, errors="coerce")
        value = pd.to_numeric(record.get(value_field), errors="coerce")
        if pd.notna(date) and pd.notna(value):
            points.append((date, float(value)))
    frame = pd.DataFrame(points, columns=["date", "value"]).sort_values("date") if points else pd.DataFrame()
    figure = go.Figure()
    if not frame.empty:
        figure.add_trace(go.Scatter(
            x=frame["date"], y=frame["value"], mode="lines+markers",
            line=dict(color=color, width=3, shape="spline"),
            marker=dict(size=7, color=color, line=dict(color="white", width=1.5)),
            hovertemplate=f"%{{x|%d/%m/%Y}}<br>%{{y:.2f}} {unit}<extra></extra>",
        ))
    figure.update_layout(
        title=dict(text=title, x=.04, y=.93, font=dict(size=16, color="#2F4054")),
        height=310, margin=dict(l=42, r=20, t=55, b=38),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,.62)",
        showlegend=False, hovermode="x unified",
        xaxis=dict(title=None, showgrid=False, zeroline=False),
        yaxis=dict(title=unit, gridcolor="rgba(115,137,157,.13)", zeroline=False),
    )
    if frame.empty:
        figure.add_annotation(text="No data", x=.5, y=.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(color="#F04F47", size=14))
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    return figure


def _render_tracking(context):
    def points(records, value_field):
        values = []
        for record in records or []:
            date = pd.to_datetime(record.get("date_time") or record.get("date"), errors="coerce")
            value = pd.to_numeric(record.get(value_field), errors="coerce")
            if pd.notna(date) and pd.notna(value):
                values.append((date, float(value)))
        return sorted(values, key=lambda item: item[0])

    water = points(context.get("water_administrations", []), "water_administered")
    weight = points(context.get("weighings", []), "weight")
    figure = go.Figure()
    if water:
        figure.add_trace(go.Scatter(
            x=[item[0] for item in water], y=[item[1] for item in water],
            name="Water restriction", mode="lines+markers", yaxis="y",
            line=dict(color="#328DA8", width=3, shape="spline"),
            marker=dict(size=7, color="#328DA8"),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} mL<extra>Water restriction</extra>",
        ))
    if weight:
        figure.add_trace(go.Scatter(
            x=[item[0] for item in weight], y=[item[1] for item in weight],
            name="Mouse weight", mode="lines+markers", yaxis="y2",
            line=dict(color="#F04F47", width=3, shape="spline"),
            marker=dict(size=7, color="#F04F47"),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} g<extra>Mouse weight</extra>",
        ))
    figure.update_layout(
        height=380, margin=dict(l=48, r=48, t=35, b=38),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified", legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title=None, showgrid=False, zeroline=False),
        yaxis=dict(title="Water (mL)", color="#328DA8", gridcolor="rgba(115,137,157,.12)", zeroline=False),
        yaxis2=dict(title="Weight (g)", color="#F04F47", overlaying="y", side="right", showgrid=False, zeroline=False),
    )
    if not water and not weight:
        figure.add_annotation(text="No data", x=.5, y=.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(color="#F04F47", size=14))
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})


def _login():
    username = st.session_state.get("info_alyx_username", "").strip()
    password = st.session_state.pop("info_alyx_password", "")
    _authenticate(username, password)


def _authenticate(username, password):
    try:
        client = connect(DEFAULT_URL, username, password)
    except ImportError:
        st.session_state["info_alyx_error"] = "Install ONE-api and python-dotenv from requirements.txt to connect."
    except Exception as error:
        detail = str(error).strip().replace(username, "<user>")
        st.session_state["info_alyx_error"] = (
            "Connection failed. Check the Alyx server/network access."
            + (f"\n\nTechnical detail: {detail}" if detail else "")
        )
    else:
        st.session_state["info_alyx_username"] = username
        st.session_state["info_alyx_client"] = client
        st.session_state.pop("info_alyx_error", None)
        st.session_state.pop("info_alyx_data", None)


def _logout():
    for key in list(st.session_state):
        if key.startswith("info_alyx_"):
            del st.session_state[key]
    st.session_state["info_alyx_auto_attempted"] = True


def _secret_credentials():
    try:
        config = st.secrets.get("alyx", {})
        return str(config.get("username", "")).strip(), str(config.get("password", ""))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return "", ""


def render_info_view(mouse_id, behavior_data=None):
    st.markdown(INFO_STYLES, unsafe_allow_html=True)
    if mouse_id is None:
        render_no_data()
        return

    username, password = _secret_credentials()
    configured_credentials = bool(username and password)
    if (configured_credentials and "info_alyx_client" not in st.session_state
            and not st.session_state.get("info_alyx_auto_attempted")):
        st.session_state["info_alyx_auto_attempted"] = True
        with st.spinner("Connecting to Alyx…"):
            _authenticate(username, password)
    client = st.session_state.get("info_alyx_client")
    if client is None:
        if configured_credentials:
            st.caption("Using the Alyx account configured in `.streamlit/secrets.toml`.")
            if st.button("Retry Alyx connection", type="primary"):
                _authenticate(username, password)
                st.rerun()
        else:
            st.write("Sign in with your Alyx account to access this mouse's information.")
            st.caption(DEFAULT_URL)
            with st.form("info_alyx_login"):
                st.text_input("Alyx username", key="info_alyx_username")
                st.text_input("Alyx password", type="password", key="info_alyx_password")
                st.form_submit_button("Log in", on_click=_login)
        if st.session_state.get("info_alyx_error"):
            st.error(st.session_state["info_alyx_error"])
        return

    cached = st.session_state.get("info_alyx_data")
    if (cached is None or cached.get("mouse") != mouse_id
            or cached.get("version") != INFO_CACHE_VERSION):
        cached = {"mouse": mouse_id, "version": INFO_CACHE_VERSION}
        with st.spinner(f"Reading Alyx information for {mouse_id}…"):
            try:
                cached["subject"] = client.rest("subjects", "read", id=str(mouse_id), no_cache=True)
            except Exception as error:
                cached["subject"] = None
                cached["subject_error"] = str(error)
            try:
                cached["rows"] = normalize_sessions(load_raw_sessions(client, DEFAULT_LAB, subject=str(mouse_id)))
            except Exception as error:
                cached["rows"] = None
                cached["rows_error"] = str(error)
            cached["context"] = _load_mouse_context(client, mouse_id)
        # Failed requests can be retried using Refresh; never show another mouse's data.
        st.session_state["info_alyx_data"] = cached

    if cached["rows"] is None:
        render_no_data()
        return
    data = pd.DataFrame(cached["rows"], columns=COLUMNS)
    data = _enrich_behavior_counts(data, behavior_data, mouse_id)
    if data.empty:
        render_no_data()
        return
    subject = cached["subject"] or {}
    if subject:
        st.markdown(_profile_band(mouse_id, subject), unsafe_allow_html=True)
    sessions_tab, tracking_tab = st.tabs(["Sessions", "Tracking"])
    with sessions_tab:
        data = data.sort_values(["date", "session_time"], ascending=False, na_position="last")
        _render_session_list(data, subject, cached.get("context", {}))
    with tracking_tab:
        _render_tracking(cached.get("context", {}))
