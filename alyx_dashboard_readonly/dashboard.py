"""Standalone read-only Streamlit dashboard for VFT sessions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from alyx_rest import DEFAULT_LAB, DEFAULT_URL, connect, load_raw_sessions
from normalize import COLUMNS, normalize_sessions


st.set_page_config(page_title="VFT Alyx sessions", layout="wide")
st.title("Virtual Foraging Task — Alyx sessions")
st.caption("Read-only Alyx REST view. One Alyx session per row.")

if "alyx_client" not in st.session_state:
    with st.form("alyx_login"):
        base_url = st.text_input("Alyx URL", value=DEFAULT_URL)
        username = st.text_input("Alyx username")
        password = st.text_input("Alyx password", type="password")
        submitted = st.form_submit_button("Log in")
    if submitted:
        try:
            st.session_state.alyx_client = connect(base_url, username, password)
            st.session_state.alyx_username = username.strip()
            st.session_state.alyx_base_url = base_url.rstrip("/")
            st.session_state.pop("session_rows", None)
            st.rerun()
        except Exception as exc:
            st.error(f"Alyx login failed: {exc}")
    st.stop()

login_info, logout_column = st.columns([5, 1])
login_info.caption(
    f"Connected to {st.session_state.alyx_base_url} as "
    f"{st.session_state.alyx_username}."
)
if logout_column.button("Log out"):
    for key in ("alyx_client", "alyx_username", "alyx_base_url", "session_rows"):
        st.session_state.pop(key, None)
    st.rerun()

refresh = st.button("Refresh Alyx sessions")
if refresh:
    st.session_state.pop("session_rows", None)

if "session_rows" not in st.session_state:
    try:
        with st.spinner("Reading session details from Alyx…"):
            raw = load_raw_sessions(st.session_state.alyx_client, DEFAULT_LAB)
            st.session_state.session_rows = normalize_sessions(raw)
    except Exception as exc:
        st.error(f"Could not read Alyx sessions: {exc}")
        st.stop()

data = pd.DataFrame(st.session_state.session_rows, columns=COLUMNS)

if data.empty:
    st.info("No sessions were returned by the read-only query.")
    st.stop()

mouse_options = sorted(value for value in data["mouse_id"].dropna().unique())
experimentalist_options = sorted({
    name.strip()
    for value in data["experimentalist"].dropna()
    for name in str(value).split(",")
    if name.strip()
})

filter_mouse, filter_user = st.columns(2)
selected_mice = filter_mouse.multiselect("Mouse", mouse_options)
selected_users = filter_user.multiselect("Experimentalist", experimentalist_options)

filtered = data
if selected_mice:
    filtered = filtered[filtered["mouse_id"].isin(selected_mice)]
if selected_users:
    selected = set(selected_users)
    filtered = filtered[
        filtered["experimentalist"].fillna("").map(
            lambda value: bool(selected.intersection(
                name.strip() for name in str(value).split(",") if name.strip()
            ))
        )
    ]

st.write(f"{len(filtered):,} session(s)")
st.dataframe(
    filtered,
    hide_index=True,
    use_container_width=True,
    height=650,
    column_config={
        "session_id": None,
        "session_comments": st.column_config.TextColumn(width="large"),
    },
)
st.download_button(
    "Download filtered sessions as CSV",
    filtered.to_csv(index=False).encode("utf-8-sig"),
    file_name="vft_alyx_sessions.csv",
    mime="text/csv",
)

with st.expander("Data completeness"):
    completeness = (
        filtered.notna().mean().mul(100).round(1).rename("coverage_percent").reset_index()
    )
    st.dataframe(completeness, hide_index=True, use_container_width=True)
    st.caption(
        "Blank cells mean the value is not stored in the Alyx Session/OtherAction records. "
        "The dashboard does not guess structured values from free-text comments."
    )
