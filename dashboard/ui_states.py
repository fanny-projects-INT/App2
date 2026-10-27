"""Native tab navigation, empty states and background data prefetching."""

UI_STATES_VERSION = 1

import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


class DataPrefetcher:
    """One bounded worker for data-only cache warming; never renders UI."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dashboard-prefetch")
        self.jobs = {}
        self.lock = Lock()

    def submit(self, key, work):
        with self.lock:
            if key in self.jobs:
                return
            # Don't accumulate work when users quickly change mice.
            if sum(not job.done() for job in self.jobs.values()) >= 2:
                return
            for old_key in list(self.jobs):
                if len(self.jobs) < 32:
                    break
                if self.jobs[old_key].done():
                    del self.jobs[old_key]
            self.jobs[key] = self.executor.submit(work)


@st.cache_resource(show_spinner=False)
def data_prefetcher():
    return DataPrefetcher()


def render_no_data(text="No data"):
    st.markdown(
        f"""
        <style>
        .mouse-empty-state {{
            min-height:60vh; display:flex; align-items:center; justify-content:center;
            color:#F04F47; text-align:center;
        }}
        .mouse-empty-state p {{ margin:0; font-size:.92rem; font-weight:650; }}
        </style>
        <div class="mouse-empty-state" role="status">
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dashboard_tabs(labels, *, key):
    """One navigation policy for every page: active content in a fragment."""
    saved_key = f"_dashboard_tab_selection_{key}"
    selected = st.session_state.get(key, st.session_state.get(saved_key, labels[0]))
    if selected not in labels:
        selected = labels[0]
    if st.session_state.get(key) != selected:
        st.session_state[key] = selected
    tabs = st.tabs(labels, key=key, on_change="rerun")
    st.session_state[saved_key] = st.session_state[key]
    return tabs
