import json
import html
import base64
import ast
import hashlib
import os
import re
import sqlite3
import subprocess
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba
from PIL import Image
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

from app_functions import (
    PROTOCOL_LABELS,
    build_session_plot_bout_timeline,
    build_session_plot_failure_distribution,
    build_session_plot_rewards_vs_failures,
    compute_failures,
    count_valid_bouts,
    count_licks,
    count_reward_per_bout,
    extract_bout_timeline_data,
    fit_exponential_trend_with_band,
    plot_bout_count_rewards,
    plot_histogram_kde_failures,
    plot_kde_failures_by_session,
    plot_protocol_strip,
    plot_regression_rewards_failures_and_slope,
    plot_stacked_lick_counts,
    prepare_mouse_dataframe,
    prepare_session_arrays,
    valid_bout_mask_from_row,
)


st.set_page_config(
    page_title="Behavior & electrophysiology dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_BEHAVIOR_DB = Path(
    r"\\SynoINVIBE_Caze\INVIBE_team_Cazettes\data\database\full_db_all_rigs.feather"
)
DEFAULT_EPHYS_ROOT = Path(r"F:\coregistration_db")
DEFAULT_OPENFIELD_ROOT = Path(r"F:\Behavior")
DEFAULT_BEHAVIOR_METRICS_DB = Path(r"F:\behavior_metrics.sqlite")
DEFAULT_BOMBCELL_CONFIG = Path.home() / "Documents" / "ephys" / "pipeline" / "config.py"

BEHAVIOR_DB = Path(os.getenv("APP_BEHAVIOR_DB", str(DEFAULT_BEHAVIOR_DB)))
EPHYS_ROOT = Path(os.getenv("APP_EPHYS_ROOT", str(DEFAULT_EPHYS_ROOT)))
OPENFIELD_ROOT = Path(os.getenv("APP_OPENFIELD_ROOT", str(DEFAULT_OPENFIELD_ROOT)))
BEHAVIOR_METRICS_DB = Path(
    os.getenv("APP_BEHAVIOR_METRICS_DB", str(DEFAULT_BEHAVIOR_METRICS_DB))
)
BOMBCELL_CONFIG = Path(
    os.getenv("APP_BOMBCELL_CONFIG", str(DEFAULT_BOMBCELL_CONFIG))
)
DEFAULT_ATLAS_DIR = Path.home() / ".brainglobe" / "allen_mouse_25um_v1.2"
ATLAS_DIR = Path(os.getenv("APP_ATLAS_DIR", str(DEFAULT_ATLAS_DIR)))
# Allen CCF coordinates of Bregma, ordered ML, AP, DV, in micrometres.
ALLEN_BREGMA_MLAPDV_UM = np.array([5739.0, 5400.0, 332.0])
PROBE_TARGETS = {"probe00": "ALM", "probe01": "STR"}
VIEW_OPTIONS = ["Home", "Overview", "Electrophysiology", "Openfield", "Info", "Chatbot"]
VIEW_LABELS = {
    "Home": "Home",
    "Overview": "Behavior",
    "Openfield": "Openfield",
    "Electrophysiology": "Ephys",
    "Info": "Info",
    "Chatbot": "Agent",
}
DEFAULT_VIEW = os.getenv("APP_DEFAULT_VIEW", "Home")
NAVIGATION_STATE_KEY = "app_active_view"
EXCLUDED_MOUSE_IDS = {"000", "001", "VF000", "VF001"}

NAVY = "#263548"
CARD_BORDER = "#E1E7EC"
PAGE_BG = "#E8EDF1"
PANEL_BG = "#F8FAFB"
MUTED = "#748091"
WHITE = "#FFFFFF"


def inject_css():
    st.markdown(
        f"""
        <style>
        html, body, [data-testid="stAppViewContainer"] {{
            scrollbar-gutter: stable !important;
            max-width: 100vw !important;
            overflow-x: hidden !important;
        }}
        [data-testid="stAppViewContainer"] {{
            overflow-y: auto !important;
        }}
        section[data-testid="stMain"],
        [data-testid="stMain"] {{
            overflow-y: auto !important;
            scrollbar-gutter: stable !important;
        }}
        .stApp {{
            width: 100% !important;
            max-width: 100vw !important;
            overflow-x: hidden !important;
            background: {PAGE_BG};
        }}
        section[data-testid="stSidebar"] {{
            background: {WHITE};
            border-right: 1px solid {CARD_BORDER};
        }}
        .block-container {{
            width: min(100%, 1600px);
            max-width: 1600px;
            box-sizing: border-box;
            overflow-x: clip;
            margin-left: auto;
            margin-right: auto;
            padding-top: 0.70rem;
            padding-left: clamp(1rem, 1.5vw, 1.5rem);
            padding-right: clamp(1rem, 1.5vw, 1.5rem);
            padding-bottom: 1.25rem;
        }}
        /* The fixed Streamlit header is transparent but still overlays the top
           navigation. Remove the layer entirely so every pixel remains clickable. */
        header[data-testid="stHeader"] {{
            display: none !important;
            height: 0 !important;
            min-height: 0 !important;
        }}
        .sidebar-title {{
            font-size: 1.4rem;
            font-weight: 800;
            color: {NAVY};
            text-align: center;
            margin: 0 0 1.4rem 0;
            letter-spacing: 0.4px;
            line-height: 1.2;
        }}
        .sidebar-title::after {{
            content: "";
            display: block;
            width: 42px;
            height: 3px;
            background: #2563EB;
            margin: 8px auto 0 auto;
            border-radius: 2px;
        }}
        .page-title {{
            font-size: 1.95rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.15;
            margin: 0;
        }}
        .st-key-navigation_fixed_shell {{
            position: relative !important;
            top: auto !important;
            left: auto !important;
            transform: none !important;
            width: 100% !important;
            min-height: 58px !important;
            z-index: 1 !important;
        }}
        .st-key-navigation_fixed_shell > div {{
            width: 100% !important;
        }}
        .navigation-flow-spacer {{ height: 22px; }}
        .st-key-navigation_fixed_shell > div > div[data-testid="stHorizontalBlock"] {{
            flex-direction: row !important;
            gap: 1rem !important;
        }}
        .st-key-app_navigation {{
            background: #D1E1EE;
            border-color: #BCCFDE !important;
            border-radius: 16px;
            padding: 0 14px 0 16px;
            margin-bottom: 0;
            box-shadow: 0 6px 20px rgba(34,50,72,0.12);
            position: relative !important;
            z-index: 1;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
        }}
        .st-key-app_navigation [data-testid="stSelectbox"] label,
        .st-key-app_navigation [data-testid="stWidgetLabel"] {{
            color: {MUTED};
            font-size: .76rem;
            font-weight: 650;
        }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] {{
            display: flex;
            justify-content: flex-start;
            padding-top: 0;
        }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] > div {{
            width: 100%;
        }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button {{
            flex: 1 1 0;
            min-height: 64px;
            cursor: pointer;
            justify-content: flex-start !important;
            padding-left: 20px !important;
            padding-right: 20px !important;
            font-size: 1rem !important;
            font-weight: 650 !important;
        }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button::before {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.35rem;
            margin-right: 8px;
            color: #52677D;
            font-size: 1.15rem;
            font-weight: 500;
            line-height: 1;
        }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button:nth-of-type(1)::before {{ content: "⌂"; }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button:nth-of-type(2)::before {{ content: "▥"; }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button:nth-of-type(3)::before {{ content: "◎"; }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button:nth-of-type(4)::before {{ content: "∿"; }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button:nth-of-type(5)::before {{ content: "ⓘ"; }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button:nth-of-type(6)::before {{ content: "◇"; }}
        .st-key-app_navigation div[data-testid="stSegmentedControl"] button p {{
            width: 100%;
            text-align: left !important;
            pointer-events: none !important;
        }}
        .st-key-app_navigation .st-key-global_home_button button {{
            min-height: 46px;
            width: 100%;
            border-radius: 10px !important;
            font-size: 1rem !important;
            font-weight: 650 !important;
        }}
        .st-key-app_navigation [data-testid="stSelectbox"],
        .st-key-app_navigation [data-testid="stSelectbox"] > div {{
            height: 64px;
        }}
        .st-key-app_navigation [data-testid="stSelectbox"] {{
            width: 50% !important;
            margin-left: auto;
        }}
        .st-key-app_navigation [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
            min-height: 64px !important;
            height: 64px !important;
            border-radius: 0 10px 10px 0 !important;
            display: flex;
            align-items: center;
        }}
        .st-key-app_navigation > div > div[data-testid="stHorizontalBlock"] {{
            align-items: center;
        }}
        .st-key-navigation_mouse {{
            height: 64px;
            display: flex;
            align-items: center;
        }}
        .st-key-navigation_mouse > div {{
            width: 100%;
            height: 64px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}
        /* Native buttons keep the whole navigation surface clickable. */
        .st-key-app_navigation {{
            min-height: 58px !important;
            padding: 0 !important;
            overflow: hidden;
            background: #F04F47;
            border: 1px solid #F04F47 !important;
            border-radius: 15px !important;
        }}
        .st-key-app_navigation div[data-testid="stHorizontalBlock"] {{
            flex-direction: row !important;
            gap: 0 !important;
            align-items: stretch !important;
        }}
        .st-key-app_navigation div[data-testid="stColumn"] {{
            width: 0 !important;
            flex: 1 1 0 !important;
            min-height: 56px !important;
        }}
        .st-key-app_navigation .stButton,
        .st-key-app_navigation .stButton > button {{
            width: 100% !important;
            height: 56px !important;
            min-height: 56px !important;
            margin: 0 !important;
            position: relative !important;
            z-index: 10 !important;
        }}
        .st-key-app_navigation .stButton > button {{
            border-radius: 0 !important;
            cursor: pointer !important;
            pointer-events: auto !important;
            font-size: .91rem !important;
            font-weight: 500 !important;
        }}
        .st-key-app_navigation button[data-testid="stBaseButton-primary"],
        .st-key-app_navigation button[kind="primary"] {{
            background: #F04F47 !important;
            border-color: #F04F47 !important;
            color: #FFFFFF !important;
        }}
        .st-key-app_navigation .stButton > button * {{
            pointer-events: none !important;
        }}
        .st-key-app_navigation div[data-testid="stColumn"]:first-child .stButton > button {{
            border-radius: 11px 0 0 11px !important;
        }}
        .st-key-app_navigation div[data-testid="stColumn"]:last-child .stButton > button {{
            border-radius: 0 11px 11px 0 !important;
        }}
        .st-key-navigation_mouse,
        .st-key-navigation_mouse > div,
        .st-key-navigation_mouse [data-testid="stSelectbox"],
        .st-key-navigation_mouse [data-testid="stSelectbox"] > div,
        .st-key-navigation_mouse [data-baseweb="select"],
        .st-key-navigation_mouse [data-baseweb="select"] > div {{
            width: 100% !important;
            height: 56px !important;
            min-height: 56px !important;
            margin: 0 !important;
        }}
        .st-key-navigation_mouse [data-baseweb="select"] > div {{
            border-radius: 0 11px 11px 0 !important;
        }}
        .st-key-app_navigation [data-testid="stSelectbox"],
        .st-key-app_navigation [data-testid="stSelectbox"] > div,
        .st-key-app_navigation [data-baseweb="select"],
        .st-key-app_navigation [data-baseweb="select"] > div {{
            width: 100% !important;
            height: 56px !important;
            min-height: 56px !important;
            margin: 0 !important;
        }}
        .st-key-app_navigation [data-baseweb="select"] > div {{
            border-radius: 0 11px 11px 0 !important;
        }}
        .st-key-app_navigation div[data-testid="stColumn"]:last-child div[data-testid="stVerticalBlock"] {{
            gap: 0 !important;
        }}
        .st-key-navigation_mouse_shell {{
            min-height: 58px !important;
            height: 58px !important;
            padding: 0 !important;
            margin-bottom: 22px;
            overflow: hidden;
            background: #FFFFFF;
            border: 1px solid #B9CEDD !important;
            border-radius: 15px !important;
            box-shadow: 0 6px 20px rgba(34,50,72,.10);
            position: relative;
        }}
        .st-key-navigation_mouse_shell [data-testid="stSelectbox"] {{
            position: absolute !important;
            inset: 0 !important;
            z-index: 2;
            cursor: pointer;
        }}
        .st-key-navigation_mouse_shell [data-testid="stSelectbox"],
        .st-key-navigation_mouse_shell [data-testid="stSelectbox"] > div,
        .st-key-navigation_mouse_shell [data-baseweb="select"],
        .st-key-navigation_mouse_shell [data-baseweb="select"] > div {{
            width: 100% !important;
            height: 56px !important;
            min-height: 56px !important;
            margin: 0 !important;
            border: 0 !important;
        }}
        .st-key-navigation_mouse_shell [data-baseweb="select"] > div {{
            border-radius: 14px !important;
            cursor: pointer !important;
        }}
        div[class*="st-key-navigation_mouse_"] [data-baseweb="select"] > div {{
            background: #FFFFFF !important;
            position: relative !important;
        }}
        div[class*="st-key-navigation_mouse_"] [data-baseweb="select"] > div > div:first-child {{
            position: absolute !important;
            inset: 0 !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 0 34px !important;
            pointer-events: none !important;
        }}
        div[class*="st-key-navigation_mouse_"] [data-baseweb="select"] > div > div:first-child > div {{
            width: 100% !important;
            text-align: center !important;
        }}
        .st-key-behavior_session_navigation {{
            background: #FFFFFF;
            border: 1px solid {CARD_BORDER};
            border-radius: 13px;
            padding: 8px 11px;
            margin: 4px 0 18px 0;
        }}
        .st-key-behavior_view_navigation {{
            display:flex;
            justify-content:center;
            margin: 0 0 22px 0;
        }}
        .st-key-behavior_view_navigation div[data-testid="stSegmentedControl"] {{
            width: min(100%, 430px);
        }}
        .st-key-behavior_view_navigation div[data-testid="stSegmentedControl"] > div {{
            width:100%; gap:0 !important;
        }}
        .st-key-behavior_view_navigation button {{
            flex:1 1 0; min-height:40px; border-radius:0 !important;
        }}
        .st-key-behavior_view_navigation button:first-child {{
            border-radius:11px 0 0 11px !important;
        }}
        .st-key-behavior_view_navigation button:last-child {{
            border-radius:0 11px 11px 0 !important;
        }}
        .st-key-home_summary,
        .st-key-home_mouse_grid {{
            background: {PANEL_BG} !important;
            border: 0 !important;
            border-radius: 16px;
            padding: 18px 20px 20px 20px;
        }}
        .st-key-home_mouse_grid {{
            position: relative;
            padding-top: 62px;
        }}
        .home-summary-band {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); align-items:center; min-height:126px; }}
        .home-summary-block {{ min-width:0; min-height:92px; display:flex; align-items:center; justify-content:center; gap:16px; padding:8px 22px; border-left:1px solid rgba(115,137,157,.18); }}
        .home-summary-block:first-child {{ border-left:0; }}
        .home-summary-icon {{ width:72px; height:72px; flex:0 0 72px; display:grid; place-items:center; border-radius:22px; color:#fff; }}
        .home-summary-icon svg {{ width:39px; height:39px; }}
        .home-summary-icon.openfield {{ background:#80639A; }}
        .home-summary-icon.ephys {{ background:#3B9372; }}
        .home-summary-mice {{ display:flex; flex-direction:column; align-items:center; justify-content:center; line-height:1; }}
        .home-summary-mice strong {{ color:#223248; font-size:3.15rem; font-weight:650; letter-spacing:-.05em; }}
        .home-summary-mice span {{ margin-top:9px; color:#718094; font-size:.78rem; font-weight:550; }}
        .home-summary-value {{ color:#223248; font-size:1.12rem; font-weight:520; white-space:nowrap; }}
        .home-summary-value strong {{ font-size:1.72rem; font-weight:650; margin-right:7px; }}
        .home-summary-pie {{ width:78px; height:78px; flex:0 0 78px; border-radius:50%; box-shadow:inset 0 0 0 1px rgba(34,50,72,.08); }}
        .home-summary-legend {{ display:flex; flex-direction:column; gap:7px; color:#536477; font-size:.76rem; line-height:1.05; }}
        .home-summary-legend span {{ white-space:nowrap; }}
        .home-summary-legend i {{ display:inline-block; width:8px; height:8px; margin-right:7px; border-radius:50%; }}
        .home-summary-legend b {{ color:#223248; font-weight:650; margin-right:3px; }}
        .behavior-mouse-band {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); align-items:center; min-height:126px; }}
        .behavior-mouse-block {{ min-width:0; min-height:92px; display:flex; align-items:center; justify-content:center; gap:16px; padding:8px 22px; border-left:1px solid rgba(115,137,157,.18); }}
        .behavior-mouse-block:first-child {{ border-left:0; }}
        .behavior-mouse-primary {{ text-align:center; line-height:1.05; }}
        .behavior-mouse-primary strong {{ display:block; color:#223248; font-size:1.72rem; font-weight:650; letter-spacing:-.025em; }}
        .behavior-mouse-primary span {{ display:block; margin-top:9px; color:#718094; font-size:.78rem; font-weight:550; letter-spacing:.035em; }}
        .behavior-mouse-pie {{ width:78px; height:78px; flex:0 0 78px; border-radius:50%; box-shadow:inset 0 0 0 1px rgba(34,50,72,.08); }}
        .behavior-mouse-legend {{ display:flex; flex-direction:column; gap:7px; color:#536477; font-size:.76rem; line-height:1.05; }}
        .behavior-mouse-legend span {{ white-space:nowrap; }}
        .behavior-mouse-legend i {{ display:inline-block; width:8px; height:8px; margin-right:7px; border-radius:50%; }}
        .behavior-mouse-legend b {{ color:#223248; font-weight:620; margin-right:3px; }}
        .openfield-metric-grid {{
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px 14px;
            padding:12px 0 2px 0;
        }}
        .openfield-metric-card {{
            min-width:0;
            min-height:62px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            gap:5px;
            padding:8px 11px;
            border-radius:12px;
            background:#F3F6F8;
            text-align:center;
        }}
        .openfield-metric-label {{
            max-width:100%;
            color:var(--metric-accent,#4D9FC7);
            font-size:.76rem;
            font-weight:650;
            line-height:1.22;
            overflow-wrap:anywhere;
        }}
        .openfield-metric-value {{
            color:{NAVY};
            font-size:.96rem;
            font-weight:560;
            line-height:1.1;
        }}
        .st-key-behavior_section_global_metrics .openfield-metric-grid {{
            padding-bottom:22px;
        }}
        @media (max-width:1050px) {{
            .home-summary-band {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            .home-summary-block:nth-child(3) {{ border-left:0; }}
            .behavior-mouse-band {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
            .behavior-mouse-block:nth-child(3) {{ border-left:0; }}
            .openfield-metric-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
        }}
        @media (max-width:680px) {{
            .openfield-metric-grid {{ grid-template-columns:1fr; }}
        }}
        .st-key-chatbot_shell {{
            background: transparent !important;
            border: 0 !important;
            border-radius: 0;
            padding: 0 0 88px 0;
            box-sizing: border-box;
            min-height: 0;
        }}
        .chatbot-heading {{
            color: {NAVY};
            font-size: 1.08rem;
            font-weight: 720;
            margin: 0 0 4px 0;
        }}
        .chatbot-description {{
            color: {MUTED};
            font-size: .86rem;
            margin: 0 0 18px 0;
        }}
        .st-key-chatbot_shell div[data-testid="stChatMessage"] {{
            background: #FFFFFF;
            border: 1px solid #DCE5ED;
            border-radius: 13px;
            padding: 10px 13px;
            min-height: 0 !important;
            height: auto !important;
        }}
        .st-key-chatbot_shell div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
            width: 86%;
            margin-left: auto;
            background: #FFFFFF;
            border-color: #D7E1EA;
        }}
        .st-key-chatbot_shell div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{
            width: 94%;
            margin-right: auto;
            background: #FFFFFF;
        }}
        .st-key-chatbot_shell [data-testid="stSpinner"],
        .st-key-chatbot_shell [data-testid="stSpinner"] > div,
        .st-key-chatbot_shell [data-testid="stStatusWidget"],
        .st-key-chatbot_shell [data-testid="stStatusWidget"] > div {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }}
        .st-key-chatbot_shell div[data-testid="stElementContainer"]:has([data-testid="stSpinner"]),
        .st-key-chatbot_shell div[data-testid="stElementContainer"]:has([data-testid="stStatusWidget"]) {{
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
        }}
        .st-key-chatbot_shell div[data-testid="stChatInput"] {{
            background: #FFFFFF !important;
            border: 1px solid #D4DFE8 !important;
            border-radius: 13px !important;
            box-shadow: 0 2px 8px rgba(34,50,72,.05);
            box-sizing: border-box;
            position: fixed !important;
            left: calc(50% - 102px) !important;
            right: auto !important;
            bottom: 18px !important;
            transform: none !important;
            width: min(calc(100vw - 320px), 1306px) !important;
            transform: translateX(-50%) !important;
            margin: 0 !important;
            z-index: 1000 !important;
            overflow: visible;
        }}
        .st-key-chatbot_shell div[data-testid="stChatInput"] > div,
        .st-key-chatbot_shell div[data-testid="stChatInput"] textarea {{
            background: #FFFFFF !important;
            border-radius: 12px !important;
        }}
        .st-key-chatbot_controls {{
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .st-key-chatbot_controls div[data-testid="stHorizontalBlock"] {{
            align-items: center;
            gap: 8px;
        }}
        .st-key-chatbot_controls button {{
            height: 38px !important;
            min-height: 38px !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            border-radius: 10px !important;
        }}
        .st-key-chatbot_clear {{
            position: fixed !important;
            right: max(194px, calc((100vw - 1234px) / 2)) !important;
            bottom: 18px !important;
            width: 56px !important;
            height: 56px !important;
            z-index: 1001 !important;
            margin: 0 !important;
        }}
        .st-key-chatbot_clear button {{
            width: 56px !important;
            height: 56px !important;
            min-height: 56px !important;
            padding: 0 !important;
            border-radius: 13px !important;
            background: #FFFFFF !important;
        }}
        .st-key-chatbot_astra {{
            position: fixed !important;
            right: max(48px, calc((100vw - 1510px) / 2)) !important;
            bottom: 18px !important;
            width: 128px !important;
            height: 56px !important;
            z-index: 1001 !important;
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border: 0 !important;
            border-radius: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
        }}
        .st-key-chatbot_astra [data-testid="stToggle"] {{
            width: 100% !important;
            min-height: 0 !important;
            margin: 0 !important;
        }}
        .st-key-chatbot_astra [data-testid="stToggle"] label {{
            width: 100% !important;
            margin: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 7px !important;
            color: {NAVY} !important;
            font-size: .70rem !important;
            white-space: nowrap !important;
        }}
        .st-key-chatbot_astra [data-testid="stToggle"] label > div {{
            flex: 0 0 auto !important;
            width: auto !important;
            margin: 0 !important;
        }}
        .st-key-chatbot_astra [data-testid="stWidgetLabel"],
        .st-key-chatbot_astra [data-testid="stWidgetLabel"] p {{
            width: auto !important;
            flex: 0 0 auto !important;
            margin: 0 !important;
            text-align: center !important;
        }}
        .st-key-chatbot_controls div[data-testid="stSegmentedControl"] {{
            width: 100%;
        }}
        .st-key-chatbot_controls div[data-testid="stSegmentedControl"] > div {{
            width: 100%;
            gap: 0 !important;
        }}
        .st-key-chatbot_controls div[data-testid="stSegmentedControl"] button {{
            flex: 1 1 0;
            border-radius: 0 !important;
            margin-left: -1px;
        }}
        .st-key-chatbot_controls div[data-testid="stSegmentedControl"] button:first-child {{
            margin-left: 0;
            border-radius: 10px 0 0 10px !important;
        }}
        .st-key-chatbot_controls div[data-testid="stSegmentedControl"] button:last-child {{
            border-radius: 0 10px 10px 0 !important;
        }}
        .st-key-chatbot_controls [data-testid="stToggle"] {{
            min-height: 38px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: fit-content;
            margin: 0 auto;
        }}
        .st-key-chatbot_controls [data-testid="stToggle"] label {{
            margin-bottom: 0;
            font-size: .84rem;
            color: {NAVY};
        }}
        .chatbot-mode-label {{
            color: {MUTED};
            font-size: .62rem;
            font-weight: 650;
            line-height: 1;
            text-align: center;
            height: 9px;
            margin: 0 0 3px 0;
            width: max-content;
            position: relative;
            left: 50%;
            transform: translateX(-50%);
        }}
        .chatbot-clear-spacer {{
            height: 12px;
        }}
        .st-key-chatbot_history {{
            position: fixed !important;
            top: 96px !important;
            left: max(24px, calc((100vw - 1552px) / 2)) !important;
            right: max(24px, calc((100vw - 1552px) / 2)) !important;
            bottom: 88px !important;
            width: auto !important;
            margin: 0 !important;
            height: auto !important;
            min-height: 330px !important;
            overflow-y: auto !important;
            z-index: 10 !important;
        }}
        .st-key-chatbot_history div[data-testid="stVerticalBlockBorderWrapper"] {{
            height: 100% !important;
            min-height: 330px !important;
            max-height: none !important;
        }}
        @media (max-width: 700px) {{
            .navigation-flow-spacer {{ height: 22px; }}
            .block-container {{
                padding-left: 1rem;
            }}
            .st-key-navigation_fixed_shell > div > div[data-testid="stHorizontalBlock"],
            .st-key-app_navigation div[data-testid="stHorizontalBlock"] {{
                flex-direction: row !important;
            }}
            .st-key-app_navigation div[data-testid="stColumn"] {{
                min-height: 50px !important;
            }}
            .st-key-app_navigation .stButton,
            .st-key-app_navigation .stButton > button {{
                height: 50px !important;
                min-height: 50px !important;
                padding: 0 5px !important;
                font-size: .72rem !important;
            }}
            .st-key-chatbot_shell div[data-testid="stChatInput"] {{
                left: 14px !important;
                right: 218px !important;
                width: auto !important;
                bottom: 12px !important;
            }}
            .st-key-chatbot_clear {{
                right: 152px !important;
                bottom: 12px !important;
            }}
            .st-key-chatbot_astra {{
                right: 14px !important;
                bottom: 12px !important;
            }}
            .st-key-chatbot_history {{
                top: 74px !important;
                left: 14px !important;
                right: 14px !important;
                bottom: 82px !important;
            }}
        }}
        .st-key-home_mouse_grid div[class*="st-key-home_mouse_card_"] {{
            min-height: 194px;
            padding: 15px 16px 13px 16px;
            border: 1px solid #D7E1EA;
            border-radius: 13px;
            background: #F5F7F9;
            box-shadow: 0 2px 7px rgba(34,50,72,.035);
        }}
        .st-key-home_mouse_grid div[class*="st-key-home_mouse_card_"]:hover {{
            background: #F1F4F7;
            border-color: #8EB7D0;
            box-shadow: 0 5px 14px rgba(34,50,72,.08);
            transform: translateY(-1px);
        }}
        .home-mouse-name {{
            color: {NAVY};
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 7px;
        }}
        .home-mouse-meta {{
            color: {MUTED};
            font-size: .82rem;
            font-weight: 450;
            line-height: 1.48;
            margin-bottom: 4px;
        }}
        .home-mouse-overview {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            min-height: 76px;
        }}
        .home-mouse-copy {{ min-width: 0; flex: 1; }}
        .home-session-pie {{
            width: 56px;
            height: 56px;
            flex: 0 0 56px;
            border-radius: 50%;
            box-shadow: inset 0 0 0 1px rgba(34,50,72,.08);
            display: grid;
            place-items: center;
        }}
        .home-session-pie-count {{
            min-width: 25px;
            height: 25px;
            padding: 0 4px;
            display: grid;
            place-items: center;
            border-radius: 50%;
            background: rgba(255,255,255,.92);
            color: #34475D;
            font-size: .66rem;
            font-weight: 650;
            line-height: 1;
            box-shadow: 0 1px 4px rgba(34,50,72,.10);
        }}
        .home-pie-composite {{
            display: flex;
            align-items: center;
            gap: 9px;
        }}
        .home-modality-counts {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .home-modality-count {{
            min-width: 34px;
            display: flex;
            align-items: center;
            justify-content: flex-start;
            gap: 4px;
            color: #536477;
            font-size: .72rem;
            line-height: 1;
        }}
        .home-modality-count i {{
            width: 18px;
            height: 18px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            color: white;
            font-size: .72rem;
            font-style: normal;
            font-weight: 700;
        }}
        .home-modality-count i svg {{ width: 13px; height: 13px; }}
        .home-modality-count.ephys i {{ background: #459A76; }}
        .home-modality-count.openfield i {{ background: #80639A; }}
        .home-modality-count b {{
            color: {NAVY};
            font-size: .76rem;
            font-weight: 650;
        }}
        .home-pie-key {{
            display: flex;
            flex-direction: column;
            gap: 5px;
            color: #536477;
            font-size: .72rem;
            font-weight: 600;
            line-height: 1.15;
        }}
        .home-pie-key span {{ white-space: nowrap; }}
        .home-pie-key i {{
            display: inline-block;
            width: 7px;
            height: 7px;
            margin-right: 4px;
            border-radius: 50%;
        }}
        .home-global-pie {{
            min-height: 76px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
        }}
        .home-global-pie .home-session-pie {{
            width: 72px;
            height: 72px;
            flex-basis: 72px;
        }}
        .st-key-home_mouse_grid div[class*="st-key-home_mouse_card_"] button {{
            width: 100%;
            min-height: 34px;
            padding: 0 6px;
            border-radius: 9px;
            font-size: .76rem;
            font-weight: 600;
            display: flex;
            align-items: center !important;
            justify-content: center !important;
        }}
        .st-key-home_mouse_grid div[class*="st-key-home_mouse_card_"] button div[data-testid="stMarkdownContainer"] {{
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .st-key-home_mouse_grid div[class*="st-key-home_mouse_card_"] button p {{
            white-space: nowrap !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1;
            margin: 0 !important;
        }}
        .metric-card {{
            background: {WHITE};
            border: 1px solid {CARD_BORDER};
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 1px 2px rgba(34,50,72,0.04);
        }}
        .metric-label {{
            font-size: 0.84rem;
            color: {MUTED};
            margin-bottom: 0.16rem;
        }}
        .metric-value {{
            font-size: 1.18rem;
            font-weight: 700;
            color: {NAVY};
            line-height: 1.2;
        }}
        .quiet-metric {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 4px 8px;
            min-height: 48px;
            text-align: center;
            box-sizing: border-box;
        }}
        .quiet-metric-label {{
            color: var(--metric-accent, {MUTED});
            font-size: 1.00rem;
            font-weight: 720;
            letter-spacing: 0.01em;
            line-height: 1.1;
        }}
        .quiet-metric-value {{
            color: {NAVY};
            font-size: 1.14rem;
            font-weight: 600;
            line-height: 1.2;
            margin-top: 0;
        }}
        .section-block {{ margin-top: 40px; margin-bottom: 20px; }}
        .section-separator {{
            border: none;
            border-top: 1px solid {CARD_BORDER};
            margin: 0 0 8px 0;
        }}
        .section-title {{
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--section-accent, {NAVY});
            margin: 0;
            line-height: 1.2;
        }}
        .section-card-title {{
            position: absolute;
            top: -63px;
            left: -21px;
            z-index: 2;
            display: inline-flex;
            align-items: center;
            width: fit-content;
            min-height: 42px;
            padding: 9px 18px 9px 20px;
            color: #FFFFFF;
            background: #F04F47;
            border-radius: 15px 0 12px 0;
            box-shadow: 0 5px 14px rgba(224, 76, 59, .13);
            font-size: 1.08rem;
            font-weight: 680;
            line-height: 1.2;
            margin: 0;
        }}
        .st-key-atlas_session_panel,
        div[class*="st-key-ephys_section_"] {{
            background: {PANEL_BG};
            border: 0 !important;
            border-radius: 16px;
            padding: 62px 20px 20px 20px;
            position: relative;
        }}
        .st-key-behavior_summary,
        div[class*="st-key-behavior_section_"] {{
            background: {PANEL_BG};
            border: 0 !important;
            border-radius: 16px;
            padding: 62px 20px 20px 20px;
            position: relative;
        }}
        .st-key-behavior_summary {{
            padding: 18px 20px 20px 20px;
        }}
        div[class*="st-key-behavior_section_"] img {{
            border-radius: 12px;
        }}
        .behavior-subtitle {{
            margin-top: 20px;
            margin-bottom: 12px;
        }}
        .st-key-atlas_session_panel div[data-testid="stPlotlyChart"],
        .st-key-atlas_session_panel .js-plotly-plot,
        .st-key-activity_viewer div[data-testid="stPlotlyChart"],
        .st-key-activity_viewer .js-plotly-plot {{
            border-radius: 14px;
            overflow: hidden;
        }}
        .st-key-activity_viewer .plot-container,
        .st-key-activity_viewer .svg-container {{
            border-radius: 14px;
            overflow: hidden;
        }}
        .st-key-activity_viewer g.heatmaplayer {{
            clip-path: inset(0 round 13px);
        }}
        div[class*="st-key-detail_probe_"][class*="_anatomy"],
        div[class*="st-key-detail_probe_"][class*="_activity"],
        div[class*="st-key-detail_probe_"][class*="_quality_control"] {{
            position: absolute !important;
            top: 11px;
            right: 18px;
            z-index: 5;
        }}
        .metric-plot-gap {{ height: 28px; }}
        .subplot-title {{
            color: #4B5563;
            font-size: 0.88rem;
            font-weight: 680;
            margin: 30px 0 16px 2px;
        }}
        .anatomy-subtitle {{
            margin-top: 12px;
            margin-bottom: 0;
        }}
        .st-key-ephys_section_anatomy g.annotation rect {{
            rx: 9px;
            ry: 9px;
        }}
        .st-key-ephys_section_quality_control .pielayer path.textline {{
            display: none !important;
        }}
        .mini-stat {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            line-height: 1.05;
            min-height: 42px;
            margin: 0;
            border: 0;
            border-radius: 0;
            background: transparent;
        }}
        .mini-stat-value {{
            color: {NAVY};
            font-size: 0.90rem;
            font-weight: 600;
        }}
        .mini-stat-label {{
            color: var(--session-accent, {MUTED});
            font-size: 0.82rem;
            font-weight: 760;
            letter-spacing: 0.01em;
            margin-bottom: 7px;
            order: -1;
        }}
        .session-identity-tile {{
            position: relative;
            z-index: 1;
            min-height: 42px;
            width: 100%;
            padding: 0;
            margin: 0;
            color: var(--session-accent, {NAVY});
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            box-sizing: border-box;
        }}
        .session-identity-mouse {{
            color: var(--session-accent, {NAVY});
            font-size: 0.82rem;
            font-weight: 780;
            letter-spacing: 0.01em;
            line-height: 1.05;
            margin-bottom: 7px;
        }}
        .session-identity-date {{
            color: {NAVY};
            font-size: 0.90rem;
            font-weight: 600;
            line-height: 1.05;
            margin: 0;
            text-align: center;
            white-space: nowrap;
        }}
        .session-card-content {{
            display: grid;
            grid-template-columns: minmax(82px, 1.25fr) minmax(50px, .72fr) minmax(65px, .9fr) minmax(65px, .9fr);
            align-items: center;
            column-gap: 10px;
            width: 100%;
            min-height: 44px;
        }}
        .st-key-session_cards {{
            background: transparent;
            border: 0 !important;
            border-radius: 0;
            padding: 2px 0 0 0;
            box-sizing: border-box;
        }}
        .stDataFrame {{
            border: none;
            border-radius: 0;
            overflow: hidden;
        }}
        .st-key-session_cards div[data-testid="stButton"] button {{
            width: 100%;
            min-height: 70px;
            justify-content: flex-start;
            text-align: left;
            white-space: normal;
            border-radius: 12px;
            padding: 10px 14px;
            line-height: 1.38;
            box-shadow: 0 1px 3px rgba(34,50,72,0.035);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        .st-key-session_cards div[data-testid="stButton"] button:hover {{
            transform: translateY(-1px);
            border-color: #60A5FA;
            box-shadow: 0 7px 18px rgba(34,50,72,0.10);
        }}
        div[data-testid="stSegmentedControl"] button {{
            border-radius: 10px !important;
            background: rgba(255,255,255,0.52) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def source_signature(path: Path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner="Loading behavioral database...", max_entries=1)
def load_behavior_database(path: str, signature):
    """Load the 750 MB Feather once and share the read-only DataFrame.

    cache_data serializes and copies its return value on every call. That is
    useful for small mutable objects, but extremely expensive for this Feather
    because many cells contain NumPy arrays. The dashboard never mutates the
    returned frame, so a shared resource is the appropriate cache here.
    """
    del signature
    dataframe = pd.read_feather(path)
    dataframe, _ = prepare_mouse_dataframe(dataframe)

    if "Number of Valid Bouts" in dataframe.columns:
        dataframe["Valid Bouts"] = dataframe["Number of Valid Bouts"]
    elif "Valid Bouts" not in dataframe.columns:
        dataframe["Valid Bouts"] = dataframe.apply(count_valid_bouts, axis=1)

    return dataframe


@st.cache_data(show_spinner=False, max_entries=2)
def load_behavior_session_counts(path: str, signature):
    del signature
    columns = [
        "Mouse_ID", "Date", "Version", "Correct Bouts", "Bout Start Times",
        "Onsite Acc Start Time", "Traveling Epochs",
    ]
    dataframe = pd.read_feather(path, columns=columns)
    dataframe["Mouse_ID"] = dataframe["Mouse_ID"].astype(str)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dataframe["Version"] = dataframe["Version"].astype(str)
    dataframe["valid_bouts"] = dataframe["Correct Bouts"].apply(
        lambda values: int(np.asarray(values, dtype=bool).sum()) if values is not None else 0
    )
    dataframe["bout_start_times"] = dataframe["Bout Start Times"].apply(
        lambda values: np.asarray(values, dtype=float) if values is not None else np.array([])
    )
    def valid_arrival_times(row):
        correct = np.asarray(row["Correct Bouts"], dtype=bool)
        arrivals = arrivals_from_traveling_epochs(row.get("Traveling Epochs"), len(correct))
        if not np.isfinite(arrivals).any():
            arrivals = np.asarray(row["Onsite Acc Start Time"], dtype=float)
        if len(correct) == len(arrivals):
            arrivals = arrivals[correct]
        return arrivals[np.isfinite(arrivals)]

    dataframe["on_site_arrival_times"] = dataframe.apply(valid_arrival_times, axis=1)
    return (
        dataframe.sort_values(["Mouse_ID", "Date", "Version"])
        .groupby(["Mouse_ID", "Date"], as_index=False)
        .tail(1)[["Mouse_ID", "Date", "valid_bouts", "bout_start_times", "on_site_arrival_times"]]
    )


def flatten_event_times(values):
    flattened = []
    for value in values if values is not None else []:
        if isinstance(value, (list, np.ndarray)):
            flattened.extend(value)
        elif pd.notna(value):
            flattened.append(value)
    return np.asarray(flattened, dtype=float)


@st.cache_data(show_spinner=False, max_entries=32)
def load_behavior_lick_bouts(path: str, signature, mouse: str, date: str):
    del signature
    columns = [
        "Mouse_ID", "Date", "Version", "Correct Bouts", "Time First Lick",
        "Time Last Lick", "Times Rewarded Licks", "Times Non Rewarded Licks",
        "Times Invalid Licks",
    ]
    dataframe = pd.read_feather(path, columns=columns)
    dataframe["Mouse_ID"] = dataframe["Mouse_ID"].astype(str)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    rows = dataframe[(dataframe["Mouse_ID"] == str(mouse)) & (dataframe["Date"] == date)]
    if rows.empty:
        return []
    row = rows.sort_values("Version").iloc[-1]
    correct = np.asarray(row["Correct Bouts"], dtype=bool)
    first = np.asarray(row["Time First Lick"], dtype=float)
    last = np.asarray(row["Time Last Lick"], dtype=float)
    count = min(len(correct), len(first), len(last))
    bouts = []
    rewarded_all = flatten_event_times(row["Times Rewarded Licks"])
    nonrewarded_all = flatten_event_times(row["Times Non Rewarded Licks"])
    invalid_all = flatten_event_times(row["Times Invalid Licks"])
    for first_lick, last_lick, is_correct in zip(first[:count], last[:count], correct[:count]):
        if not is_correct or not (np.isfinite(first_lick) and np.isfinite(last_lick) and last_lick > first_lick):
            continue
        rewarded = rewarded_all[(rewarded_all >= first_lick) & (rewarded_all <= last_lick)]
        nonrewarded = nonrewarded_all[(nonrewarded_all >= first_lick) & (nonrewarded_all <= last_lick)]
        invalid = invalid_all[(invalid_all >= first_lick) & (invalid_all <= last_lick)]
        all_licks = np.sort(np.concatenate([rewarded, nonrewarded, invalid]))
        if 1 < len(all_licks) <= 30:
            bouts.append({
                "first_lick": float(all_licks[0]),
                "last_lick": float(all_licks[-1]),
                "rewarded": rewarded,
                "nonrewarded": nonrewarded,
            })
    return bouts


@st.cache_data(show_spinner=False, max_entries=2)
def load_behavior_video_index(path: str, signature):
    del signature
    columns = ["Mouse_ID", "Date", "Version", "Session Dur", "Correct Bouts"]
    dataframe = pd.read_feather(path, columns=columns)
    dataframe["Mouse_ID"] = dataframe["Mouse_ID"].astype(str)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    dataframe["Version"] = dataframe["Version"].astype(str)
    dataframe["valid_bouts"] = dataframe["Correct Bouts"].apply(
        lambda values: int(np.asarray(values, dtype=bool).sum()) if values is not None else 0
    )
    dataframe["_version_order"] = pd.to_numeric(dataframe["Version"], errors="coerce").fillna(-1)
    return (
        dataframe.dropna(subset=["Date"])
        .sort_values(["Mouse_ID", "Date", "_version_order"])
        .groupby(["Mouse_ID", "Date"], as_index=False)
        .tail(1)[["Mouse_ID", "Date", "Version", "Session Dur", "valid_bouts"]]
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner="Preparing behavioral replay...", max_entries=8)
def load_behavior_video_session(
    path: str,
    signature,
    mouse: str,
    date: str,
    version: str | None,
    database_index: int | None = None,
):
    del signature
    columns = [
        "Mouse_ID", "Date", "Version", "Timestamps", "Speed", "Position",
        "Bout for Timestamps", "Bout Start Times", "Onsite Acc Start Time",
        "Traveling Epochs",
        "Time First Lick", "Time Last Lick", "Position First Lick", "Position Last Lick",
        "Times Rewarded Licks", "Times Non Rewarded Licks", "Times Invalid Licks",
        "Correct Bouts", "Error Bouts", "Missed Bouts", "Manual Reward Bouts",
        "Proba Switch by Bout", "Proba Reward by Bout", "Time to Switch Epochs",
    ]
    dataframe = pd.read_feather(path, columns=columns)
    dataframe["Mouse_ID"] = dataframe["Mouse_ID"].astype(str)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    # The ephys pipeline stores the original Feather row index after scoring all
    # same-day candidates. Prefer that exact row; Version is a compatibility
    # fallback for alignments produced before behavior_database_index existed.
    if database_index is not None:
        try:
            selected = dataframe.loc[int(database_index)]
        except (KeyError, TypeError, ValueError):
            selected = None
        if (
            selected is not None
            and str(selected["Mouse_ID"]) == str(mouse)
            and selected["Date"] == date
            and (version is None or str(selected["Version"]) == str(version))
        ):
            return selected.to_dict()
    if version is None:
        return None
    rows = dataframe[
        (dataframe["Mouse_ID"] == str(mouse))
        & (dataframe["Date"] == date)
        & (dataframe["Version"].astype(str) == str(version))
    ]
    if rows.empty:
        return None
    return rows.iloc[-1].to_dict()


def ephys_behavior_selection(session_dir, probe):
    """Return the behavior row identity selected by the ephys alignment pipeline."""
    alignment_path = casefold_path(
        Path(session_dir), "shift", probe, "alignment_affine.json"
    )
    if not alignment_path.exists():
        return None, None
    try:
        with alignment_path.open(encoding="utf-8") as alignment_file:
            alignment = json.load(alignment_file)
        database_index = alignment.get("behavior_database_index")
        if database_index is not None:
            database_index = int(database_index)
        version = alignment.get("behavior_version")
        return database_index, None if version is None else str(version)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None, None


def replay_numeric_array(value):
    if value is None:
        return np.array([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.array([], dtype=float)


def arrivals_from_traveling_epochs(value, bout_count):
    """Return site-entry times aligned to bouts (travel i ends at the start of bout i+1)."""
    arrivals = np.full(max(0, int(bout_count)), np.nan, dtype=float)
    if value is None or len(arrivals) < 2:
        return arrivals
    ends = []
    for epoch in np.asarray(value, dtype=object).reshape(-1):
        values = replay_numeric_array(epoch)
        ends.append(values[-1] if len(values) >= 2 and np.isfinite(values[-1]) else np.nan)
    count = min(len(ends), len(arrivals) - 1)
    arrivals[1:count + 1] = np.asarray(ends[:count], dtype=float)
    return arrivals


def replay_event_times(value):
    values = []
    if value is None:
        return np.array([], dtype=float)
    for item in np.asarray(value, dtype=object).reshape(-1):
        if isinstance(item, (list, tuple, np.ndarray)):
            values.extend(replay_numeric_array(item).tolist())
        else:
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                continue
    result = np.asarray(values, dtype=float)
    return np.sort(result[np.isfinite(result)])


def infer_replay_sites(samples, position_min, position_max):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2 or np.ptp(samples) < max(8.0, 0.12 * (position_max - position_min)):
        return np.array([np.nanmedian(samples)]) if len(samples) else np.array([])
    centers = np.quantile(samples, [0.25, 0.75]).astype(float)
    for _ in range(12):
        labels = np.argmin(np.abs(samples[:, None] - centers[None, :]), axis=1)
        updated = np.array(
            [np.median(samples[labels == index]) if np.any(labels == index) else centers[index]
             for index in range(2)]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    return np.sort(centers)


def build_behavior_replay_payload(row, mouse, date):
    timestamps = replay_numeric_array(row.get("Timestamps"))
    speed = replay_numeric_array(row.get("Speed"))
    position = replay_numeric_array(row.get("Position"))
    sample_count = min(len(timestamps), len(speed), len(position))
    if sample_count < 2:
        return None

    timestamps = timestamps[:sample_count]
    speed = speed[:sample_count]
    position = position[:sample_count]
    finite = np.isfinite(timestamps) & np.isfinite(speed) & np.isfinite(position)
    timestamps, speed, position = timestamps[finite], speed[finite], position[finite]
    if len(timestamps) < 2:
        return None
    order = np.argsort(timestamps)
    timestamps, speed, position = timestamps[order], speed[order], position[order]
    time_origin = float(timestamps[0])
    time_s = timestamps - time_origin
    duration = float(time_s[-1])

    bout_for_samples = replay_numeric_array(row.get("Bout for Timestamps"))
    if len(bout_for_samples) == sample_count:
        bout_for_samples = bout_for_samples[:sample_count][finite][order]
    else:
        bout_for_samples = np.zeros(len(time_s), dtype=float)

    max_points = 45000
    stride = max(1, int(np.ceil(len(time_s) / max_points)))
    keep = np.arange(0, len(time_s), stride, dtype=int)
    if keep[-1] != len(time_s) - 1:
        keep = np.append(keep, len(time_s) - 1)

    def relative_times(value):
        values = replay_event_times(value) - time_origin
        return values[(values >= 0) & (values <= duration)]

    bout_starts_raw = replay_numeric_array(row.get("Bout Start Times"))
    arrivals_raw = arrivals_from_traveling_epochs(
        row.get("Traveling Epochs"),
        max(len(bout_starts_raw), len(replay_numeric_array(row.get("Time First Lick")))),
    )
    if not np.isfinite(arrivals_raw).any():
        arrivals_raw = replay_numeric_array(row.get("Onsite Acc Start Time"))
    first_lick_raw = replay_numeric_array(row.get("Time First Lick"))
    last_lick_raw = replay_numeric_array(row.get("Time Last Lick"))
    first_position = replay_numeric_array(row.get("Position First Lick"))
    last_position = replay_numeric_array(row.get("Position Last Lick"))
    correct = np.asarray(row.get("Correct Bouts", []), dtype=bool).reshape(-1)

    position_min = float(np.floor(np.nanmin(position) / 10) * 10)
    position_max = float(np.ceil(np.nanmax(position) / 10) * 10)
    if position_max <= position_min:
        position_max = position_min + 1
    track_period = position_max - position_min
    position_delta = np.diff(position)
    continuous_delta = (position_delta + track_period / 2) % track_period - track_period / 2
    world_position = np.r_[position[0], position[0] + np.cumsum(continuous_delta)]

    site_samples = np.concatenate(
        [first_position[np.isfinite(first_position)], last_position[np.isfinite(last_position)]]
    )
    sites = infer_replay_sites(site_samples, position_min, position_max)
    if len(sites) == 0:
        sites = np.quantile(position, [0.3, 0.7])
    site_bounds = []
    finite_site_samples = site_samples[np.isfinite(site_samples)]
    for site_index, site in enumerate(sites):
        if len(finite_site_samples):
            labels = np.argmin(np.abs(finite_site_samples[:, None] - sites[None, :]), axis=1)
            local = finite_site_samples[labels == site_index]
        else:
            local = np.array([], dtype=float)
        if len(local) >= 4:
            lower, upper = np.quantile(local, [0.08, 0.92])
            half_width = float(np.clip((upper - lower) / 2, 4.0, 15.0))
            center = float((lower + upper) / 2)
        else:
            center, half_width = float(site), 5.0
        site_bounds.append([round(center - half_width, 3), round(center + half_width, 3)])

    bout_count = max(
        len(bout_starts_raw), len(arrivals_raw), len(first_lick_raw), len(first_position), len(correct)
    )
    bout_sites = np.full(bout_count, np.nan)
    for index in range(bout_count):
        candidate = first_position[index] if index < len(first_position) else np.nan
        if not np.isfinite(candidate) and index < len(arrivals_raw) and np.isfinite(arrivals_raw[index]):
            candidate = np.interp(arrivals_raw[index], timestamps, position)
        if np.isfinite(candidate):
            bout_sites[index] = int(np.argmin(np.abs(sites - candidate)))

    switch_times_raw = np.full(bout_count, np.nan)
    rewarded_by_bout = row.get("Times Rewarded Licks")
    if rewarded_by_bout is not None:
        for index, bout_licks in enumerate(
            np.asarray(rewarded_by_bout, dtype=object).reshape(-1)[:bout_count]
        ):
            values = replay_numeric_array(bout_licks)
            values = values[np.isfinite(values)]
            if len(values):
                # The last rewarded lick switches the site off. Subsequent
                # non-rewarded/invalid licks remain visible in the replay.
                switch_times_raw[index] = values[-1]

    valid_site_indices = np.flatnonzero(np.isfinite(bout_sites))
    switches = []
    if len(valid_site_indices):
        active_schedule = [{"t": 0.0, "site": int(bout_sites[valid_site_indices[0]])}]
        next_valid_by_bout = {
            int(left): int(right) for left, right in zip(valid_site_indices[:-1], valid_site_indices[1:])
        }
        for left in valid_site_indices:
            right = next_valid_by_bout.get(int(left))
            from_site = int(bout_sites[left])
            to_site = int(bout_sites[right]) if right is not None else from_site
            switch_time = switch_times_raw[left] if left < len(switch_times_raw) else np.nan
            switch_time = float(switch_time - time_origin) if np.isfinite(switch_time) else np.nan
            if 0 <= switch_time <= duration:
                switches.append({"t": round(switch_time, 3), "from": from_site, "to": to_site})
                if to_site != from_site:
                    active_schedule.append({"t": round(switch_time, 3), "site": to_site})
    else:
        active_schedule = [{"t": 0.0, "site": 0}]

    arrivals = []
    for index, arrival in enumerate(arrivals_raw):
        event_time = float(arrival - time_origin) if np.isfinite(arrival) else np.nan
        if not (0 <= event_time <= duration):
            continue
        site = int(bout_sites[index]) if index < len(bout_sites) and np.isfinite(bout_sites[index]) else -1
        arrivals.append(
            {
                "t": round(event_time, 3),
                "site": site,
                "valid": bool(correct[index]) if index < len(correct) else False,
                "bout": index + 1,
            }
        )

    rewarded = relative_times(row.get("Times Rewarded Licks"))
    nonrewarded = relative_times(row.get("Times Non Rewarded Licks"))
    invalid = relative_times(row.get("Times Invalid Licks"))
    bout_starts = bout_starts_raw - time_origin
    bout_starts = bout_starts[np.isfinite(bout_starts) & (bout_starts >= 0) & (bout_starts <= duration)]

    dt = np.diff(time_s)
    integrated_distance = float(np.sum(np.maximum(speed[:-1], 0) * np.maximum(dt, 0)))
    payload = {
        "mouse": str(mouse),
        "date": str(date),
        "version": str(row.get("Version", "")),
        "duration": round(duration, 3),
        "timeOrigin": round(time_origin, 6),
        "positionMin": round(position_min, 3),
        "positionMax": round(position_max, 3),
        "sitePositions": [round(float(value), 3) for value in sites],
        "siteBounds": site_bounds,
        "time": np.round(time_s[keep], 3).tolist(),
        "speed": np.round(np.maximum(speed[keep], 0), 3).tolist(),
        "position": np.round(position[keep], 3).tolist(),
        "worldPosition": np.round(world_position[keep], 3).tolist(),
        "bout": np.nan_to_num(bout_for_samples[keep], nan=0).astype(int).tolist(),
        "boutStarts": np.round(bout_starts, 3).tolist(),
        "arrivals": arrivals,
        "rewardedLicks": np.round(rewarded, 3).tolist(),
        "nonrewardedLicks": np.round(nonrewarded, 3).tolist(),
        "invalidLicks": np.round(invalid, 3).tolist(),
        "switches": switches,
        "activeSchedule": active_schedule,
        "summary": {
            "validBouts": int(correct.sum()),
            "totalBouts": int(bout_count),
            "totalLicks": int(len(rewarded) + len(nonrewarded) + len(invalid)),
            "medianSpeed": round(float(np.nanmedian(speed)), 2),
            "peakSpeed": round(float(np.nanpercentile(speed, 99)), 2),
            "distanceM": round(integrated_distance / 100, 1),
        },
    }
    return payload


def _attach_ephys_to_replay_legacy(payload, regional_activity, alignment):
    payload = dict(payload)
    payload["alignment"] = {
        "available": bool(alignment),
        "qcPass": bool(alignment.get("qc_pass", False)) if alignment else False,
        "residualMs": round(
            1000 * float(alignment.get("residual_median_abs_s", np.nan)), 3
        ) if alignment and np.isfinite(float(alignment.get("residual_median_abs_s", np.nan))) else None,
    }
    payload["regionActivity"] = None
    if not regional_activity:
        return payload

    absolute_time = np.asarray(regional_activity.get("time_s", []), dtype=float)
    relative_time = absolute_time - float(payload.get("timeOrigin", 0.0))
    keep = np.isfinite(relative_time) & (relative_time >= 0) & (relative_time <= payload["duration"])
    if not np.any(keep):
        return payload
    payload["regionActivity"] = {
        "time": np.round(relative_time[keep], 3).tolist(),
        "regions": {
            region: np.round(np.asarray(values, dtype=float)[keep], 3).tolist()
            for region, values in regional_activity.get("regions", {}).items()
        },
        "unitCounts": regional_activity.get("unit_counts", {}),
        "probe": regional_activity.get("probe", ""),
    }
    return payload


def attach_ephys_to_replay(payload, probe_activity_items):
    payload = dict(payload)
    payload["probeActivities"] = []
    for item in probe_activity_items:
        regional_activity = item.get("regional_activity")
        alignment = item.get("alignment") or {}
        probe_name = item.get("probe", "probe")
        probe_payload = {
            "probe": probe_name,
            "alignment": {
                "available": bool(alignment),
                "qcPass": bool(alignment.get("qc_pass", False)),
                "residualMs": None,
            },
            "time": [],
            "regions": {},
            "regionColors": {},
            "firingRates": {},
            "unitCounts": {},
        }
        residual = float(alignment.get("residual_median_abs_s", np.nan))
        if np.isfinite(residual):
            probe_payload["alignment"]["residualMs"] = round(1000 * residual, 3)
        if regional_activity:
            absolute_time = np.asarray(regional_activity.get("time_s", []), dtype=float)
            relative_time = absolute_time - float(payload.get("timeOrigin", 0.0))
            keep = np.isfinite(relative_time) & (relative_time >= 0) & (relative_time <= payload["duration"])
            if np.any(keep):
                probe_payload["time"] = np.round(relative_time[keep], 3).tolist()
                probe_payload["regions"] = {
                    region: np.round(np.asarray(values, dtype=float)[keep], 3).tolist()
                    for region, values in regional_activity.get("regions", {}).items()
                }
                probe_payload["regionColors"] = {
                    region: brain_region_color(region)
                    for region in probe_payload["regions"]
                }
                probe_payload["firingRates"] = {
                    region: np.round(np.asarray(values, dtype=float)[keep], 3).tolist()
                    for region, values in regional_activity.get("firing_rates", {}).items()
                }
                probe_payload["unitCounts"] = regional_activity.get("unit_counts", {})
        payload["probeActivities"].append(probe_payload)
    return payload


def build_behavior_replay_html(payload, behavior_only=False):
    replay_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    safe_mouse = html.escape(payload["mouse"])
    safe_date = html.escape(payload["date"])
    template = r"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width,initial-scale=1">
      <style>
        :root {
          --ink:#223248; --muted:#748091; --line:#D7E0E8; --panel:#E9EEF2;
          --paper:#F7FAFC; --cyan:#48B8D0; --violet:#8B75C9; --green:#54A986;
          --red:#E56D78; --orange:#E7A85C; --navy:#111B2A;
        }
        * { box-sizing:border-box; }
        html,body { margin:0; background:transparent; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
        .replay-shell { background:transparent; border:0; padding:0; overflow:hidden; }
        .replay-head { display:flex; align-items:flex-end; justify-content:space-between; gap:20px; margin:0 2px 16px; }
        .eyebrow { color:var(--violet); font-size:11px; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
        h1 { font-size:23px; line-height:1.1; margin:5px 0 0; color:#172033; letter-spacing:-.02em; }
        h1 span { color:var(--muted); font-weight:570; margin-left:8px; }
        .summary { display:flex; gap:20px; align-items:center; justify-content:flex-end; flex-wrap:wrap; }
        .summary-item { text-align:right; min-width:68px; }
        .summary-label { color:var(--muted); font-size:10px; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
        .summary-value { color:var(--ink); font-size:15px; font-weight:750; margin-top:3px; }
        .top-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:24px; width:100%; align-items:start; }
        .top-item { min-width:0; padding:8px 4px 0; }
        .top-item > .trace-title { margin:0 0 22px; justify-content:flex-start; text-align:left; }
        .stage-card,.metrics-card { background:rgba(247,250,252,.72); border:1px solid var(--line); border-radius:15px; }
        .stage-card { position:relative; min-width:0; width:calc(100% - 116px); margin-left:72px; height:270px; overflow:hidden; background:var(--navy); border-color:#263548; }
        #scene { width:100%; height:270px; display:block; }
        .spike-probes { display:grid; gap:30px; min-width:0; width:calc(100% - 26px); margin-left:26px; height:270px; align-items:start; }
        .spike-probe { height:270px; min-width:0; position:relative; overflow:visible; background:transparent; border:0; }
        .spike-probe canvas { width:100%; height:270px; display:block; }
        .spike-probe-label { position:absolute; top:auto; bottom:0; left:0; right:0; color:#344256; text-align:center; font-size:11px; line-height:14px; font-weight:760; letter-spacing:.02em; pointer-events:none; }
        .brain-atlas { height:270px; min-width:0; position:relative; display:grid; grid-template-columns:minmax(0,1fr) 24px; column-gap:6px; align-items:center; overflow:visible; background:transparent; border:0; }
        .brain-atlas-plot { grid-column:1; width:100%; height:270px; display:block; }
        .atlas-plane-slider { grid-column:2; position:static; width:18px; height:238px; writing-mode:vertical-lr; direction:rtl; transform:none; accent-color:#7289B0; margin:0; justify-self:end; align-self:center; }
        .stage-caption { position:absolute; top:16px; left:18px; display:flex; gap:9px; align-items:center; pointer-events:none; }
        .live-dot { width:7px; height:7px; border-radius:50%; background:var(--red); box-shadow:0 0 0 5px rgba(229,109,120,.14); }
        .stage-caption span:last-child { color:#D8E2ED; font-size:11px; font-weight:750; letter-spacing:.11em; text-transform:uppercase; }
        .event-toast { position:absolute; left:50%; top:18px; transform:translate(-50%,-8px); color:white; font-size:12px; font-weight:750; padding:7px 12px; border-radius:999px; opacity:0; transition:opacity .16s ease,transform .16s ease; pointer-events:none; white-space:nowrap; }
        .event-toast.show { opacity:1; transform:translate(-50%,0); }
        .metrics-card { padding:11px; display:grid; grid-template-columns:1fr 1fr; gap:7px; align-content:start; }
        .metric { background:rgba(255,255,255,.50); border:1px solid #DCE4EA; border-radius:11px; min-height:59px; padding:8px 10px; display:flex; flex-direction:column; justify-content:center; }
        .metric.wide { grid-column:1/-1; min-height:47px; }
        .metric-label { font-size:10px; font-weight:780; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
        .metric-value { margin-top:5px; font-size:19px; line-height:1; font-weight:790; color:var(--ink); }
        .metric-value small { font-size:11px; font-weight:650; color:var(--muted); margin-left:2px; }
        .metric-sub { margin-top:6px; color:var(--muted); font-size:10px; }
        .state-pill { align-self:flex-start; margin-top:6px; padding:5px 9px; border-radius:999px; color:white; font-size:11px; font-weight:760; background:var(--cyan); }
        .timeline-stack { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); column-gap:24px; margin-top:38px; align-items:start; }
        .timeline-column { display:grid; grid-template-columns:minmax(0,1fr); row-gap:34px; min-width:0; }
        .timeline-card { padding:8px 4px 7px; overflow:hidden; background:transparent; border:0; border-radius:0; }
        .trace-title { display:flex; align-items:center; justify-content:space-between; gap:12px; margin:0 0 22px; }
        .trace-title strong { color:#4B5563; font-size:13px; }
        .trace-title span { color:var(--muted); font-size:11px; }
        .timeline-canvas { width:100%; display:block; }
        #speedTimeline { height:125px; }
        #eventTimeline { height:135px; }
        .neural-timeline { height:145px; }
        #neuralPlots { display:grid; grid-template-columns:minmax(0,1fr); row-gap:34px; min-width:0; }
        .probe-neural { min-width:0; }
        .legend { display:flex; align-items:center; justify-content:center; gap:8px; flex-wrap:wrap; color:#667384; font-size:9px; margin:0; }
        .legend i { width:6px; height:6px; border-radius:50%; display:inline-block; margin-right:3px; vertical-align:0; }
        .controls { margin-top:24px; padding:4px 0 0; display:grid; grid-template-columns:auto minmax(180px,1fr) auto auto; gap:14px; align-items:center; background:transparent; border:0; border-radius:0; }
        .behavior-side-item { display:none; }
        .behavior-only .top-grid { grid-template-columns:repeat(2,minmax(0,1fr)); gap:28px; }
        .behavior-only .histology-item { display:none; }
        .behavior-only .behavior-side-item { display:block; }
        .behavior-only > .timeline-stack { display:none; }
        .behavior-only .stage-card { width:100%; margin-left:0; }
        .behavior-only .behavior-side-slot .timeline-column { row-gap:24px; }
        .behavior-only .behavior-side-slot .timeline-card { padding:8px 4px 2px; }
        .behavior-only .top-item > .trace-title,
        .behavior-only .behavior-side-slot .trace-title { margin-bottom:28px; }
        .behavior-only .behavior-side-slot #eventTimeline { height:112px; }
        .behavior-only .behavior-side-slot #speedTimeline { height:112px; }
        .behavior-only .controls { margin-top:22px; }
        button,select { font:inherit; }
        .round-button { width:38px; height:38px; border:0; border-radius:11px; color:white; background:var(--ink); cursor:pointer; font-size:15px; transition:transform .12s ease,background .12s ease; }
        .round-button:hover { transform:translateY(-1px); background:#314762; }
        .time { min-width:94px; text-align:right; color:var(--ink); font-size:12px; font-variant-numeric:tabular-nums; font-weight:680; }
        select { height:38px; border:1px solid var(--line); color:var(--ink); background:rgba(255,255,255,.64); border-radius:10px; padding:0 10px; font-weight:700; outline:none; }
        input[type=range] { width:100%; accent-color:var(--violet); cursor:pointer; }
        @media (max-width:850px) {
          .top-grid { grid-template-columns:1fr; }
          .stage-card,.spike-probes { width:100%; margin-left:0; }
          .timeline-stack,#neuralPlots { grid-template-columns:1fr; }
          .metrics-card { grid-template-columns:repeat(3,1fr); }
          .metric.wide { grid-column:auto; }
          .replay-head { align-items:flex-start; flex-direction:column; }
          .summary { justify-content:flex-start; }
        }
      </style>
    </head>
    <body>
      <div class="replay-shell __MODE_CLASS__">
        <div class="top-grid">
          <div class="top-item video-item">
            <div class="trace-title"><strong>Behavior reconstruction</strong></div>
            <div class="stage-card">
              <canvas id="scene"></canvas>
              <div class="event-toast" id="eventToast"></div>
            </div>
          </div>
          <div class="top-item histology-item">
            <div class="trace-title"><strong>Histological view</strong></div>
            <div class="spike-probes" id="spikeProbes"></div>
          </div>
          <div class="top-item behavior-side-item">
            <div class="behavior-side-slot" id="behaviorSideSlot"></div>
          </div>
        </div>

        <div class="timeline-stack">
          <div class="timeline-column">
            <div class="timeline-card">
              <div class="trace-title"><strong>Behavioral events</strong></div>
              <canvas class="timeline-canvas" id="eventTimeline"></canvas>
            </div>
            <div class="timeline-card">
              <div class="trace-title"><strong>Speed</strong></div>
              <canvas class="timeline-canvas" id="speedTimeline"></canvas>
            </div>
          </div>
          <div id="neuralPlots"></div>
        </div>

        <div class="controls">
          <button class="round-button" id="playButton" aria-label="Play">▶</button>
          <input id="scrubber" type="range" min="0" step="0.01" value="0">
          <div class="time" id="clock">00:00 / 00:00</div>
          <select id="playbackRate" aria-label="Playback speed">
            <option value="1" selected>1×</option><option value="2">2×</option><option value="5">5×</option>
            <option value="10">10×</option><option value="20">20×</option><option value="50">50×</option>
          </select>
        </div>
      </div>

      <script>
        const D = __DATA__;
        const replayShell = document.querySelector('.replay-shell');
        const behaviorSideSlot = document.getElementById('behaviorSideSlot');
        if (replayShell.classList.contains('behavior-only') && behaviorSideSlot) {
          const behaviorColumn = document.querySelector('.timeline-stack .timeline-column');
          if (behaviorColumn) behaviorSideSlot.appendChild(behaviorColumn);
        }
        const scene = document.getElementById('scene');
        const speedTimeline = document.getElementById('speedTimeline');
        const eventTimeline = document.getElementById('eventTimeline');
        const neuralPlots = document.getElementById('neuralPlots');
        const spikeProbes = document.getElementById('spikeProbes');
        const scrubber = document.getElementById('scrubber');
        const playButton = document.getElementById('playButton');
        const playbackRate = document.getElementById('playbackRate');
        const toast = document.getElementById('eventToast');
        let currentTime = 0, previousTime = 0, playing = false, frameStamp = null;
        let toastUntil = 0, lickUntil = 0, lickColor = '#E56D78', direction = 1;
        const TIMELINE_LEFT=112, TIMELINE_RIGHT=14;
        scrubber.max = D.duration;

        const COLORS = { arrival:'#48B8D0', rewarded:'#54A986', nonrewarded:'#E7A85C', invalid:'#E56D78', switched:'#8B75C9' };
        const byId = id => document.getElementById(id);
        const REGION_COLORS=['#8B75C9','#48B8D0','#D17A5C','#54A986','#C08A45','#C45E86'];
        const neuralViews=[], spikeViews=[];
        const probeActivities=D.probeActivities||[];
        spikeProbes.style.gridTemplateColumns=`${probeActivities.map(()=>"44px").join(" ")} minmax(220px,1fr)`;
        probeActivities.forEach((activity,probeIndex)=>{
          const view=document.createElement('div');view.className='timeline-card probe-neural';
          const head=document.createElement('div');head.className='trace-title';
          const title=document.createElement('strong');title.textContent=activity.probe+' activity';
          head.appendChild(title);view.appendChild(head);
          const canvas=document.createElement('canvas');canvas.className='timeline-canvas neural-timeline';view.appendChild(canvas);
          const legend=document.createElement('div');legend.className='legend';
          Object.keys(activity.regions).forEach((region,index)=>{
            const item=document.createElement('span'),dot=document.createElement('i');dot.style.background=(activity.regionColors||{})[region]||REGION_COLORS[index%REGION_COLORS.length];
            item.appendChild(dot);item.appendChild(document.createTextNode(region));legend.appendChild(item);
          });
          view.appendChild(legend);neuralPlots.appendChild(view);neuralViews.push({canvas,activity});
          const spikeCard=document.createElement('div');spikeCard.className='spike-probe';
          const spikeCanvas=document.createElement('canvas');spikeCard.appendChild(spikeCanvas);
          const spikeLabel=document.createElement('div');spikeLabel.className='spike-probe-label';spikeLabel.textContent=activity.probe;spikeCard.appendChild(spikeLabel);
          spikeProbes.appendChild(spikeCard);spikeViews.push({canvas:spikeCanvas,activity,probeIndex});
        });
        const atlasCard=document.createElement('div');atlasCard.className='brain-atlas';
        const atlasCanvas=document.createElement('canvas');atlasCanvas.className='brain-atlas-plot';atlasCard.appendChild(atlasCanvas);
        const atlasSlider=document.createElement('input');atlasSlider.type='range';atlasSlider.className='atlas-plane-slider';atlasSlider.min=0;atlasSlider.max=Math.max(0,((D.atlasSlices||{}).slices||[]).length-1);atlasSlider.step=1;atlasSlider.value=Math.max(0,Math.min(Number(atlasSlider.max),Number((D.atlasSlices||{}).defaultSlice)||0));atlasCard.appendChild(atlasSlider);
        spikeProbes.appendChild(atlasCard);

        function canvasContext(canvas, height) {
          const width = Math.max(10, canvas.getBoundingClientRect().width);
          const ratio = Math.min(window.devicePixelRatio || 1, 2);
          const wantedW = Math.round(width * ratio), wantedH = Math.round(height * ratio);
          if (canvas.width !== wantedW || canvas.height !== wantedH) { canvas.width = wantedW; canvas.height = wantedH; }
          const ctx = canvas.getContext('2d');
          ctx.setTransform(ratio,0,0,ratio,0,0);
          return {ctx,width,height};
        }
        function rounded(ctx,x,y,w,h,r) {
          const radius = Math.min(r,w/2,h/2);
          ctx.beginPath(); ctx.roundRect(x,y,w,h,radius); return ctx;
        }
        function lowerBound(values, target) {
          let lo=0, hi=values.length;
          while (lo<hi) { const mid=(lo+hi)>>1; if (values[mid]<target) lo=mid+1; else hi=mid; }
          return lo;
        }
        function wrappedPosition(value) {
          const period=D.positionMax-D.positionMin;
          return ((value-D.positionMin)%period+period)%period+D.positionMin;
        }
        function sampleAt(t) {
          let right = Math.min(Math.max(lowerBound(D.time,t),1),D.time.length-1), left=right-1;
          const span = Math.max(D.time[right]-D.time[left],1e-6), alpha=(t-D.time[left])/span;
          const lerp = key => D[key][left] + (D[key][right]-D[key][left])*alpha;
          const worldPosition=lerp('worldPosition'), delta=D.worldPosition[right]-D.worldPosition[left];
          if (Math.abs(delta)>.02) direction = Math.sign(delta);
          return {speed:lerp('speed'), position:wrappedPosition(worldPosition), worldPosition, bout:alpha<.5?D.bout[left]:D.bout[right], index:right};
        }
        function activeSiteAt(t) {
          let site = D.activeSchedule.length ? D.activeSchedule[0].site : 0;
          for (const item of D.activeSchedule) { if (item.t<=t) site=item.site; else break; }
          return Math.max(0,Math.min(site,D.sitePositions.length-1));
        }
        function formatTime(seconds) {
          seconds=Math.max(0,Math.round(seconds)); const m=Math.floor(seconds/60), s=seconds%60;
          return String(m).padStart(2,'0')+':'+String(s).padStart(2,'0');
        }
        function rollingSpeed(t, seconds=2) {
          let end=Math.min(lowerBound(D.time,t),D.time.length-1), start=Math.max(0,lowerBound(D.time,t-seconds));
          let total=0,count=0; for(let i=start;i<=end;i++){total+=D.speed[i];count++;}
          return count?total/count:0;
        }
        function latestCrossed(array, from, to, objectTimes=false) {
          let found=null;
          for (const item of array) { const value=objectTimes?item.t:item; if(value>from && value<=to) found=item; if(value>to) break; }
          return found;
        }
        function flashEvents(from,to) {
          if (to<from) return;
          const candidates=[];
          const sw=latestCrossed(D.switches,from,to,true); if(sw)candidates.push({t:sw.t,label:'Site deactivation · Site '+String.fromCharCode(65+sw.from)+' → '+String.fromCharCode(65+sw.to),type:'switched'});
          const ar=latestCrossed(D.arrivals,from,to,true); if(ar)candidates.push({t:ar.t,label:'Arrival (site entry) '+(ar.site>=0?String.fromCharCode(65+ar.site):''),type:'arrival'});
          const rw=latestCrossed(D.rewardedLicks,from,to); if(rw!==null)candidates.push({t:rw,label:'Rewarded lick',type:'rewarded'});
          const nr=latestCrossed(D.nonrewardedLicks,from,to); if(nr!==null)candidates.push({t:nr,label:'Non-rewarded lick',type:'nonrewarded'});
          const iv=latestCrossed(D.invalidLicks,from,to); if(iv!==null)candidates.push({t:iv,label:'Invalid lick',type:'invalid'});
          if (!candidates.length) return;
          const event=candidates.sort((a,b)=>a.t-b.t).at(-1);
          toast.textContent=event.label; toast.style.background=COLORS[event.type]; toast.classList.add('show');
          toastUntil=performance.now()+650;
          if (['rewarded','nonrewarded','invalid'].includes(event.type)) {
            lickUntil=performance.now()+90; lickColor=COLORS[event.type];
          }
        }
        function eventNear(array,t,window=.22,objectTimes=false) {
          const values=objectTimes?array.map(item=>item.t):array;if(!values.length)return false;
          const right=lowerBound(values,t),left=Math.max(0,right-1);
          const leftDistance=Math.abs(t-values[left]),rightDistance=right<values.length?Math.abs(t-values[right]):Infinity;
          return Math.min(leftDistance,rightDistance)<=window;
        }
        function drawMouseLegacy(ctx,x,y,speed,t,licking) {
          const moving=speed>1, phase=t*Math.max(speed,4)*.22, bob=moving?Math.abs(Math.sin(phase))*3:0;
          const stretch=Math.min(speed/55,.18), scale=1.24;
          ctx.save();ctx.translate(x,y-bob);ctx.scale(direction*scale,scale);
          if(speed>5){ctx.strokeStyle='rgba(112,199,216,.12)';ctx.lineWidth=2;for(let i=0;i<3;i++){ctx.beginPath();ctx.moveTo(-42-i*9,1+i*7);ctx.lineTo(-65-i*13,1+i*7);ctx.stroke();}}
          ctx.strokeStyle='#AFC0CF';ctx.lineWidth=3;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(-21,5);ctx.bezierCurveTo(-38,-2,-43,17,-58,11+Math.sin(phase)*4);ctx.stroke();
          const legA=Math.sin(phase)*9,legB=Math.sin(phase+Math.PI)*9;
          ctx.strokeStyle='#C7D4DE';ctx.lineWidth=3.4;ctx.beginPath();ctx.moveTo(-10,9);ctx.lineTo(-11+legA,20);ctx.lineTo(-4+legA,22);ctx.moveTo(-3,10);ctx.lineTo(-4+legB,20);ctx.lineTo(3+legB,22);ctx.moveTo(9,9);ctx.lineTo(8+legB,19);ctx.lineTo(15+legB,21);ctx.moveTo(15,7);ctx.lineTo(15+legA,17);ctx.lineTo(21+legA,19);ctx.stroke();
          ctx.shadowColor='rgba(72,184,208,.38)';ctx.shadowBlur=20;ctx.fillStyle='#DDE7EE';ctx.beginPath();ctx.ellipse(-1,0,26+stretch*12,14-stretch*2,-.04,0,Math.PI*2);ctx.fill();
          ctx.shadowBlur=8;ctx.fillStyle='#F4F7F9';ctx.beginPath();ctx.ellipse(22,-4,14,10,-.08,0,Math.PI*2);ctx.fill();
          ctx.fillStyle='#D7A2AF';ctx.beginPath();ctx.arc(17,-13,6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#F2CDD4';ctx.lineWidth=1.3;ctx.stroke();
          ctx.fillStyle='#192638';ctx.beginPath();ctx.arc(27,-7,2.1,0,Math.PI*2);ctx.fill();ctx.fillStyle='#D88698';ctx.beginPath();ctx.arc(35,-2,2.5,0,Math.PI*2);ctx.fill();
          ctx.strokeStyle='#CFDAE3';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(30,-1);ctx.lineTo(43,-6);ctx.moveTo(31,1);ctx.lineTo(45,1);ctx.stroke();
          if(licking){const tongue=7+4*Math.abs(Math.sin(performance.now()/75));ctx.strokeStyle=lickColor;ctx.lineWidth=3.2;ctx.beginPath();ctx.moveTo(34,1);ctx.quadraticCurveTo(39+tongue/2,4,40+tongue,1);ctx.stroke();ctx.fillStyle=lickColor;ctx.beginPath();ctx.arc(40+tongue,1,2.1,0,Math.PI*2);ctx.fill();}
          ctx.restore();
        }
        function drawMouse(ctx,x,y,speed,t,licking,lickIndicatorColor) {
          const moving=speed>1, cadence=2.0+Math.min(speed,55)*.075;
          const phase=t*cadence*Math.PI*2, stride=moving?Math.min(11,3+speed*.18):1.2;
          const bob=moving?(1-Math.cos(phase*2))*0.8:0, stretch=Math.min(speed/70,.14), scale=1.24;
          ctx.save();ctx.translate(x,y-bob);ctx.scale(direction*scale,scale);ctx.lineCap='round';ctx.lineJoin='round';
          if(speed>7){ctx.strokeStyle='rgba(112,199,216,.11)';ctx.lineWidth=2;for(let i=0;i<3;i++){ctx.beginPath();ctx.moveTo(-42-i*9,2+i*7);ctx.lineTo(-63-i*12,2+i*7);ctx.stroke();}}
          ctx.strokeStyle='#AFC0CF';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(-21,5);ctx.bezierCurveTo(-38,-3,-44,15,-59,10+Math.sin(phase*.5)*3);ctx.stroke();
          function leg(anchorX,offset,far){
            const p=phase+offset,swing=Math.sin(p)*stride,lift=moving?Math.max(0,-Math.cos(p))*5:0;
            ctx.strokeStyle=far?'rgba(174,191,204,.68)':'#C7D4DE';ctx.lineWidth=far?2.7:3.3;
            ctx.beginPath();ctx.moveTo(anchorX,7);ctx.quadraticCurveTo(anchorX+swing*.30,14-lift*.30,anchorX+swing*.72,18-lift*.65);ctx.lineTo(anchorX+swing,22-lift);ctx.lineTo(anchorX+swing+5,22-lift);ctx.stroke();
          }
          leg(-11,Math.PI,true);leg(10,0,true);leg(-5,0,false);leg(16,Math.PI,false);
          ctx.shadowColor='rgba(72,184,208,.38)';ctx.shadowBlur=20;ctx.fillStyle='#DDE7EE';ctx.beginPath();ctx.ellipse(-1,0,26+stretch*12,14-stretch*2,-.04,0,Math.PI*2);ctx.fill();
          ctx.shadowBlur=8;ctx.fillStyle='#F4F7F9';ctx.beginPath();ctx.ellipse(22,-4,14,10,-.08,0,Math.PI*2);ctx.fill();
          ctx.fillStyle='#D7A2AF';ctx.beginPath();ctx.arc(17,-13,6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#F2CDD4';ctx.lineWidth=1.3;ctx.stroke();
          ctx.fillStyle='#192638';ctx.beginPath();ctx.arc(27,-7,2.1,0,Math.PI*2);ctx.fill();ctx.fillStyle='#D88698';ctx.beginPath();ctx.arc(35,-2,2.5,0,Math.PI*2);ctx.fill();
          ctx.strokeStyle='#CFDAE3';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(30,-1);ctx.lineTo(43,-6);ctx.moveTo(31,1);ctx.lineTo(45,1);ctx.stroke();
          ctx.restore();
        }
        function drawScene(t,sample) {
          const {ctx,width:w,height:h}=canvasContext(scene,270);
          const bg=ctx.createLinearGradient(0,0,w,h);bg.addColorStop(0,'#0C1624');bg.addColorStop(1,'#172538');ctx.fillStyle=bg;ctx.fillRect(0,0,w,h);
          const margin=28,trackY=h*.50,trackH=104,trackW=w-2*margin,mouseX=w*.47,worldScale=Math.min(7.2,w/125),headX=mouseX+direction*35;
          ctx.fillStyle='rgba(255,255,255,.025)';rounded(ctx,margin-8,trackY-trackH/2-13,trackW+16,trackH+26,42).fill();ctx.fillStyle='#26384B';rounded(ctx,margin,trackY-trackH/2,trackW,trackH,33).fill();
          const period=D.positionMax-D.positionMin,worldX=p=>headX+(p-sample.worldPosition)*worldScale;
          ctx.strokeStyle='rgba(216,226,237,.13)';ctx.lineWidth=2;const markStart=Math.floor((sample.worldPosition-trackW/worldScale/2)/10)*10,markStop=sample.worldPosition+trackW/worldScale/2;for(let mark=markStart;mark<=markStop;mark+=10){const x=worldX(mark);if(x<margin||x>margin+trackW)continue;ctx.beginPath();ctx.moveTo(x,trackY+30);ctx.lineTo(x+direction*8,trackY+30);ctx.stroke();ctx.fillStyle='#718296';ctx.font='600 9px Inter, sans-serif';ctx.textAlign='center';ctx.fillText(Math.round(wrappedPosition(mark)),x,trackY+48);}
          ctx.strokeStyle='rgba(216,226,237,.14)';ctx.setLineDash([15,17]);ctx.beginPath();ctx.moveTo(margin+12,trackY);ctx.lineTo(margin+trackW-12,trackY);ctx.stroke();ctx.setLineDash([]);
          const active=activeSiteAt(t);
          let lickNow=performance.now()<lickUntil,currentLickColor=lickColor;
          if(eventNear(D.rewardedLicks,t,.045)){lickNow=true;currentLickColor=COLORS.rewarded;}
          else if(eventNear(D.nonrewardedLicks,t,.045)){lickNow=true;currentLickColor=COLORS.nonrewarded;}
          else if(eventNear(D.invalidLicks,t,.045)){lickNow=true;currentLickColor=COLORS.invalid;}
          let lickSite=0,lickDistance=Infinity;D.sitePositions.forEach((site,index)=>{const raw=Math.abs(sample.position-site),distance=Math.min(raw,period-raw);if(distance<lickDistance){lickDistance=distance;lickSite=index;}});
          const siteColors={solid:'#4A9FC7',active:'rgba(74,159,199,.34)',inactive:'rgba(74,159,199,.075)'};
          D.sitePositions.forEach((site,index)=>{const bounds=(D.siteBounds&&D.siteBounds[index])||[site-5,site+5],nearestLap=Math.round((sample.worldPosition-site)/period),colors=siteColors;for(let lap=nearestLap-1;lap<=nearestLap+1;lap++){const x1=worldX(bounds[0]+lap*period),x2=worldX(bounds[1]+lap*period),left=Math.min(x1,x2),right=Math.max(x1,x2),isActive=index===active,isLickPulse=lickNow&&index===lickSite;if(right<margin||left>margin+trackW)continue;const roadTop=trackY-trackH/2,roadBottom=trackY+trackH/2;ctx.fillStyle=isActive?colors.active:colors.inactive;ctx.fillRect(left,roadTop,right-left,roadBottom-roadTop);if(isLickPulse){ctx.save();ctx.globalAlpha=.38;ctx.shadowColor=currentLickColor;ctx.shadowBlur=10;ctx.fillStyle=currentLickColor;ctx.fillRect(left,roadTop,right-left,roadBottom-roadTop);ctx.restore();}ctx.strokeStyle=isLickPulse?currentLickColor:(isActive?colors.solid:'rgba(74,159,199,.20)');ctx.lineWidth=isLickPulse?3:(isActive?2.5:1);ctx.beginPath();ctx.moveTo(left,roadTop);ctx.lineTo(left,roadBottom);ctx.moveTo(right,roadTop);ctx.lineTo(right,roadBottom);ctx.stroke();ctx.fillStyle=isActive?colors.solid:'rgba(74,159,199,.42)';ctx.font='750 10px Inter, sans-serif';ctx.textAlign='center';ctx.fillText('SITE '+String.fromCharCode(65+index),(left+right)/2,roadTop-13);}});
          drawMouse(ctx,mouseX,trackY-5,sample.speed,t,false,currentLickColor);
          const progress=Math.max(0,Math.min(1,t/D.duration));ctx.fillStyle='rgba(255,255,255,.09)';rounded(ctx,margin,h-36,trackW,4,2).fill();ctx.fillStyle='#8B75C9';rounded(ctx,margin,h-36,trackW*progress,4,2).fill();
        }
        function timelineWindow(t){const size=Math.min(6,Math.max(D.duration,1e-6)),start=t-size*.375;return {start,end:start+size};}
        function drawCursor(ctx,t,start,end,left,width,top,height){const x=left+(t-start)/(end-start)*width;ctx.strokeStyle='#223248';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,top-2);ctx.lineTo(x,top+height+2);ctx.stroke();ctx.fillStyle='#223248';ctx.beginPath();ctx.arc(x,top-2,3,0,Math.PI*2);ctx.fill();}
        function drawTimeLabels(ctx,start,end,left,width,height){ctx.textAlign='center';ctx.fillStyle='#8290A0';ctx.font='10px Inter, sans-serif';for(let k=0;k<=3;k++){const value=start+(end-start)*k/3;if(value<0||value>D.duration)continue;const x=left+width*k/3;ctx.fillText(formatTime(value),x,height-7);}}
        function smoothedSpeed(index){let weighted=0,total=0;for(let offset=-2;offset<=2;offset++){const i=Math.max(0,Math.min(D.speed.length-1,index+offset)),weight=3-Math.abs(offset);weighted+=D.speed[i]*weight;total+=weight;}return weighted/total;}
        function drawSpeedTimeline(t){const {ctx,width:w,height:h}=canvasContext(speedTimeline,125),left=TIMELINE_LEFT,right=TIMELINE_RIGHT,top=8,bottom=22,plotW=w-left-right,plotH=h-top-bottom,{start,end}=timelineWindow(t),peak=Math.max(10,D.summary.peakSpeed*1.08);ctx.clearRect(0,0,w,h);ctx.strokeStyle='rgba(116,128,145,.15)';ctx.lineWidth=1;ctx.font='10px Inter, sans-serif';ctx.fillStyle='#8290A0';ctx.textAlign='right';for(let k=0;k<=2;k++){const y=top+plotH*k/2;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+plotW,y);ctx.stroke();ctx.fillText((peak*(1-k/2)).toFixed(0),left-8,y+3);}const i0=Math.max(0,lowerBound(D.time,start)-1),i1=Math.min(D.time.length-1,lowerBound(D.time,end)+1),points=[];for(let i=i0;i<=i1;i++){const x=left+(D.time[i]-start)/(end-start)*plotW,y=top+plotH-smoothedSpeed(i)/peak*plotH;points.push([x,Math.max(top,Math.min(top+plotH,y))]);}if(points.length){const fill=ctx.createLinearGradient(0,top,0,top+plotH);fill.addColorStop(0,'rgba(72,184,208,.30)');fill.addColorStop(1,'rgba(72,184,208,.02)');ctx.beginPath();ctx.moveTo(points[0][0],top+plotH);for(const p of points)ctx.lineTo(p[0],p[1]);ctx.lineTo(points.at(-1)[0],top+plotH);ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle=COLORS.arrival;ctx.lineWidth=2.2;ctx.stroke();}drawCursor(ctx,t,start,end,left,plotW,top,plotH);drawTimeLabels(ctx,start,end,left,plotW,h);}
        function visibleEvents(events,start,end,objectTimes=false){return events.filter(item=>{const value=objectTimes?item.t:item;return value>=start&&value<=end;});}
        function drawEventTimeline(t){const {ctx,width:w,height:h}=canvasContext(eventTimeline,135),left=TIMELINE_LEFT,right=TIMELINE_RIGHT,top=4,bottom=19,plotW=w-left-right,plotH=h-top-bottom,{start,end}=timelineWindow(t),lanes=[['Arrival (site entry)',D.arrivals,COLORS.arrival,true],['Rewarded lick',D.rewardedLicks,COLORS.rewarded,false],['Non-rewarded',D.nonrewardedLicks,COLORS.nonrewarded,false],['Invalid lick',D.invalidLicks,COLORS.invalid,false],['Site deactivation',D.switches,COLORS.switched,true]],laneH=plotH/lanes.length;ctx.clearRect(0,0,w,h);lanes.forEach((lane,index)=>{const y=top+index*laneH;ctx.fillStyle=index%2?'rgba(116,128,145,.035)':'rgba(116,128,145,.065)';rounded(ctx,left,y+1,plotW,laneH-2,4).fill();ctx.fillStyle=lane[2];ctx.font='700 9px Inter, sans-serif';ctx.textAlign='right';ctx.fillText(lane[0],left-9,y+laneH*.66);for(const item of visibleEvents(lane[1],start,end,lane[3])){const value=lane[3]?item.t:item,x=left+(value-start)/(end-start)*plotW;ctx.strokeStyle=lane[2];ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,y+3);ctx.lineTo(x,y+laneH-4);ctx.stroke();}});drawCursor(ctx,t,start,end,left,plotW,top,plotH);drawTimeLabels(ctx,start,end,left,plotW,h);}
        function drawNeuralTimeline(t){
          neuralViews.forEach(({canvas,activity})=>{
            const {ctx,width:w,height:h}=canvasContext(canvas,145),left=TIMELINE_LEFT,right=TIMELINE_RIGHT,top=7,bottom=21,plotW=w-left-right,plotH=h-top-bottom,{start,end}=timelineWindow(t);
            ctx.clearRect(0,0,w,h);
            if(!activity.time.length){ctx.fillStyle='#8290A0';ctx.font='600 12px Inter, sans-serif';ctx.textAlign='center';ctx.fillText('Regional activity unavailable',w/2,h/2);return;}
            const names=Object.keys(activity.regions),time=activity.time,zMin=-5,zMax=5;
            ctx.strokeStyle='rgba(116,128,145,.15)';ctx.fillStyle='#8290A0';ctx.font='10px Inter, sans-serif';ctx.textAlign='right';
            for(let k=0;k<=2;k++){const value=zMax-(zMax-zMin)*k/2,y=top+plotH*k/2;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+plotW,y);ctx.stroke();ctx.fillText(value.toFixed(0),left-8,y+3);}
            names.forEach((name,index)=>{const values=activity.regions[name],i0=Math.max(0,lowerBound(time,start)-1),i1=Math.min(time.length-1,lowerBound(time,end)+1);ctx.beginPath();for(let i=i0;i<=i1;i++){const x=left+(time[i]-start)/(end-start)*plotW,value=Math.max(zMin,Math.min(zMax,values[i])),y=top+(zMax-value)/(zMax-zMin)*plotH;i===i0?ctx.moveTo(x,y):ctx.lineTo(x,y);}ctx.strokeStyle=(activity.regionColors||{})[name]||REGION_COLORS[index%REGION_COLORS.length];ctx.lineWidth=2.1;ctx.stroke();});
            drawCursor(ctx,t,start,end,left,plotW,top,plotH);drawTimeLabels(ctx,start,end,left,plotW,h);
          });
        }
        function drawSpikeProbes(t){
          spikeViews.forEach(({canvas,activity,probeIndex})=>{
            const {ctx,width:w,height:h}=canvasContext(canvas,270);
            ctx.clearRect(0,0,w,h);
            const names=Object.keys(activity.regions||{}),top=8,bottom=30,shankW=Math.max(30,Math.min(36,w*.38)),x=(w-shankW)/2,usable=h-top-bottom;
            ctx.save();ctx.shadowColor='rgba(72,184,208,.16)';ctx.shadowBlur=12;ctx.fillStyle='#152233';rounded(ctx,x,top,shankW,usable,7).fill();ctx.restore();
            ctx.strokeStyle='rgba(89,111,134,.55)';ctx.lineWidth=1;rounded(ctx,x+.5,top+.5,shankW-1,usable-1,7).stroke();
            if(!activity.time.length||!names.length)return;
            const index=Math.min(activity.time.length-1,Math.max(0,lowerBound(activity.time,t)));
            names.forEach((name,regionIndex)=>{
              const y=top+usable*regionIndex/names.length,segmentH=usable/names.length;
              const z=Math.max(-5,Math.min(5,Number(activity.regions[name][index])||0));
              const firingValues=(activity.firingRates||{})[name]||[],firingRate=Math.max(0,Number(firingValues[index])||0);
              const energy=Math.pow(Math.max(0,Math.min(1,(z-1.05)/3.95)),1.2),color=(activity.regionColors||{})[name]||REGION_COLORS[regionIndex%REGION_COLORS.length];
              const cadence=Math.max(.7,Math.min(12,firingRate)),pulse=.5+.5*Math.sin(2*Math.PI*t*cadence),light=energy*(.25+.75*pulse);
              ctx.save();ctx.globalAlpha=.08+.92*light;ctx.shadowColor=color;ctx.shadowBlur=light>0?5+38*light:0;ctx.fillStyle=color;rounded(ctx,x+3,y+2,shankW-6,Math.max(3,segmentH-4),4).fill();ctx.restore();
              ctx.strokeStyle='rgba(225,235,242,.20)';ctx.lineWidth=.7;
              for(let contact=1;contact<4;contact++){const cy=y+segmentH*contact/4;ctx.beginPath();ctx.moveTo(x+7,cy);ctx.lineTo(x+shankW-7,cy);ctx.stroke();}
              if(segmentH>18){
                ctx.fillStyle='rgba(231,238,244,.88)';ctx.font='700 8px Inter, sans-serif';
                ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(name,w/2,y+segmentH/2);
              }
            });
            ctx.fillStyle='#152233';ctx.beginPath();ctx.moveTo(x+shankW*.22,top+usable-1);ctx.lineTo(x+shankW*.78,top+usable-1);ctx.lineTo(x+shankW*.5,top+usable+10);ctx.closePath();ctx.fill();
          });
        }
        const atlasImages={};
        function atlasImage(url){
          if(!url)return null;if(atlasImages[url])return atlasImages[url];
          const image=new Image();image.onload=()=>render(currentTime);image.src=url;atlasImages[url]=image;return image;
        }
        function drawBrainAtlas(t){
          const {ctx,width:w,height:h}=canvasContext(atlasCanvas,270),slices=((D.atlasSlices||{}).slices||[]);
          ctx.clearRect(0,0,w,h);if(!slices.length)return;
          const slice=slices[Math.max(0,Math.min(slices.length-1,Number(atlasSlider.value)))],atlasTop=6,atlasBottom=20,scale=Math.min(w/274,(h-atlasTop-atlasBottom)/192),drawW=274*scale,drawH=192*scale,drawX=(w-drawW)/2,drawY=atlasTop+(h-atlasTop-atlasBottom-drawH)/2;
          const base=atlasImage(slice.base);if(base&&base.complete)ctx.drawImage(base,drawX,drawY,drawW,drawH);
          Object.values(slice.masks||{}).forEach(url=>{
            const mask=atlasImage(url);if(!mask||!mask.complete)return;
            ctx.save();ctx.globalAlpha=.10;ctx.drawImage(mask,drawX,drawY,drawW,drawH);ctx.restore();
          });
          (D.probeActivities||[]).forEach((activity,probeIndex)=>{
            if(!activity.time.length)return;
            const index=Math.min(activity.time.length-1,Math.max(0,lowerBound(activity.time,t)));
            Object.keys(activity.regions||{}).forEach(name=>{
              const mask=atlasImage((slice.masks||{})[name]);if(!mask||!mask.complete)return;
              const z=Math.max(-5,Math.min(5,Number(activity.regions[name][index])||0));
              const rates=(activity.firingRates||{})[name]||[],rate=Math.max(0,Number(rates[index])||0);
              const energy=Math.pow(Math.max(0,Math.min(1,(z+.10)/4.90)),.82),rawPulse=.5+.5*Math.sin(2*Math.PI*t*Math.max(.7,Math.min(12,rate))),pulse=Math.pow(rawPulse,1.45);
              const light=energy*(.08+.92*pulse);ctx.save();ctx.globalCompositeOperation='screen';ctx.globalAlpha=.16+.84*light;ctx.shadowColor='#FFFFFF';ctx.shadowBlur=14+56*light;ctx.drawImage(mask,drawX,drawY,drawW,drawH);if(light>.30){ctx.globalAlpha=.30+.50*light;ctx.shadowBlur=24+64*light;ctx.drawImage(mask,drawX,drawY,drawW,drawH);}if(light>.68){ctx.globalAlpha=.22*light;ctx.drawImage(mask,drawX,drawY,drawW,drawH);}ctx.restore();
            });
          });
          ctx.fillStyle='#A9B8C8';ctx.font='600 9px Inter, sans-serif';ctx.textAlign='center';ctx.fillText('AP plane '+slice.index,w/2,h-5);
        }
        function updateMetrics(t,sample) {
          const active=activeSiteAt(t), sitePos=D.sitePositions[active] ?? NaN, rawDistance=Math.abs(sample.position-sitePos), distance=Number.isFinite(rawDistance)?Math.min(rawDistance,(D.positionMax-D.positionMin)-rawDistance):NaN;
          byId('speedNow').textContent=sample.speed.toFixed(1);byId('speedMean').textContent=rollingSpeed(t).toFixed(1);byId('positionNow').textContent=sample.position.toFixed(1);
          byId('targetDistance').textContent=Number.isFinite(distance)?distance.toFixed(1)+' cm from active site':'—';
          byId('boutNow').textContent=sample.bout>0?sample.bout:'—';byId('boutProgress').textContent=Math.round(100*t/D.duration)+'% of session';
          byId('siteNow').textContent='Site '+String.fromCharCode(65+active);byId('sitePosition').textContent=Number.isFinite(sitePos)?sitePos.toFixed(0)+' cm · active':'—';
          byId('timeNow').textContent=formatTime(t);byId('percentNow').textContent=Math.round(100*t/D.duration)+'% complete';
          let state='Running', color=COLORS.arrival;
          if(eventNear(D.rewardedLicks,t,.24)){state='Rewarded lick';color=COLORS.rewarded;}
          else if(eventNear(D.nonrewardedLicks,t,.24)){state='Non-rewarded lick';color=COLORS.nonrewarded;}
          else if(eventNear(D.invalidLicks,t,.24)){state='Invalid lick';color=COLORS.invalid;}
          else if(eventNear(D.switches,t,.45,true)){state='Site deactivation';color=COLORS.switched;}
          else if(eventNear(D.arrivals,t,.45,true)){state='Arrival (site entry)';color=COLORS.arrival;}
          else if(Number.isFinite(distance)&&distance<7){state='On site';color='#66798C';}
          else if(sample.speed<1){state='Paused';color='#8290A0';}
          const pill=byId('stateNow');pill.textContent=state;pill.style.background=color;
        }
        function render(t) {
          const sample=sampleAt(t);drawScene(t,sample);drawSpikeProbes(t);drawBrainAtlas(t);drawSpeedTimeline(t);drawEventTimeline(t);drawNeuralTimeline(t);
          scrubber.value=t;byId('clock').textContent=formatTime(t)+' / '+formatTime(D.duration);
          if(performance.now()>toastUntil)toast.classList.remove('show');
        }
        function tick(stamp) {
          if(!playing)return;if(frameStamp===null)frameStamp=stamp;
          const elapsed=(stamp-frameStamp)/1000;frameStamp=stamp;previousTime=currentTime;currentTime=Math.min(D.duration,currentTime+elapsed*Number(playbackRate.value));
          flashEvents(previousTime,currentTime);render(currentTime);
          if(currentTime>=D.duration){playing=false;playButton.textContent='▶';frameStamp=null;return;}
          requestAnimationFrame(tick);
        }
        playButton.addEventListener('click',()=>{if(currentTime>=D.duration)currentTime=0;playing=!playing;playButton.textContent=playing?'❚❚':'▶';frameStamp=null;if(playing)requestAnimationFrame(tick);});
        scrubber.addEventListener('input',()=>{previousTime=currentTime;currentTime=Number(scrubber.value);frameStamp=null;render(currentTime);});
        atlasSlider.addEventListener('input',()=>render(currentTime));
        window.addEventListener('keydown',event=>{if(event.code==='Space'){event.preventDefault();playButton.click();}});
        window.addEventListener('resize',()=>render(currentTime));
        render(0);
      </script>
    </body>
    </html>
    """
    return (
        template.replace("__DATA__", replay_json)
        .replace("__MOUSE__", safe_mouse)
        .replace("__DATE__", safe_date)
        .replace("__MODE_CLASS__", "behavior-only" if behavior_only else "")
    )


BEHAVIOR_COLOR_MAP = {
    "#2563eb": "#4B91B5",
    "#60a5fa": "#78AEC8",
    "#22c55e": "#54A986",
    "#f59e0b": "#C48752",
    "#94a3b8": "#A5B0BC",
    "#ef4444": "#C96372",
    "#7c3aed": "#80639A",
    "#f4d35e": "#C4AF70",
    "#8ec5ff": "#82AFC8",
    "#f29a8e": "#BE817B",
    "#1e293b": NAVY,
    "#64748b": MUTED,
    "#e2e8f0": "#D7E0E8",
    "#cbd5e1": "#C5D0DA",
}


def _behavior_color(color):
    try:
        rgba = to_rgba(color)
        mapped = BEHAVIOR_COLOR_MAP.get(to_hex(rgba).lower())
        return to_rgba(mapped, alpha=rgba[3]) if mapped else rgba
    except (TypeError, ValueError):
        return color


def style_behavior_figure(figure, plot_name):
    """Apply the quiet electrophysiology visual language to Matplotlib plots."""
    if plot_name == "protocol_strip":
        figure.set_size_inches(14, 1.35, forward=True)
    elif plot_name == "session_timeline":
        figure.set_size_inches(14, 4.4, forward=True)
    elif plot_name == "regression":
        figure.set_size_inches(17, 4.8, forward=True)
    elif len(figure.axes) == 1 and plot_name.startswith("session_"):
        figure.set_size_inches(7.2, 4.5, forward=True)

    figure.patch.set_facecolor(PANEL_BG)
    for axis in figure.axes:
        axis.set_facecolor(PANEL_BG)
        if len(figure.axes) == 1:
            axis.set_title("")
        else:
            axis.title.set_color(NAVY)
            axis.title.set_fontfamily("Arial")
            axis.title.set_fontsize(11)
            axis.title.set_fontweight("normal")
        axis.set_axisbelow(True)
        axis.xaxis.label.set_color(MUTED)
        axis.yaxis.label.set_color(MUTED)
        axis.xaxis.label.set_fontsize(10)
        axis.yaxis.label.set_fontsize(10)
        axis.xaxis.label.set_fontweight("normal")
        axis.yaxis.label.set_fontweight("normal")
        axis.tick_params(axis="both", colors=MUTED, labelsize=9, width=.7, length=3)
        axis.grid(color="#D7E0E8", linewidth=.7, alpha=.72)
        for side, spine in axis.spines.items():
            spine.set_visible(side in {"left", "bottom"})
            spine.set_color("#C5D0DA")
            spine.set_linewidth(.8)
        for line in axis.lines:
            line.set_color(_behavior_color(line.get_color()))
            line.set_linewidth(min(float(line.get_linewidth()), 2.2))
        for patch in axis.patches:
            patch.set_facecolor(_behavior_color(patch.get_facecolor()))
            patch.set_edgecolor(_behavior_color(patch.get_edgecolor()))
        for collection in axis.collections:
            faces = collection.get_facecolors()
            edges = collection.get_edgecolors()
            if len(faces):
                collection.set_facecolors([_behavior_color(color) for color in faces])
            if len(edges):
                collection.set_edgecolors([_behavior_color(color) for color in edges])
        for text_item in axis.texts:
            text_item.set_color(_behavior_color(text_item.get_color()))
            text_item.set_fontfamily("Arial")
            text_item.set_fontsize(min(float(text_item.get_fontsize()), 9.5))
            text_item.set_fontweight("normal")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for legend_text in legend.get_texts():
                legend_text.set_color(MUTED)
                legend_text.set_fontfamily("Arial")
                legend_text.set_fontsize(9)
            if legend.get_title() is not None:
                legend.get_title().set_color(MUTED)
                legend.get_title().set_fontfamily("Arial")
                legend.get_title().set_fontsize(9)
    figure.tight_layout(pad=1.25)
    return figure


def figure_to_png(figure, plot_name):
    if figure is None:
        return None
    figure = style_behavior_figure(figure, plot_name)
    figure.patch.set_facecolor(PANEL_BG)
    for axis in figure.axes:
        axis.set_facecolor(PANEL_BG)
    output = BytesIO()
    figure.savefig(output, format="png", dpi=150, bbox_inches="tight", facecolor=PANEL_BG)
    plt.close(figure)
    return output.getvalue()


@st.cache_data(show_spinner="Building plot...", max_entries=256)
def _build_behavior_plot_legacy(path: str, signature, mouse_id: str, plot_name: str, session_key=None):
    plot_style_version = "behavior-panel-v1"
    del plot_style_version
    dataframe = load_behavior_database(path, signature)
    mouse_df = dataframe[dataframe["Mouse_ID"] == mouse_id].copy()
    session_cmap = LinearSegmentedColormap.from_list("session_cmap", ["#2563EB", "#EF4444"])

    if plot_name == "protocol_strip":
        figure = plot_protocol_strip(mouse_df, mouse_id)
    elif plot_name == "bout_count_rewards":
        figure = plot_bout_count_rewards(mouse_df, mouse_id)
    elif plot_name == "stacked_lick_counts":
        figure = plot_stacked_lick_counts(mouse_df, mouse_id)
    elif plot_name == "histogram_kde_failures":
        figure = plot_histogram_kde_failures(mouse_df, mouse_id)
    elif plot_name == "kde_failures_by_session":
        figure = plot_kde_failures_by_session(mouse_df, mouse_id, session_cmap)
    elif plot_name == "regression":
        figure = plot_regression_rewards_failures_and_slope(mouse_df, mouse_id, session_cmap)
    else:
        date_iso, version = session_key
        session_rows = mouse_df[
            (mouse_df["Date"].dt.strftime("%Y-%m-%d") == date_iso)
            & (mouse_df["Version"].astype(str) == str(version))
        ]
        if session_rows.empty:
            return None
        row = session_rows.iloc[0]
        if plot_name == "session_timeline":
            figure = build_session_plot_bout_timeline(row, title="Bout timeline")
        elif plot_name == "session_rewards_failures":
            figure = build_session_plot_rewards_vs_failures(row, mouse_id, date_iso)
        elif plot_name == "session_failure_distribution":
            figure = build_session_plot_failure_distribution(row)
        else:
            raise ValueError(f"Unknown plot: {plot_name}")

    return figure_to_png(figure, plot_name)


BEHAVIOR_PLOT_COLORS = {
    "blue": "#278DBB", "violet": "#7654A8", "green": "#31956E",
    "orange": "#D18432", "rose": "#C94E6B", "gray": "#9AA8B6",
}
BEHAVIOR_PROTOCOL_COLORS = {1: "#C09B2F", 2: "#318FBB", 3: "#B95662", 4: "#80639A"}


def _behavior_date_colors(count):
    """Chronological blue-to-red scale used without a space-hungry date legend."""
    if count <= 0:
        return []
    date_cmap = LinearSegmentedColormap.from_list(
        "behavior_date_gradient", ["#246BCE", "#7A58A8", "#D94A55"]
    )
    return [to_hex(date_cmap(index / max(1, count - 1))) for index in range(count)]


def _finish_behavior_plot(figure, height=390, margin=None, showlegend=False):
    figure = style_detail_figure(figure, height=height)
    figure.update_layout(
        margin=margin or {"l": 54, "r": 24, "t": 16, "b": 48},
        showlegend=showlegend,
        hovermode="closest",
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.01,
            "xanchor": "left", "x": 0, "font": {"size": 10}, "title": None,
        },
    )
    figure.update_xaxes(
        showline=True, linecolor="#C5D0DA", linewidth=1,
        ticks="outside", tickcolor="#C5D0DA", gridcolor="rgba(116,128,145,.16)",
    )
    figure.update_yaxes(
        showline=True, linecolor="#C5D0DA", linewidth=1,
        ticks="outside", tickcolor="#C5D0DA", gridcolor="rgba(116,128,145,.16)",
    )
    return figure


def _valid_failure_values(row):
    values = np.asarray(compute_failures(row), dtype=float)
    valid = valid_bout_mask_from_row(row, target_len=len(values))
    values = values[valid]
    return values[np.isfinite(values) & (values > 0)]


def _failure_distribution_figure(values, max_x=None):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    max_x = int(max_x or min(50, max(5, np.nanmax(values))))
    x = np.arange(1, max_x + 1)
    counts = np.bincount(np.rint(values).astype(int), minlength=max_x + 1)[1:max_x + 1]
    probability = counts / counts.sum() if counts.sum() else counts.astype(float)
    smooth = gaussian_filter1d(probability.astype(float), sigma=1.0)
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=x, y=probability, marker={"color": "#68AACA", "cornerradius": 5,
        "line": {"width": 0}}, name="Observed",
        hovertemplate="%{x} failures<br>%{y:.3f} probability<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=x, y=smooth, mode="lines", line={"color": BEHAVIOR_PLOT_COLORS["blue"], "width": 2.6},
        fill="tozeroy", fillcolor="rgba(39,141,187,.14)", name="Smoothed",
        hovertemplate="%{x} failures<br>%{y:.3f} smoothed probability<extra></extra>",
    ))
    figure.update_xaxes(title="Consecutive failures", range=[.5, max_x + .5])
    figure.update_yaxes(title="Probability", rangemode="tozero")
    return _finish_behavior_plot(figure, height=380, showlegend=False)


@st.cache_data(show_spinner="Building plot...", max_entries=256)
def build_behavior_plot(path: str, signature, mouse_id: str, plot_name: str, session_key=None):
    dataframe = load_behavior_database(path, signature)
    mouse_df = dataframe[dataframe["Mouse_ID"] == mouse_id].copy().sort_values("Date")
    mouse_df["SessionIndex"] = np.arange(1, len(mouse_df) + 1)
    dates = mouse_df["Date"].dt.strftime("%Y-%m-%d")

    if plot_name == "protocol_strip":
        counts = mouse_df["Protocol"].dropna().astype(int).value_counts().reindex([1, 2, 3, 4], fill_value=0)
        figure = go.Figure()
        for protocol in [1, 2, 3, 4]:
            if counts[protocol]:
                figure.add_trace(go.Bar(
                    x=[int(counts[protocol])], y=["Sessions"], orientation="h",
                    name=PROTOCOL_LABELS[protocol], marker={"color": BEHAVIOR_PROTOCOL_COLORS[protocol], "cornerradius": 8},
                    text=[f"{PROTOCOL_LABELS[protocol]} · {int(counts[protocol])}"], textposition="inside",
                    hovertemplate=f"{PROTOCOL_LABELS[protocol]}<br>{int(counts[protocol])} sessions<extra></extra>",
                ))
        figure.update_layout(barmode="stack", bargap=.42)
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return _finish_behavior_plot(figure, height=150, margin={"l": 18, "r": 18, "t": 8, "b": 8}, showlegend=False)

    if plot_name in {"bout_count_rewards", "stacked_lick_counts"}:
        x = mouse_df["SessionIndex"]
        custom = np.column_stack([dates, mouse_df["Protocol"].fillna(0).astype(int)])
        figure = go.Figure()
        if plot_name == "bout_count_rewards":
            valid = mouse_df.apply(count_valid_bouts, axis=1)
            rewarded = pd.to_numeric(mouse_df["Number of Rewarded Licks"], errors="coerce").fillna(0)
            figure.add_trace(go.Scatter(x=x, y=valid, mode="lines+markers", name="Valid bouts",
                line={"color": NAVY, "width": 2.5}, marker={"size": 7}, customdata=custom,
                hovertemplate="%{customdata[0]}<br>%{y} valid bouts<extra></extra>"))
            figure.add_trace(go.Scatter(x=x, y=rewarded, mode="lines+markers", name="Rewarded licks",
                line={"color": BEHAVIOR_PLOT_COLORS["green"], "width": 2.3}, marker={"size": 7, "symbol": "square"}, customdata=custom,
                hovertemplate="%{customdata[0]}<br>%{y} rewarded licks<extra></extra>"))
            figure.update_yaxes(title="Count")
        else:
            groups = [
                ("Rewarded", "rewarded", BEHAVIOR_PLOT_COLORS["green"]),
                ("Non-rewarded", "non_rewarded", BEHAVIOR_PLOT_COLORS["orange"]),
                ("Invalid", "invalid", BEHAVIOR_PLOT_COLORS["gray"]),
            ]
            for label, kind, color in groups:
                values = [count_licks(row, kind) for _, row in mouse_df.iterrows()]
                figure.add_trace(go.Bar(x=x, y=values, name=label, marker={"color": color, "cornerradius": 5}, customdata=custom,
                    hovertemplate="%{customdata[0]}<br>%{y} " + label.lower() + " licks<extra></extra>"))
            figure.update_layout(barmode="stack", bargap=.30)
            figure.update_yaxes(title="Licks")
        figure.update_xaxes(title="Training session", tickmode="array", tickvals=x, ticktext=x)
        return _finish_behavior_plot(figure, height=400, showlegend=True)

    if plot_name == "training_outcomes":
        x = mouse_df["SessionIndex"].to_numpy()
        custom = np.column_stack([dates, mouse_df["Protocol"].fillna(0).astype(int)])
        valid = mouse_df.apply(count_valid_bouts, axis=1).to_numpy(dtype=float)
        rewarded = pd.to_numeric(mouse_df["Number of Rewarded Licks"], errors="coerce").fillna(0).to_numpy(dtype=float)
        duration_minutes = pd.to_numeric(
            mouse_df.get("Session Dur"), errors="coerce"
        ).to_numpy(dtype=float) / 60.0
        rewarded_per_minute = np.divide(
            rewarded, duration_minutes,
            out=np.full_like(rewarded, np.nan, dtype=float),
            where=np.isfinite(duration_minutes) & (duration_minutes > 0),
        )
        protocols = mouse_df["Protocol"].fillna(0).astype(int).to_numpy()
        figure = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=.10,
        )
        metrics = [
            (valid, "Valid bouts", "Bouts", ""),
            (rewarded_per_minute, "Rewarded licks / min", "Rewarded licks / min", " / min"),
        ]
        for column, (values, label, axis_title, value_suffix) in enumerate(metrics, start=1):
            for protocol in [2, 3, 4]:
                keep = (protocols == protocol) & np.isfinite(values)
                if not keep.any():
                    continue
                figure.add_trace(go.Scatter(
                    x=x[keep], y=values[keep], mode="markers",
                    name=PROTOCOL_LABELS[protocol], legendgroup=f"protocol-{protocol}",
                    showlegend=column == 1,
                    marker={
                        "size": 9, "color": BEHAVIOR_PROTOCOL_COLORS[protocol],
                        "line": {"color": PANEL_BG, "width": 1.2},
                    },
                    customdata=custom[keep],
                    hovertemplate=(
                        "%{customdata[0]}<br>" + PROTOCOL_LABELS[protocol]
                        + "<br>%{y:.1f}" + value_suffix + "<extra></extra>"
                    ),
                ), row=1, col=column)
            finite = np.isfinite(values)
            if finite.sum() >= 2 and len(np.unique(x[finite])) >= 2:
                model = LinearRegression().fit(x[finite, None], values[finite])
                line_x = np.array([float(x[finite].min()), float(x[finite].max())])
                line_y = model.predict(line_x[:, None])
                figure.add_trace(go.Scatter(
                    x=line_x, y=line_y, mode="lines", name="Linear trend",
                    legendgroup="linear-trend", showlegend=column == 1,
                    line={"color": "#24364B", "width": 2.6},
                    hoverinfo="skip",
                ), row=1, col=column)
            figure.update_xaxes(
                title="Training session", range=[.5, max(1.5, len(x) + .5)],
                row=1, col=column,
            )
            figure.update_yaxes(
                title=axis_title, rangemode="tozero", row=1, col=column,
            )
        figure = _finish_behavior_plot(
            figure, height=420, showlegend=True,
            margin={"l": 54, "r": 22, "t": 18, "b": 88},
        )
        figure.update_layout(
            legend={
                "orientation": "h", "yanchor": "top", "y": -.20,
                "xanchor": "center", "x": .5, "font": {"size": 10}, "title": None,
            }
        )
        return figure

    if plot_name == "histogram_kde_failures":
        arrays = [_valid_failure_values(row) for _, row in mouse_df.iterrows()]
        arrays = [values for values in arrays if len(values)]
        return _failure_distribution_figure(np.concatenate(arrays) if arrays else np.array([]))

    if plot_name == "kde_failures_by_session":
        sessions = mouse_df[(mouse_df["Protocol"] == 3) & (mouse_df["Proba_val"] == .30)]
        valid_sessions = [(row["Date"], _valid_failure_values(row)) for _, row in sessions.iterrows()]
        valid_sessions = [(date, values) for date, values in valid_sessions if len(values) >= 100]
        if not valid_sessions:
            return None
        figure = go.Figure()
        xs = np.linspace(0, 30, 240)
        colors = _behavior_date_colors(len(valid_sessions))
        for index, (date, values) in enumerate(valid_sessions):
            kde = gaussian_kde(values); kde.set_bandwidth(kde.factor * .8)
            figure.add_trace(go.Scatter(x=xs, y=kde(xs), mode="lines", name=pd.Timestamp(date).strftime("%Y-%m-%d"),
                line={"color": colors[index], "width": 2.5}, showlegend=False,
                hovertemplate="%{x:.1f} failures<br>%{y:.3f} density<extra></extra>"))
        figure.update_xaxes(title="Consecutive failures", range=[0, 30])
        figure.update_yaxes(title="Density", rangemode="tozero")
        return _finish_behavior_plot(figure, height=380, showlegend=False, margin={"l": 54, "r": 20, "t": 18, "b": 48})

    if plot_name == "regression":
        sessions = mouse_df[(mouse_df["Protocol"] == 3) & (mouse_df["Proba_val"] == .30)]
        records = []
        for _, row in sessions.iterrows():
            failures = np.asarray(compute_failures(row), dtype=float)
            rewards = np.asarray(count_reward_per_bout(row), dtype=float)
            count = min(len(failures), len(rewards)); failures, rewards = failures[:count], rewards[:count]
            keep = valid_bout_mask_from_row(row, target_len=count) & np.isfinite(failures) & np.isfinite(rewards) & (failures > 0) & (failures <= 30) & (rewards <= 7)
            if keep.sum() < 100 or len(np.unique(rewards[keep])) < 2: continue
            model = LinearRegression().fit(rewards[keep, None], failures[keep])
            records.append((pd.Timestamp(row["Date"]), keep.sum(), float(model.coef_[0]), float(model.intercept_), float(failures[keep].mean())))
        if not records: return None
        figure = make_subplots(rows=1, cols=3, subplot_titles=("Reward vs failures", "Slope over time", "Mean failures over time"), horizontal_spacing=.09)
        line_x = np.linspace(1, 7, 80); colors = _behavior_date_colors(len(records))
        for index, (date, count, slope, intercept, mean) in enumerate(records):
            figure.add_trace(go.Scatter(x=line_x, y=slope*line_x+intercept, mode="lines", name=date.strftime("%Y-%m-%d"),
                line={"color": colors[index], "width": 2.3}, showlegend=False,
                hovertemplate=date.strftime("%Y-%m-%d") + "<br>%{x:.1f} rewards<br>%{y:.2f} failures<extra></extra>"), row=1, col=1)
        labels = [record[0].strftime("%m-%d") for record in records]; indexes = np.arange(len(records)); slopes = np.array([r[2] for r in records]); means = np.array([r[4] for r in records])
        for column, values in [(2, slopes), (3, means)]:
            trend = fit_exponential_trend_with_band(values)
            if trend is not None:
                figure.add_trace(go.Scatter(
                    x=trend["x_smooth"], y=trend["y_lower"], mode="lines",
                    line={"width": 0}, hoverinfo="skip", showlegend=False,
                ), row=1, col=column)
                figure.add_trace(go.Scatter(
                    x=trend["x_smooth"], y=trend["y_upper"], mode="lines",
                    line={"width": 0}, fill="tonexty",
                    fillcolor="rgba(128,99,154,.14)", hoverinfo="skip", showlegend=False,
                ), row=1, col=column)
                figure.add_trace(go.Scatter(
                    x=trend["x_smooth"], y=trend["y_smooth"], mode="lines",
                    line={"color": BEHAVIOR_PLOT_COLORS["violet"], "width": 2.4},
                    hoverinfo="skip", showlegend=False,
                ), row=1, col=column)
            figure.add_trace(go.Scatter(
                x=indexes, y=values, mode="markers", showlegend=False,
                marker={"color": NAVY, "size": 7},
                customdata=labels,
                hovertemplate="%{customdata}<br>%{y:.2f}<extra></extra>",
            ), row=1, col=column)
            figure.update_xaxes(tickmode="array", tickvals=indexes, ticktext=labels, tickangle=-35, row=1, col=column)
        figure.update_xaxes(title="Reward count", range=[1, 7], row=1, col=1); figure.update_yaxes(title="Consecutive failures", row=1, col=1)
        figure.update_yaxes(title="Linear slope", row=1, col=2); figure.update_yaxes(title="Mean failures", row=1, col=3)
        return _finish_behavior_plot(figure, height=410, showlegend=False, margin={"l": 54, "r": 24, "t": 58, "b": 64})

    date_iso, version = session_key
    rows = mouse_df[(mouse_df["Date"].dt.strftime("%Y-%m-%d") == date_iso) & (mouse_df["Version"].astype(str) == str(version))]
    if rows.empty: return None
    row = rows.iloc[0]
    if plot_name == "session_failure_distribution":
        _, failures, _ = prepare_session_arrays(row, reward_cut=7)
        return _failure_distribution_figure(failures, max_x=25)
    if plot_name == "session_rewards_failures":
        rewards, failures, _ = prepare_session_arrays(row, reward_cut=7)
        reward_bins = np.sort(np.unique(rewards.astype(int)))
        means = np.array([failures[rewards == reward].mean() for reward in reward_bins])
        stds = np.array([failures[rewards == reward].std() for reward in reward_bins])
        counts = np.array([np.sum(rewards == reward) for reward in reward_bins])
        figure = go.Figure()
        if len(reward_bins):
            lower = np.maximum(1, means - stds)
            upper = means + stds
            figure.add_trace(go.Scatter(
                x=reward_bins, y=lower, mode="lines", line={"width": 0},
                hoverinfo="skip", showlegend=False,
            ))
            figure.add_trace(go.Scatter(
                x=reward_bins, y=upper, mode="lines", line={"width": 0},
                fill="tonexty", fillcolor="rgba(75,145,181,.16)",
                hoverinfo="skip", showlegend=False,
            ))
            figure.add_trace(go.Scatter(
                x=reward_bins, y=means, mode="lines+markers", showlegend=False,
                line={"color": BEHAVIOR_PLOT_COLORS["blue"], "width": 2.5},
                marker={"color": PANEL_BG, "line": {"color": BEHAVIOR_PLOT_COLORS["blue"], "width": 2}, "size": 8},
                customdata=np.column_stack((stds, counts)),
                hovertemplate="%{x:.0f} rewards<br>%{y:.2f} mean failures<br>±%{customdata[0]:.2f} SD<br>n=%{customdata[1]:.0f}<extra></extra>",
            ))
        figure.update_xaxes(title="Consecutive rewards", dtick=1, range=[.8, 7.2])
        figure.update_yaxes(title="Consecutive failures", rangemode="tozero")
        return _finish_behavior_plot(figure, height=390)
    if plot_name == "session_timeline":
        data = extract_bout_timeline_data(row)
        if data is None: return None
        centers = data["t_start"] + data["duration"] / 2
        figure = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Rewarded", "Non-rewarded", "Manual reward"),
            vertical_spacing=.13,
        )
        status_text = np.where(data["is_valid"], "Valid", "Invalid")
        event_rows = [
            (data["rewarded"], "#45A77C"),
            (data["non_rewarded"], "#D99A48"),
            (data["manual"], "#638FB5"),
        ]
        if len(centers) > 1:
            positive_spacing = np.diff(np.sort(np.unique(centers)))
            positive_spacing = positive_spacing[positive_spacing > 0]
            bar_width = float(np.nanmedian(positive_spacing) * .28) if len(positive_spacing) else 1.0
        else:
            bar_width = max(1.0, float(data["duration"][0]) * .25) if len(centers) else 1.0
        for row_index, (values, color) in enumerate(event_rows, start=1):
            values = np.asarray(values, dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            custom = np.column_stack((data["bout_id"], status_text))
            figure.add_trace(go.Bar(
                x=centers, y=values, width=bar_width,
                marker={"color": color, "line": {"width": 0}, "cornerradius": 3},
                customdata=custom, showlegend=False,
                hovertemplate=(
                    "Bout %{customdata[0]} · %{customdata[1]}<br>"
                    "%{y:.0f} licks<extra></extra>"
                ),
            ), row=row_index, col=1)
            figure.update_yaxes(rangemode="tozero", nticks=3, title=None, row=row_index, col=1)
            figure.update_xaxes(showgrid=False, row=row_index, col=1)
        figure.update_xaxes(title="Recording time (s)", row=3, col=1)
        figure = _finish_behavior_plot(
            figure, height=350, showlegend=False,
            margin={"l": 48, "r": 18, "t": 34, "b": 50},
        )
        figure.update_annotations(
            font={"size": 11, "color": "#4B5563"},
            x=0, xanchor="left",
        )
        return figure
    raise ValueError(f"Unknown plot: {plot_name}")


def discover_openfield_sessions(openfield_root: Path):
    """Discover OFT, LDT and SIT sessions without assuming a single folder layout."""
    columns = ["mouse", "assay", "date", "session_dir", "videos", "csv_files"]
    rows_by_session = {}
    if not openfield_root.exists():
        return pd.DataFrame(columns=columns)

    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    for mouse_dir in sorted(path for path in openfield_root.iterdir() if path.is_dir()):
        if not is_visible_mouse(mouse_dir.name):
            continue
        assay_dirs = {
            child.name.upper(): child
            for child in mouse_dir.iterdir()
            if child.is_dir() and child.name.upper() in {"OFT", "LDT", "SIT"}
        }
        for assay, assay_dir in assay_dirs.items():
            for session_dir in sorted(path for path in assay_dir.iterdir() if path.is_dir()):
                try:
                    session_date = pd.Timestamp(
                        session_dir.name.replace("_", "-")
                    ).strftime("%Y-%m-%d")
                except ValueError:
                    continue
                videos = sorted(
                    str(path) for path in session_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in video_suffixes
                )
                csv_files = sorted(str(path) for path in session_dir.rglob("*.csv"))
                rows_by_session[(mouse_dir.name, assay, session_date)] = {
                    "mouse": mouse_dir.name,
                    "assay": assay,
                    "date": session_date,
                    "session_dir": str(session_dir),
                    "videos": videos,
                    "csv_files": csv_files,
                }

    # SIT recordings currently arrive as flat files before entering the processed
    # per-mouse tree. Group the alone/visitor recordings into one dated session.
    incoming_sit = openfield_root / "incomming" / "SIT"
    if incoming_sit.exists():
        sit_pattern = re.compile(
            r"^(VF\d+)_(\d{4}_\d{2}_\d{2})_SIT(?:_|\b)", re.IGNORECASE
        )
        for video_path in sorted(path for path in incoming_sit.iterdir() if path.is_file()):
            if video_path.suffix.lower() not in video_suffixes:
                continue
            match = sit_pattern.match(video_path.stem)
            if not match or not is_visible_mouse(match.group(1).upper()):
                continue
            mouse_id = match.group(1).upper()
            session_date = pd.Timestamp(
                match.group(2).replace("_", "-")
            ).strftime("%Y-%m-%d")
            key = (mouse_id, "SIT", session_date)
            row = rows_by_session.setdefault(key, {
                "mouse": mouse_id,
                "assay": "SIT",
                "date": session_date,
                "session_dir": str(incoming_sit),
                "videos": [],
                "csv_files": [],
            })
            row["videos"].append(str(video_path))

    # A complete SIT session is one Alone + Visitor pair. Incomplete acquisition
    # dates stay out of both the counter and the session selector.
    for key, row in list(rows_by_session.items()):
        if row["assay"] != "SIT":
            continue
        video_stems = [Path(path).stem.lower() for path in row["videos"]]
        has_alone = any("alone" in stem for stem in video_stems)
        has_visitor = any("visitor" in stem for stem in video_stems)
        if not (has_alone and has_visitor):
            del rows_by_session[key]

    rows = sorted(
        rows_by_session.values(),
        key=lambda row: (row["mouse"], row["date"], row["assay"]),
    )
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(show_spinner=False)
def load_behavior_metrics_table(database_path, signature, table_name):
    allowed_tables = {
        "oft_sessions", "ldt_sessions", "sit_sessions", "mouse_metrics",
        "paired_behavior_metrics"
    }
    if table_name not in allowed_tables:
        raise ValueError(f"Unsupported behavior metrics table: {table_name}")
    connection = sqlite3.connect(f"file:{Path(database_path).as_posix()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', connection)
    finally:
        connection.close()
    for column in ("oft_date", "ldt_date", "sit_date"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(
                frame[column].astype(str).str.replace("_", "-", regex=False),
                errors="coerce",
            )
    return frame


def behavior_metrics_table(table_name):
    if not BEHAVIOR_METRICS_DB.exists():
        return pd.DataFrame()
    try:
        return load_behavior_metrics_table(
            str(BEHAVIOR_METRICS_DB), source_signature(BEHAVIOR_METRICS_DB), table_name
        )
    except (sqlite3.DatabaseError, pd.errors.DatabaseError):
        return pd.DataFrame()


def metric_display_name(metric_name):
    label = re.sub(r"^(OFT|LDT|SIT|z_OFT|z_LDT|z_SIT)_", "", str(metric_name))
    return label.replace("_", " ").strip().title()


def casefold_child(parent: Path, child_name: str):
    """Resolve one filesystem child without assuming filename capitalization."""
    parent = Path(parent)
    direct = parent / child_name
    if direct.exists() or not parent.is_dir():
        return direct
    wanted = child_name.casefold()
    try:
        return next(
            child for child in parent.iterdir()
            if child.name.casefold() == wanted
        )
    except (StopIteration, OSError):
        return direct


def casefold_path(base: Path, *parts):
    resolved = Path(base)
    for part in parts:
        resolved = casefold_child(resolved, str(part))
    return resolved


def discover_ibl_channel_locations(ephys_root: Path):
    rows_by_insertion = {}
    if not ephys_root.exists():
        return pd.DataFrame(
            columns=["mouse", "date", "probe", "path", "session_dir", "source"]
        )

    for mouse_dir in sorted(path for path in ephys_root.iterdir() if path.is_dir()):
        for session_dir in sorted(path for path in mouse_dir.iterdir() if path.is_dir()):
            try:
                session_date = pd.Timestamp(session_dir.name.replace("_", "-")).strftime("%Y-%m-%d")
            except ValueError:
                continue
            alf_dir = casefold_path(session_dir, "alf")
            if not alf_dir.exists():
                continue
            for probe_dir in sorted(
                path for path in alf_dir.iterdir()
                if path.is_dir() and path.name.casefold().startswith("probe")
            ):
                locations_path = casefold_path(probe_dir, "channel_locations.json")
                if locations_path.exists():
                    probe_name = probe_dir.name.casefold()
                    key = (mouse_dir.name, session_date, probe_name)
                    rows_by_insertion[key] = {
                        "mouse": mouse_dir.name,
                        "date": session_date,
                        "probe": probe_name,
                        "path": str(locations_path),
                        "session_dir": str(session_dir),
                        "source": "IBL adjusted",
                    }

        # Brainreg tracks are used only when a final IBL GUI alignment is absent.
        tracks_dir = casefold_path(
            mouse_dir, "brainreg", "segmentation", "atlas_space", "tracks"
        )
        if tracks_dir.exists():
            pattern = re.compile(
                rf"^{re.escape(mouse_dir.name)}_(?P<date>\d{{4}}_\d{{2}}_\d{{2}})_(?P<probe>probe\d+)$",
                flags=re.IGNORECASE,
            )
            for track_path in sorted(tracks_dir.glob("*.npy")):
                match = pattern.match(track_path.stem)
                if not match:
                    continue
                session_date = pd.Timestamp(match.group("date").replace("_", "-")).strftime("%Y-%m-%d")
                probe = match.group("probe").casefold()
                key = (mouse_dir.name, session_date, probe)
                rows_by_insertion.setdefault(
                    key,
                    {
                        "mouse": mouse_dir.name,
                        "date": session_date,
                        "probe": probe,
                        "path": str(track_path),
                        "session_dir": str(mouse_dir / match.group("date")),
                        "source": "Brainreg initial",
                    },
                )

    return pd.DataFrame(rows_by_insertion.values())


def channel_locations_signature(inventory: pd.DataFrame):
    watched = []
    for row in inventory.itertuples(index=False):
        locations_path = Path(row.path)
        session_dir = Path(row.session_dir)
        related = [
            locations_path,
            casefold_path(session_dir, "bombcell", row.probe, "bombcell_labels.csv"),
        ]
        for path in related:
            if path.exists():
                watched.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
    return tuple(watched)


@st.cache_data(show_spinner="Loading final IBL channel locations...", max_entries=16)
def load_ibl_probe_tracks(track_rows, signature):
    del signature
    tracks = []
    for row in track_rows:
        with Path(row["path"]).open(encoding="utf-8") as locations_file:
            channel_locations = json.load(locations_file)

        channel_rows = []
        for channel_name, location in channel_locations.items():
            if not channel_name.startswith("channel_") or not isinstance(location, dict):
                continue
            xyz = [location.get(axis) for axis in ("x", "y", "z")]
            if any(value is None for value in xyz):
                continue
            channel_rows.append(
                {
                    "axial": float(location.get("axial", len(channel_rows))),
                    "ml": float(xyz[0]),
                    "ap": float(xyz[1]),
                    "dv": float(xyz[2]),
                }
            )

        if len(channel_rows) < 2:
            continue

        # Paired Neuropixels sites share an axial coordinate. Averaging them
        # avoids giving duplicated sites extra weight in the straight-line fit.
        channels = pd.DataFrame(channel_rows).groupby("axial", as_index=False).mean()
        xyz_um = channels[["ml", "ap", "dv"]].to_numpy(dtype=float)

        # IBL JSON: ML/AP/DV in µm relative to Bregma.
        # Allen mesh: AP/DV/ML in µm from the anterior/superior/right volume edge.
        ccf_mlapdv = ALLEN_BREGMA_MLAPDV_UM + xyz_um * np.array([1.0, -1.0, -1.0])
        coordinates_apdvml = ccf_mlapdv[:, [1, 2, 0]]
        regions = sorted(
            {
                str(location.get("brain_region"))
                for channel_name, location in channel_locations.items()
                if channel_name.startswith("channel_")
                and isinstance(location, dict)
                and location.get("brain_region")
            }
        )
        session_dir = Path(row["path"]).parents[2]
        labels_path = casefold_path(
            session_dir, "bombcell", row["probe"], "bombcell_labels.csv"
        )
        total_units = 0
        good_units = 0
        if labels_path.exists():
            labels = pd.read_csv(labels_path, index_col=0)
            total_units = len(labels)
            if "bombcell_label" in labels.columns:
                good_units = int((labels["bombcell_label"] == "good").sum())

        tracks.append(
            {
                **row,
                "coordinates": coordinates_apdvml,
                "regions": regions,
                "total_units": total_units,
                "good_units": good_units,
            }
        )

    return tracks


@st.cache_data(show_spinner="Loading probe trajectories...", max_entries=16)
def load_probe_tracks(track_rows, signature):
    del signature
    tracks = []
    for row in track_rows:
        track_path = Path(row["path"])
        if row["source"] == "IBL adjusted":
            with track_path.open(encoding="utf-8") as locations_file:
                channel_locations = json.load(locations_file)
            channel_rows = []
            for channel_name, location in channel_locations.items():
                if not channel_name.startswith("channel_") or not isinstance(location, dict):
                    continue
                xyz = [location.get(axis) for axis in ("x", "y", "z")]
                if any(value is None for value in xyz):
                    continue
                channel_rows.append(
                    {
                        "axial": float(location.get("axial", len(channel_rows))),
                        "ml": float(xyz[0]),
                        "ap": float(xyz[1]),
                        "dv": float(xyz[2]),
                    }
                )
            if len(channel_rows) < 2:
                continue
            channels = pd.DataFrame(channel_rows).groupby("axial", as_index=False).mean()
            xyz_um = channels[["ml", "ap", "dv"]].to_numpy(dtype=float)
            ccf_mlapdv = ALLEN_BREGMA_MLAPDV_UM + xyz_um * np.array([1.0, -1.0, -1.0])
            coordinates = ccf_mlapdv[:, [1, 2, 0]]
            regions = sorted(
                {
                    str(location.get("brain_region"))
                    for channel_name, location in channel_locations.items()
                    if channel_name.startswith("channel_")
                    and isinstance(location, dict)
                    and location.get("brain_region")
                }
            )
        else:
            coordinates = np.asarray(np.load(track_path), dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[1] != 3 or len(coordinates) < 2:
                continue
            regions = []
            regions_path = track_path.with_suffix(".csv")
            if regions_path.exists():
                region_table = pd.read_csv(regions_path)
                if "Region acronym" in region_table.columns:
                    regions = sorted(region_table["Region acronym"].dropna().astype(str).unique())

        session_dir = Path(row["session_dir"])
        labels_path = casefold_path(
            session_dir, "bombcell", row["probe"], "bombcell_labels.csv"
        )
        total_units = 0
        good_units = 0
        if labels_path.exists():
            labels = pd.read_csv(labels_path, index_col=0)
            total_units = len(labels)
            if "bombcell_label" in labels.columns:
                good_units = int((labels["bombcell_label"] == "good").sum())

        tracks.append(
            {
                **row,
                "coordinates": coordinates,
                "regions": regions,
                "total_units": total_units,
                "good_units": good_units,
            }
        )
    return tracks


def resolve_probe_targets(tracks):
    """Assign session targets from insertion count and AP position."""
    targets = {}
    for session in {track["date"] for track in tracks}:
        session_tracks = [track for track in tracks if track["date"] == session]
        if len(session_tracks) == 1:
            targets[(session, session_tracks[0]["probe"])] = "ALM"
            continue
        frontal_first = sorted(
            session_tracks,
            key=lambda track: float(np.nanmedian(track["coordinates"][:, 0])),
        )
        for index, track in enumerate(frontal_first):
            targets[(session, track["probe"])] = "ALM" if index == 0 else "DLS"
    return targets


def probe_analysis_signature(locations_path: Path, session_dir=None, probe=None):
    if session_dir is not None:
        session_dir = Path(session_dir)
        probe_dir = casefold_path(session_dir, "alf", probe)
        watched = [
            locations_path,
            locations_path.with_suffix(".csv"),
            casefold_path(probe_dir, "clusters.metrics.csv"),
            casefold_path(probe_dir, "clusters.channels.npy"),
            casefold_path(probe_dir, "spikes.times.npy"),
            casefold_path(probe_dir, "spikes.depths.npy"),
            casefold_path(probe_dir, "spikes.clusters.npy"),
            casefold_path(session_dir, "bombcell", probe, "bombcell_labels.csv"),
            casefold_path(session_dir, "shift", probe, "alignment_affine.json"),
        ]
        return tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in watched
            if path.exists()
        )
    probe_dir = locations_path.parent
    session_dir = probe_dir.parents[1]
    probe = probe_dir.name
    watched = [
        locations_path,
        casefold_path(probe_dir, "clusters.metrics.csv"),
        casefold_path(probe_dir, "clusters.channels.npy"),
        casefold_path(probe_dir, "spikes.times.npy"),
        casefold_path(probe_dir, "spikes.depths.npy"),
        casefold_path(probe_dir, "spikes.clusters.npy"),
        casefold_path(session_dir, "bombcell", probe, "bombcell_labels.csv"),
        casefold_path(session_dir, "shift", probe, "alignment_affine.json"),
    ]
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in watched
        if path.exists()
    )


@st.cache_data(show_spinner="Loading session metrics...", max_entries=32)
def load_probe_analysis(
    locations_path: str,
    signature,
    source="IBL adjusted",
    session_path=None,
    probe_name=None,
    on_site_arrival_times=(),
):
    del signature
    locations_path = Path(locations_path)
    if source == "IBL adjusted":
        probe_dir = locations_path.parent
        session_dir = probe_dir.parents[1]
        probe = probe_dir.name
        with locations_path.open(encoding="utf-8") as locations_file:
            channel_locations = json.load(locations_file)
    else:
        session_dir = Path(session_path)
        probe = probe_name
        probe_dir = casefold_path(session_dir, "alf", probe)
        channel_locations = {}
    channel_records = []
    for channel_name, location in channel_locations.items():
        suffix = channel_name.rsplit("_", 1)[-1]
        if not suffix.isdigit() or not isinstance(location, dict):
            continue
        channel_records.append(
            {
                "channel_id": int(suffix),
                "depth_um": float(location.get("axial", np.nan)),
                "brain_region": str(location.get("brain_region", "Unknown")),
            }
        )
    if channel_records:
        channels = pd.DataFrame(channel_records).drop_duplicates("channel_id").set_index("channel_id")
    else:
        channels = pd.DataFrame(columns=["depth_um", "brain_region"])

    metrics_path = casefold_path(probe_dir, "clusters.metrics.csv")
    channels_path = casefold_path(probe_dir, "clusters.channels.npy")
    labels_path = casefold_path(session_dir, "bombcell", probe, "bombcell_labels.csv")

    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    peak_channels = np.load(channels_path) if channels_path.exists() else np.array([], dtype=int)
    labels = pd.read_csv(labels_path, index_col=0) if labels_path.exists() else pd.DataFrame()
    if not labels.empty:
        labels.index = labels.index.astype(int)

    unit_count = max(len(peak_channels), len(labels), len(metrics))
    units = pd.DataFrame({"unit_id": np.arange(unit_count, dtype=int)})
    if len(peak_channels):
        peak_channel_by_unit = pd.Series(np.asarray(peak_channels, dtype=int))
        units["peak_channel"] = units["unit_id"].map(peak_channel_by_unit)
        units["depth_um"] = units["peak_channel"].map(channels["depth_um"])
        units["brain_region"] = units["peak_channel"].map(channels["brain_region"])
    else:
        units["depth_um"] = np.nan
        units["brain_region"] = "Unknown"

    if "bombcell_label" in labels.columns:
        units["bombcell_label"] = units["unit_id"].map(labels["bombcell_label"])
    else:
        units["bombcell_label"] = "unlabelled"

    metric_columns = [
        "cluster_id",
        "num_spikes",
        "firing_rate",
        "presence_ratio",
        "snr",
        "rp_contamination",
        "amplitude_cutoff",
        "amplitude_median",
        "drift_ptp",
    ]
    available = [column for column in metric_columns if column in metrics.columns]
    if "cluster_id" in available:
        units = units.merge(metrics[available], left_on="unit_id", right_on="cluster_id", how="left")
        units = units.drop(columns="cluster_id")

    for column in metric_columns[1:]:
        if column not in units.columns:
            units[column] = np.nan

    units["bombcell_label"] = units["bombcell_label"].fillna("unlabelled")
    units["brain_region"] = units["brain_region"].fillna("Unknown")

    alignment_path = casefold_path(session_dir, "shift", probe, "alignment_affine.json")
    alignment = {}
    if alignment_path.exists():
        with alignment_path.open(encoding="utf-8") as alignment_file:
            alignment = json.load(alignment_file)
    clock_a = float(alignment.get("a", 1.0))
    clock_b = float(alignment.get("b", 0.0))

    spikes_path = casefold_path(probe_dir, "spikes.times.npy")
    spike_depths_path = casefold_path(probe_dir, "spikes.depths.npy")
    spike_clusters_path = casefold_path(probe_dir, "spikes.clusters.npy")
    duration_s = np.nan
    activity_heatmap = None
    peri_bout = None
    regional_activity = None
    if spikes_path.exists():
        spike_times = np.load(spikes_path, mmap_mode="r")
        if len(spike_times) > 1:
            start_s = float(spike_times[0])
            stop_s = float(spike_times[-1])
            duration_s = stop_s - start_s
            behavior_start_s = clock_a * start_s + clock_b
            behavior_stop_s = clock_a * stop_s + clock_b
            if spike_clusters_path.exists():
                spike_clusters = np.load(spike_clusters_path, mmap_mode="r")
                spike_count = min(len(spike_times), len(spike_clusters))
                cluster_values = np.asarray(spike_clusters[:spike_count], dtype=int)
                valid_cluster_values = cluster_values[
                    (cluster_values >= 0) & (cluster_values < len(units))
                ]
                computed_spike_counts = np.bincount(
                    valid_cluster_values, minlength=len(units)
                )[:len(units)].astype(float)
                units["num_spikes"] = pd.to_numeric(
                    units["num_spikes"], errors="coerce"
                ).fillna(pd.Series(computed_spike_counts, index=units.index))
                if duration_s > 0:
                    computed_firing_rates = computed_spike_counts / duration_s
                    units["firing_rate"] = pd.to_numeric(
                        units["firing_rate"], errors="coerce"
                    ).fillna(pd.Series(computed_firing_rates, index=units.index))
                good_unit_table = units[units["bombcell_label"] == "good"].copy()
                good_unit_table = good_unit_table.dropna(subset=["peak_channel"])
                good_unit_table = good_unit_table.sort_values("depth_um", na_position="last")
                good_cluster_ids = good_unit_table["unit_id"].to_numpy(dtype=int)
                if len(good_cluster_ids) and spike_count:
                    channel_count = 384
                    time_edges = np.linspace(behavior_start_s, behavior_stop_s, 301)
                    counts = np.zeros((channel_count, len(time_edges) - 1), dtype=float)
                    cluster_lookup = np.full(int(good_cluster_ids.max()) + 1, -1, dtype=int)
                    cluster_lookup[good_cluster_ids] = good_unit_table["peak_channel"].to_numpy(dtype=int)
                    unit_rank_lookup = np.full(len(cluster_lookup), -1, dtype=int)
                    unit_rank_lookup[good_cluster_ids] = np.arange(len(good_cluster_ids))

                    replay_region_counts = good_unit_table["brain_region"].astype(str).value_counts()
                    replay_regions = [
                        region for region, count in replay_region_counts.items()
                        if region.lower() != "unknown" and count >= 10
                    ]
                    regional_bin_s = 0.10
                    regional_edges = np.arange(
                        np.floor(behavior_start_s / regional_bin_s) * regional_bin_s,
                        behavior_stop_s + 2 * regional_bin_s,
                        regional_bin_s,
                    )
                    regional_counts = np.zeros(
                        (len(replay_regions), max(0, len(regional_edges) - 1)), dtype=np.float32
                    )
                    regional_cluster_lookup = np.full(len(cluster_lookup), -1, dtype=int)
                    replay_region_codes = {region: index for index, region in enumerate(replay_regions)}
                    for cluster_id, region in zip(
                        good_cluster_ids, good_unit_table["brain_region"].astype(str)
                    ):
                        if region in replay_region_codes:
                            regional_cluster_lookup[cluster_id] = replay_region_codes[region]

                    chunk_size = 1_000_000
                    for chunk_start in range(0, spike_count, chunk_size):
                        chunk_stop = min(chunk_start + chunk_size, spike_count)
                        chunk_times = (
                            clock_a * np.asarray(spike_times[chunk_start:chunk_stop], dtype=float)
                            + clock_b
                        )
                        chunk_clusters = np.asarray(
                            spike_clusters[chunk_start:chunk_stop], dtype=int
                        )
                        valid = (
                            np.isfinite(chunk_times)
                            & (chunk_clusters >= 0)
                            & (chunk_clusters < len(cluster_lookup))
                        )
                        if len(replay_regions):
                            region_rows = np.full(len(chunk_clusters), -1, dtype=int)
                            region_rows[valid] = regional_cluster_lookup[chunk_clusters[valid]]
                            regional_bins = np.searchsorted(
                                regional_edges, chunk_times, side="right"
                            ) - 1
                            regional_valid = (
                                valid
                                & (region_rows >= 0)
                                & (regional_bins >= 0)
                                & (regional_bins < regional_counts.shape[1])
                            )
                            np.add.at(
                                regional_counts,
                                (region_rows[regional_valid], regional_bins[regional_valid]),
                                1,
                            )
                        rows = np.full(len(chunk_clusters), -1, dtype=int)
                        rows[valid] = cluster_lookup[chunk_clusters[valid]]
                        time_bins = np.searchsorted(time_edges, chunk_times, side="right") - 1
                        valid &= (
                            (rows >= 0)
                            & (rows < channel_count)
                            & (time_bins >= 0)
                            & (time_bins < counts.shape[1])
                        )
                        np.add.at(counts, (rows[valid], time_bins[valid]), 1)

                    if len(replay_regions):
                        region_rates = regional_counts / regional_bin_s
                        for region_index, region in enumerate(replay_regions):
                            region_rates[region_index] /= max(1, int(replay_region_counts[region]))
                        region_rates = gaussian_filter1d(region_rates, sigma=1.15, axis=1)
                        absolute_region_rates = region_rates.copy()
                        region_means = region_rates.mean(axis=1, keepdims=True)
                        region_stds = region_rates.std(axis=1, keepdims=True)
                        region_stds[region_stds < 1e-9] = 1.0
                        region_rates = np.clip(
                            (region_rates - region_means) / region_stds, -4.0, 4.0
                        )
                        regional_activity = {
                            "time_s": (regional_edges[:-1] + regional_edges[1:]) / 2,
                            "regions": {
                                region: region_rates[index]
                                for index, region in enumerate(replay_regions)
                            },
                            "firing_rates": {
                                region: absolute_region_rates[index]
                                for index, region in enumerate(replay_regions)
                            },
                            "unit_counts": {
                                region: int(replay_region_counts[region])
                                for region in replay_regions
                            },
                            "probe": probe,
                            "scale": "z-score",
                        }
                    bin_width_s = float(np.diff(time_edges).mean())
                    rates = gaussian_filter1d(counts / bin_width_s, sigma=0.65, axis=1)
                    low = np.percentile(rates, 10, axis=1, keepdims=True)
                    high = np.percentile(rates, 99, axis=1, keepdims=True)
                    scale = high - low
                    scale[scale <= 0] = 1
                    relative_activity = np.clip((rates - low) / scale, 0, 1)
                    channel_metadata = channels.reindex(np.arange(channel_count))
                    good_per_channel = good_unit_table["peak_channel"].value_counts()
                    labels_for_hover = []
                    for channel_id in range(channel_count):
                        metadata = channel_metadata.loc[channel_id]
                        depth_value = metadata.get("depth_um", np.nan)
                        depth = f"{depth_value:.0f} µm" if pd.notna(depth_value) else "depth n/a"
                        region = metadata.get("brain_region", "Unknown")
                        unit_count = int(good_per_channel.get(channel_id, 0))
                        labels_for_hover.append(
                            f"Channel {channel_id} · {region} · {depth} · {unit_count} good units"
                        )
                    activity_heatmap = {
                        "time_min": ((time_edges[:-1] + time_edges[1:]) / 2) / 60,
                        "channel_labels": labels_for_hover,
                        "good_unit_count": len(good_unit_table),
                        "relative_activity": relative_activity,
                    }
                    arrival_times_array = np.asarray(on_site_arrival_times, dtype=float)
                    arrival_times_array = arrival_times_array[np.isfinite(arrival_times_array)]
                    peri_edges = np.linspace(-4, 8, 121)
                    peri_rates = []
                    region_names = good_unit_table["brain_region"].fillna("Unknown").astype(str)
                    region_counts = region_names.value_counts()
                    plotted_regions = [
                        region for region, count in region_counts.items()
                        if region.lower() != "unknown" and count >= 10
                    ]
                    cluster_region_lookup = np.full(len(cluster_lookup), -1, dtype=int)
                    region_code_by_name = {region: index for index, region in enumerate(plotted_regions)}
                    for cluster_id, region in zip(good_cluster_ids, region_names):
                        if region in region_code_by_name:
                            cluster_region_lookup[cluster_id] = region_code_by_name[region]
                    peri_region_rates = {region: [] for region in plotted_regions}
                    for arrival_time in arrival_times_array:
                        ephys_left = (arrival_time + peri_edges[0] - clock_b) / clock_a
                        ephys_right = (arrival_time + peri_edges[-1] - clock_b) / clock_a
                        left = int(np.searchsorted(spike_times, ephys_left))
                        right = int(np.searchsorted(spike_times, ephys_right))
                        local_times = (
                            clock_a * np.asarray(spike_times[left:right], dtype=float)
                            + clock_b
                            - arrival_time
                        )
                        local_clusters = np.asarray(spike_clusters[left:right], dtype=int)
                        local_valid = (
                            (local_clusters >= 0)
                            & (local_clusters < len(unit_rank_lookup))
                        )
                        local_good = np.zeros(len(local_clusters), dtype=bool)
                        local_good[local_valid] = unit_rank_lookup[local_clusters[local_valid]] >= 0
                        hist, _ = np.histogram(local_times[local_good], bins=peri_edges)
                        peri_rates.append(hist / (len(good_cluster_ids) * np.diff(peri_edges)))
                        local_region_codes = np.full(len(local_clusters), -1, dtype=int)
                        local_region_codes[local_valid] = cluster_region_lookup[local_clusters[local_valid]]
                        for region, region_code in region_code_by_name.items():
                            region_hist, _ = np.histogram(
                                local_times[local_region_codes == region_code], bins=peri_edges
                            )
                            peri_region_rates[region].append(
                                region_hist / (region_counts[region] * np.diff(peri_edges))
                            )
                    if peri_rates:
                        peri_rates = np.asarray(peri_rates)
                        peri_bout = {
                            "time_s": (peri_edges[:-1] + peri_edges[1:]) / 2,
                            "mean_hz": peri_rates.mean(axis=0),
                            "sem_hz": peri_rates.std(axis=0, ddof=1) / np.sqrt(len(peri_rates)),
                            "arrival_count": len(peri_rates),
                            "regions": {
                                region: np.asarray(region_rates).mean(axis=0)
                                for region, region_rates in peri_region_rates.items()
                                if region_rates
                            },
                        }
    return {
        "units": units,
        "channels": channels.reset_index(),
        "activity_heatmap": activity_heatmap,
        "peri_bout": peri_bout,
        "regional_activity": regional_activity,
        "duration_s": duration_s,
        "alignment": alignment,
    }


def build_lick_svm_view(spike_times, spike_clusters, good_units, lick_bouts):
    """Compact MOs population-state SVM from the ephys analysis notebook."""
    if not lick_bouts:
        return None
    mos_units = good_units[
        good_units["brain_region"].astype(str).str.startswith("MOs")
    ]["unit_id"].to_numpy(dtype=int)
    if len(mos_units) < 10:
        return None

    spike_count = min(len(spike_times), len(spike_clusters))
    times = np.asarray(spike_times[:spike_count], dtype=float)
    clusters = np.asarray(spike_clusters[:spike_count], dtype=int)
    mask = np.isin(clusters, mos_units)
    times, clusters = times[mask], clusters[mask]
    if not len(times):
        return None

    bin_s = 0.10
    edges = np.arange(
        min(bout["first_lick"] for bout in lick_bouts) - 1,
        max(bout["last_lick"] for bout in lick_bouts) + 1 + bin_s,
        bin_s,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    unit_ids = np.unique(clusters)
    population = np.zeros((len(centers), len(unit_ids)), dtype=np.float32)
    for index, unit_id in enumerate(unit_ids):
        counts, _ = np.histogram(times[clusters == unit_id], bins=edges)
        population[:, index] = gaussian_filter1d(counts / bin_s, sigma=3.0)

    bout_mask = np.zeros(len(centers), dtype=bool)
    for bout in lick_bouts:
        start = np.argmin(np.abs(centers - bout["first_lick"]))
        stop = np.argmin(np.abs(centers - bout["last_lick"]))
        bout_mask[min(start, stop): max(start, stop) + 1] = True
    if bout_mask.sum() < 10:
        return None

    scaled_bouts = StandardScaler().fit_transform(population[bout_mask])
    lookup = np.full(len(centers), -1, dtype=int)
    lookup[np.flatnonzero(bout_mask)] = np.arange(len(scaled_bouts))
    rewarded_points, nonrewarded_points, last_points = [], [], []
    for bout in lick_bouts:
        last_index = lookup[np.argmin(np.abs(centers - bout["last_lick"]))]
        if last_index >= 0:
            last_points.append(scaled_bouts[last_index])
        for group, destination in (
            (bout["rewarded"], rewarded_points),
            (bout["nonrewarded"], nonrewarded_points),
        ):
            for lick_time in group:
                lick_index = lookup[np.argmin(np.abs(centers - lick_time))]
                if lick_index >= 0:
                    destination.append(scaled_bouts[lick_index])
    rewarded_points = np.asarray(rewarded_points)
    nonrewarded_points = np.asarray(nonrewarded_points)
    last_points = np.asarray(last_points)
    if min(len(rewarded_points), len(nonrewarded_points), len(last_points)) < 2:
        return None

    other_points = np.vstack([rewarded_points, nonrewarded_points])
    features = np.vstack([last_points, other_points])
    labels = np.r_[np.ones(len(last_points), dtype=int), np.zeros(len(other_points), dtype=int)]
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=20_000).fit(features, labels)
    normal = svm.coef_[0] / np.linalg.norm(svm.coef_[0])
    boundary = -svm.intercept_[0] / np.linalg.norm(svm.coef_[0])
    pca = PCA(n_components=min(3, scaled_bouts.shape[1])).fit(scaled_bouts)
    tangent = next(
        (
            candidate / np.linalg.norm(candidate)
            for component in pca.components_
            if np.linalg.norm(candidate := component - np.dot(component, normal) * normal) > 1e-12
        ),
        None,
    )
    if tangent is None:
        return None
    project = lambda points: np.column_stack([points @ tangent, points @ normal])
    return {
        "rewarded": project(rewarded_points),
        "nonrewarded": project(nonrewarded_points),
        "last": project(last_points),
        "boundary": boundary,
        "margin": 1 / np.linalg.norm(svm.coef_[0]),
        "unit_count": len(mos_units),
        "accuracy": svm.score(features, labels),
    }


def atlas_mesh_signature(mesh_path: Path):
    stat = mesh_path.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner="Loading Allen atlas surface...", max_entries=4)
def load_obj_mesh(mesh_path: str, signature):
    del signature
    vertices = []
    faces = []
    with Path(mesh_path).open("r", encoding="utf-8") as mesh_file:
        for line in mesh_file:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = [int(value.split("/", 1)[0]) - 1 for value in line.split()[1:]]
                for index in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[index], indices[index + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


@st.cache_data(show_spinner=False)
def load_atlas_structure_table(structures_path: str, signature):
    del signature
    return pd.read_csv(structures_path)


def replay_mesh_payload(mesh, max_faces):
    """Compact, normalized triangle sample for the fixed Replay atlas renderer."""
    vertices, faces = mesh
    if not len(vertices) or not len(faces):
        return []
    take = np.linspace(0, len(faces) - 1, min(max_faces, len(faces)), dtype=int)
    triangles = vertices[faces[take]][:, :, [2, 0, 1]].astype(float)
    return np.round(triangles, 1).tolist()


def build_replay_atlas_geometry(brain_mesh, exact_region_meshes):
    vertices = brain_mesh[0][:, [2, 0, 1]].astype(float)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)) / 2)
    scale = scale if np.isfinite(scale) and scale > 0 else 1.0

    def mesh_payload(mesh):
        mesh_vertices, faces = mesh
        xyz = (mesh_vertices[:, [2, 0, 1]].astype(float) - center) / scale
        return {
            "x": np.round(xyz[:, 0], 4).tolist(),
            "y": np.round(xyz[:, 1], 4).tolist(),
            "z": np.round(-xyz[:, 2], 4).tolist(),
            "i": faces[:, 0].astype(int).tolist(),
            "j": faces[:, 1].astype(int).tolist(),
            "k": faces[:, 2].astype(int).tolist(),
        }

    return {
        "brain": mesh_payload(brain_mesh),
        "regions": {region: mesh_payload(mesh) for region, mesh in exact_region_meshes.items()},
    }


@st.cache_data(show_spinner="Preparing Allen atlas slices...", max_entries=12)
def build_replay_atlas_slices(annotation_path, annotation_signature, structures_path, structures_signature, region_names, default_ap_frame):
    del annotation_signature, structures_signature
    structures = pd.read_csv(structures_path)
    acronym_rows = {str(row.acronym).lower(): row for row in structures.itertuples(index=False)}
    region_ids = {}
    for region in region_names:
        row = acronym_rows.get(str(region).lower())
        if row is None:
            continue
        token = f"/{int(row.id)}/"
        region_ids[region] = structures.loc[
            structures["structure_id_path"].astype(str).str.contains(token, regex=False), "id"
        ].astype(int).to_numpy()
    def png_url(image):
        output = BytesIO()
        image.save(output, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    atlas = Image.open(annotation_path)
    frame_indices = np.linspace(40, atlas.n_frames - 41, 29, dtype=int)
    slices = []
    for frame_index in frame_indices:
        atlas.seek(int(frame_index))
        labels = np.asarray(atlas).astype(np.int64)
        inside = labels > 0
        edges = inside & (
            (labels != np.roll(labels, 1, axis=0)) | (labels != np.roll(labels, 1, axis=1))
        )
        rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
        rgba[..., :3] = [17, 27, 42]
        rgba[..., 3] = np.where(inside, 218, 0)
        rgba[edges, :3] = [91, 111, 133]
        rgba[edges, 3] = 150
        base_image = Image.fromarray(rgba, "RGBA").resize((274, 192), Image.Resampling.NEAREST)
        masks = {}
        for region, ids in region_ids.items():
            mask = np.isin(labels, ids)
            if not mask.any():
                continue
            color = brain_region_color(region).lstrip("#")
            rgb = tuple(int(color[offset:offset + 2], 16) for offset in (0, 2, 4))
            region_rgba = np.zeros((*labels.shape, 4), dtype=np.uint8)
            region_rgba[..., :3] = rgb
            region_rgba[..., 3] = np.where(mask, 235, 0)
            masks[region] = png_url(
                Image.fromarray(region_rgba, "RGBA").resize((274, 192), Image.Resampling.NEAREST)
            )
        slices.append({"index": int(frame_index), "base": png_url(base_image), "masks": masks})
    atlas.close()
    default_slice = int(np.argmin(np.abs(frame_indices - float(default_ap_frame))))
    return {
        "slices": slices, "plane": "coronal", "resolutionUm": 25,
        "defaultSlice": default_slice,
    }


def fit_straight_probe(coordinates):
    """Fit a 3D line by total least squares and retain the observed extent."""
    center = coordinates.mean(axis=0)
    _, _, axes = np.linalg.svd(coordinates - center, full_matrices=False)
    direction = axes[0]
    projections = (coordinates - center) @ direction
    fitted = np.vstack(
        [
            center + projections.min() * direction,
            center + projections.max() * direction,
        ]
    )
    if np.linalg.norm(fitted[0] - coordinates[0]) > np.linalg.norm(fitted[1] - coordinates[0]):
        fitted = fitted[::-1]
    residuals = np.linalg.norm(
        coordinates - (center + np.outer(projections, direction)),
        axis=1,
    )
    return fitted, float(np.sqrt(np.mean(residuals**2)))


SESSION_PALETTE = [
    "#2563EB",  # blue
    "#F05A3C",  # vermilion
    "#00A878",  # emerald
    "#C13CE2",  # magenta
    "#00B8D9",  # cyan
    "#E23D6F",  # raspberry
    "#6D4AFF",  # violet
    "#78C800",  # lime
    "#FF7A1A",  # orange
    "#007F8B",  # deep teal
    "#D72D2D",  # red
    "#3F51B5",  # indigo
    "#00C16A",  # green
    "#F238A0",  # pink
    "#1687E8",  # sky blue
    "#8E35B5",  # purple
    "#D58B00",  # amber
    "#00A6A6",  # turquoise
    "#B64B63",  # muted crimson
    "#5746A8",  # deep violet
]

SESSION_CARD_PALETTE = [
    "#3299B5", "#C66B55", "#765FB1", "#429773",
    "#BA5683", "#4D78BB", "#7F9E43", "#A45CB5",
]

BRAIN_REGION_PALETTE = [
    "#D17A3F", "#4C8DAE", "#9A6BC4", "#5E9B76", "#C05F78",
    "#8A7B45", "#5577B7", "#A65E9A", "#668E8A", "#B56D54",
    "#7E8DB5", "#B07C9A", "#739A86", "#B58A5C", "#6F91A1",
    "#9B7A62", "#7D6FA8", "#A56F72",
]

BRAIN_REGION_COLORS = {
    "mos2/3": "#4C8DAE",
    "mos5": "#9A6BC4",
    "mos6a": "#5E9B76",
    "mop6a": "#D17A3F",
    "cp": "#5577B7",
    "orbvl5": "#C05F78",
    "orbvl2/3": "#668E8A",
    "orbvl6a": "#B56D54",
    "orbm6b": "#A65E9A",
    "pl6a": "#8A7B45",
    "frp6a": "#7E8DB5",
    "aon": "#B07C9A",
    "dp": "#739A86",
    "fa": "#B58A5C",
}


def brain_region_color(region):
    """Stable color for one brain-area acronym across every ephys view."""
    key = str(region).strip().casefold()
    if key in BRAIN_REGION_COLORS:
        return BRAIN_REGION_COLORS[key]
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest()
    index = int.from_bytes(digest, "big")
    return BRAIN_REGION_PALETTE[index % len(BRAIN_REGION_PALETTE)]


def session_color(session):
    return SESSION_PALETTE[pd.Timestamp(session).toordinal() % len(SESSION_PALETTE)]


def session_card_color(session):
    return SESSION_CARD_PALETTE[pd.Timestamp(session).toordinal() % len(SESSION_CARD_PALETTE)]


def color_with_alpha(color, alpha):
    value = color.lstrip("#")
    if len(value) == 6:
        red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
        return f"rgba({red}, {green}, {blue}, {alpha})"
    return color


def clip_line_to_brain(fitted, annotation_image, frame_cache, resolution_um=25.0):
    # The tip is the deepest endpoint. Extend only toward the dorsal entry,
    # then retain the annotated interval containing (or nearest to) the tip.
    tip_index = int(np.argmax(fitted[:, 1]))
    tip = fitted[tip_index]
    shaft_end = fitted[1 - tip_index]
    direction = shaft_end - tip
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        return fitted, shaft_end
    direction /= norm
    distances = np.linspace(0, norm + 5000, 501)
    samples = tip + distances[:, None] * direction
    indices = np.rint(samples / resolution_um).astype(int)
    shape = (annotation_image.n_frames, annotation_image.height, annotation_image.width)
    in_bounds = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    inside = np.zeros(len(samples), dtype=bool)
    for ap_index in np.unique(indices[in_bounds, 0]):
        ap_index = int(ap_index)
        if ap_index not in frame_cache:
            annotation_image.seek(ap_index)
            frame_cache[ap_index] = np.asarray(annotation_image).copy()
        rows = np.flatnonzero(in_bounds & (indices[:, 0] == ap_index))
        inside[rows] = frame_cache[ap_index][indices[rows, 1], indices[rows, 2]] > 0

    inside_indices = np.flatnonzero(inside)
    if len(inside_indices) < 2:
        return fitted, fitted[1]
    breaks = np.flatnonzero(np.diff(inside_indices) > 1) + 1
    runs = np.split(inside_indices, breaks)
    containing = [run for run in runs if run[0] == 0]
    selected_run = containing[0] if containing else min(runs, key=lambda run: run[0])
    clipped = np.vstack([samples[selected_run[0]], samples[selected_run[-1]]])
    entry = clipped[1]
    return clipped, entry


def build_probe_figure(
    tracks,
    brain_mesh,
    region_meshes=None,
    selected_session=None,
    selected_probe=None,
    annotation_path=None,
    height=520,
):
    figure = go.Figure()
    vertices, faces = brain_mesh
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 2],
            y=vertices[:, 0],
            z=vertices[:, 1],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name="Allen mouse brain",
            color="#9AA9BD",
            opacity=0.10,
            flatshading=False,
            hoverinfo="skip",
            showlegend=False,
            lighting={
                "ambient": 0.72,
                "diffuse": 0.62,
                "fresnel": 0.18,
                "roughness": 0.72,
                "specular": 0.22,
            },
            lightposition={"x": 9000, "y": -12000, "z": 6000},
        )
    )

    region_styles = {
        "MOs": {"color": "#B3A06F", "opacity": 0.22},
        "STR": {"color": "#B3A06F", "opacity": 0.22},
        "CP": {"color": "#C3A85F", "opacity": 0.27},
    }
    for region, mesh in (region_meshes or {}).items():
        region_vertices, region_faces = mesh
        style = region_styles.get(region, {"color": "#8B5CF6", "opacity": 0.28})
        figure.add_trace(
            go.Mesh3d(
                x=region_vertices[:, 2],
                y=region_vertices[:, 0],
                z=region_vertices[:, 1],
                i=region_faces[:, 0],
                j=region_faces[:, 1],
                k=region_faces[:, 2],
                name=region,
                color=style["color"],
                opacity=style["opacity"],
                flatshading=False,
                hovertemplate=f"<b>{region}</b><extra></extra>",
                showlegend=False,
                lighting={"ambient": 0.70, "diffuse": 0.65, "roughness": 0.70},
            )
        )

    sessions = sorted({track["date"] for track in tracks})
    session_colors = {session: session_color(session) for session in sessions}
    annotation_image = Image.open(annotation_path) if annotation_path else None
    frame_cache = {}
    for track in sorted(tracks, key=lambda item: (item["date"], item["probe"])):
        fitted, fit_rmse = fit_straight_probe(track["coordinates"])
        if annotation_image is not None:
            fitted, entry = clip_line_to_brain(fitted, annotation_image, frame_cache)
        else:
            entry = fitted[np.argmin(fitted[:, 1])]
        session = track["date"]
        probe = track["probe"]
        color = session_colors[session]
        is_selected = session == selected_session
        is_active_probe = is_selected and probe == selected_probe

        figure.add_trace(
            go.Scatter3d(
                x=fitted[:, 2],
                y=fitted[:, 0],
                z=fitted[:, 1],
                mode="lines",
                legendgroup=session,
                showlegend=False,
                hoverinfo="skip",
                opacity=0.72 if is_selected else 0.14,
                line={"color": color, "width": 10},
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[entry[2]],
                y=[entry[0]],
                z=[entry[1]],
                mode="markers+text",
                legendgroup=session,
                showlegend=False,
                marker={
                    "color": color,
                    "size": 12 if is_selected else 3.5,
                    "symbol": "cross",
                    "opacity": 1.0 if is_selected else 0.62,
                    "line": {"color": "#FFFFFF", "width": 3 if is_selected else 1},
                },
                text=[probe if is_active_probe else ""],
                textposition="top center",
                textfont={"color": "#FFFFFF", "size": 13, "family": "Arial Black"},
                hovertemplate=f"<b>{session} · {probe}</b><br>Brain entry<extra></extra>",
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=fitted[:, 2],
                y=fitted[:, 0],
                z=fitted[:, 1],
                mode="lines",
                name=f"{session} · {probe}",
                legendgroup=session,
                legendgrouptitle_text=session,
                showlegend=False,
                line={
                    "color": color,
                    "width": 5,
                },
                opacity=1.0 if is_selected else 0.58,
                hovertemplate=(
                    f"<b>{session} · {probe}</b><br>"
                    f"Linear fit RMS: {fit_rmse:.1f} µm"
                    "<extra></extra>"
                ),
            )
        )

    if annotation_image is not None:
        annotation_image.close()

    figure.update_layout(
        height=height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="#111B2A",
        plot_bgcolor="#111B2A",
        uirevision="allen-probe-scene-profile-v2",
        showlegend=False,
        scene={
            "aspectmode": "data",
            "bgcolor": "#111B2A",
            "xaxis": {"visible": False, "showbackground": False},
            "yaxis": {"visible": False, "showbackground": False},
            "zaxis": {"visible": False, "showbackground": False, "autorange": "reversed"},
            "camera": {
                "eye": {"x": 1.20, "y": 0.0, "z": 0.14},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "dragmode": "turntable",
        },
    )
    return figure


QC_COLORS = {
    "good": "#22C55E",
    "mua": "#F59E0B",
    "noise": "#EF4444",
    "non_soma_good": "#14B8A6",
    "non_soma_mua": "#8B5CF6",
    "unlabelled": "#94A3B8",
}


def style_detail_figure(figure, height=330):
    figure.update_layout(
        height=height,
        margin={"l": 16, "r": 16, "t": 20, "b": 16},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": NAVY, "size": 12},
        showlegend=False,
        hoverlabel={"bgcolor": NAVY, "font_color": "white"},
    )
    figure.update_xaxes(gridcolor="#E2E8F0", zeroline=False)
    figure.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return figure


def build_regions_figure(units):
    selected = units[units["bombcell_label"] == "good"]
    counts = selected["brain_region"].value_counts().head(12).sort_values()
    figure = go.Figure(
        go.Bar(
            x=counts.values,
            y=counts.index,
            orientation="h",
            marker={"color": "#2563EB", "cornerradius": 5},
            hovertemplate="%{y}<br>%{x} good units<extra></extra>",
        )
    )
    figure.update_xaxes(title="Good units")
    return style_detail_figure(figure)


def build_depth_figure(units):
    figure = go.Figure()
    for label, group in units.dropna(subset=["depth_um", "firing_rate"]).groupby("bombcell_label"):
        figure.add_trace(
            go.Scatter(
                x=group["firing_rate"],
                y=group["depth_um"],
                mode="markers",
                name=label,
                marker={"color": QC_COLORS.get(label, "#94A3B8"), "size": 7, "opacity": 0.72},
                customdata=group[["unit_id", "brain_region"]],
                hovertemplate=(
                    "Unit %{customdata[0]} · %{customdata[1]}<br>"
                    "%{x:.2f} spikes/s · %{y:.0f} µm<extra></extra>"
                ),
            )
        )
    figure.update_layout(showlegend=True, legend={"orientation": "h", "y": 1.08, "title": None})
    figure.update_xaxes(title="Firing rate (spikes/s)")
    figure.update_yaxes(title="Axial position (µm)")
    styled = style_detail_figure(figure)
    styled.update_layout(showlegend=True)
    return styled


def build_probe_profile_figure(channels, units):
    channel_view = channels.dropna(subset=["depth_um"]).copy()
    if channel_view.empty:
        return None
    channel_view["brain_region"] = channel_view["brain_region"].fillna("Unknown").astype(str)
    channel_view = (
        channel_view.sort_values("depth_um")
        .groupby("depth_um", as_index=False)
        .agg(brain_region=("brain_region", lambda values: values.mode().iat[0]))
    )
    channel_view["segment"] = channel_view["brain_region"].ne(
        channel_view["brain_region"].shift()
    ).cumsum()
    spacing = float(channel_view["depth_um"].diff().median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 20.0

    segments = (
        channel_view.groupby(["segment", "brain_region"], as_index=False)
        .agg(depth_min=("depth_um", "min"), depth_max=("depth_um", "max"))
    )
    segments["depth_min"] -= spacing / 2
    segments["depth_max"] += spacing / 2

    good_units = units[units["bombcell_label"] == "good"].copy()
    good_counts = good_units["brain_region"].value_counts()
    region_span_mm = (
        segments.assign(span_um=segments["depth_max"] - segments["depth_min"])
        .groupby("brain_region")["span_um"]
        .sum()
        / 1000
    )
    density = good_counts.div(region_span_mm).replace([np.inf, -np.inf], np.nan).fillna(0)
    max_density = max(float(density.max()) if len(density) else 0.0, 1.0)
    palette = ["#8EA8B8", "#A79ABC", "#85AA9D", "#B7A276", "#A98B86", "#8798B2"]

    figure = go.Figure()
    for segment in segments.itertuples(index=False):
        region = segment.brain_region
        midpoint = (segment.depth_min + segment.depth_max) / 2
        span = segment.depth_max - segment.depth_min
        color = palette[sum(map(ord, region)) % len(palette)]
        region_density = float(density.get(region, 0.0))
        region_count = int(good_counts.get(region, 0))
        figure.add_shape(
            type="rect",
            x0=-0.10,
            x1=0.10,
            y0=segment.depth_min,
            y1=segment.depth_max,
            fillcolor=color,
            line={"color": PAGE_BG, "width": 1},
        )
        if span >= 35:
            figure.add_annotation(
                x=-0.16,
                y=midpoint,
                text=f"<b>{region}</b>",
                showarrow=False,
                xanchor="right",
                font={"size": 12, "color": NAVY},
            )
        if region_count:
            bar_end = 0.28 + 0.78 * region_density / max_density
            figure.add_shape(
                type="rect",
                x0=0.28,
                x1=bar_end,
                y0=segment.depth_min + span * 0.22,
                y1=segment.depth_max - span * 0.22,
                fillcolor="#4B9B7B",
                opacity=0.82,
                line={"width": 0},
            )
            if span >= 45:
                figure.add_annotation(
                    x=bar_end + 0.03,
                    y=midpoint,
                    text=f"{region_count} · {region_density:.1f}/mm",
                    showarrow=False,
                    xanchor="left",
                    font={"size": 11, "color": MUTED},
                )

    figure.add_annotation(
        x=-0.10,
        y=1.04,
        xref="x",
        yref="paper",
        text="REGIONS",
        showarrow=False,
        xanchor="center",
        font={"size": 11, "color": MUTED},
    )
    figure.add_annotation(
        x=0.28,
        y=1.04,
        xref="x",
        yref="paper",
        text="GOOD UNITS · DENSITY",
        showarrow=False,
        xanchor="left",
        font={"size": 11, "color": "#3A8069"},
    )
    figure.update_xaxes(visible=False, range=[-0.58, 1.42], fixedrange=True)
    figure.update_yaxes(
        title="Depth along probe (µm)",
        autorange="reversed",
        showgrid=False,
        zeroline=False,
    )
    styled = style_detail_figure(figure, height=590)
    styled.update_layout(margin={"l": 36, "r": 28, "t": 42, "b": 26})
    return styled


def build_probe_profile_figure_v2(channels, units):
    channel_view = channels.dropna(subset=["depth_um"]).copy()
    if channel_view.empty:
        return None
    channel_view["brain_region"] = channel_view["brain_region"].fillna("Unknown").astype(str)
    channel_view = (
        channel_view.sort_values("depth_um")
        .groupby("depth_um", as_index=False)
        .agg(brain_region=("brain_region", lambda values: values.mode().iat[0]))
    )
    channel_view["segment"] = channel_view["brain_region"].ne(
        channel_view["brain_region"].shift()
    ).cumsum()
    spacing = float(channel_view["depth_um"].diff().median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 20.0
    segments = (
        channel_view.groupby(["segment", "brain_region"], as_index=False)
        .agg(depth_min=("depth_um", "min"), depth_max=("depth_um", "max"))
    )
    segments["depth_min"] -= spacing / 2
    segments["depth_max"] += spacing / 2
    segments["midpoint"] = (segments["depth_min"] + segments["depth_max"]) / 2
    segments["span_mm"] = (segments["depth_max"] - segments["depth_min"]) / 1000

    good_counts = units.loc[
        units["bombcell_label"] == "good", "brain_region"
    ].value_counts()
    total_span = segments.groupby("brain_region")["span_mm"].sum()
    density = good_counts.div(total_span).replace([np.inf, -np.inf], np.nan).fillna(0)
    segments["good_units"] = segments["brain_region"].map(good_counts).fillna(0).astype(int)
    segments["density"] = segments["brain_region"].map(density).fillna(0.0)

    palette = ["#8EA8B8", "#A79ABC", "#85AA9D", "#B7A276", "#A98B86", "#8798B2"]
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[0.38, 0.62],
        horizontal_spacing=0.08,
        subplot_titles=("Probe regions", "Good-unit density"),
    )
    for segment in segments.itertuples(index=False):
        color = palette[sum(map(ord, segment.brain_region)) % len(palette)]
        figure.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[segment.depth_min, segment.depth_max],
                mode="lines",
                line={"color": color, "width": 28},
                text=[segment.brain_region, segment.brain_region],
                customdata=[[segment.brain_region], [segment.brain_region]],
                hovertemplate="<b>%{customdata[0]}</b><br>%{y:.0f} µm<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=0.13,
            y=segment.midpoint,
            xref="x",
            yref="y",
            text=f"<b>{segment.brain_region}</b>",
            showarrow=False,
            xanchor="left",
            font={"size": 12, "color": NAVY},
        )

    figure.add_trace(
        go.Bar(
            x=segments["density"],
            y=segments["midpoint"],
            width=(segments["depth_max"] - segments["depth_min"]) * 0.68,
            orientation="h",
            marker={"color": "#4B9B7B", "cornerradius": 5},
            customdata=np.column_stack(
                [segments["brain_region"], segments["good_units"]]
            ),
            text=[
                f"{count} good · {value:.1f}/mm" if count else ""
                for count, value in zip(segments["good_units"], segments["density"])
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>%{customdata[1]} good units"
                "<br>%{x:.1f} units/mm<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(visible=False, range=[-0.18, 0.8], row=1, col=1)
    max_density = max(float(segments["density"].max()), 1.0)
    figure.update_xaxes(
        title="Good units / mm",
        range=[0, max_density * 1.42],
        showgrid=False,
        row=1,
        col=2,
    )
    figure.update_yaxes(
        title="Depth (µm)",
        autorange="reversed",
        showgrid=False,
        row=1,
        col=1,
    )
    styled = style_detail_figure(figure, height=570)
    styled.update_layout(margin={"l": 42, "r": 80, "t": 58, "b": 34})
    return styled


def _build_probe_anatomy_figure_legacy(channels, units):
    channel_view = channels.dropna(subset=["depth_um"]).copy()
    if channel_view.empty:
        return None
    channel_view["brain_region"] = channel_view["brain_region"].fillna("Unknown").astype(str)
    channel_view = (
        channel_view.sort_values("depth_um")
        .groupby("depth_um", as_index=False)
        .agg(brain_region=("brain_region", lambda values: values.mode().iat[0]))
    )
    channel_view["segment"] = channel_view["brain_region"].ne(
        channel_view["brain_region"].shift()
    ).cumsum()
    spacing = float(channel_view["depth_um"].diff().median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 20.0
    segments = (
        channel_view.groupby(["segment", "brain_region"], as_index=False)
        .agg(depth_min=("depth_um", "min"), depth_max=("depth_um", "max"))
    )
    segments["depth_min"] -= spacing / 2
    segments["depth_max"] += spacing / 2

    good_units = units[
        (units["bombcell_label"] == "good") & units["depth_um"].notna()
    ]
    good_counts = good_units["brain_region"].value_counts()
    depth_min = float(segments["depth_min"].min())
    depth_max = float(segments["depth_max"].max())
    density_edges = np.linspace(depth_min, depth_max, 121)
    density_counts, _ = np.histogram(good_units["depth_um"], bins=density_edges)
    bin_width_mm = np.diff(density_edges).mean() / 1000
    smoothed_density = gaussian_filter1d(density_counts.astype(float), sigma=1.15) / bin_width_mm
    density_centers = (density_edges[:-1] + density_edges[1:]) / 2
    good_rates = pd.to_numeric(good_units["firing_rate"], errors="coerce").to_numpy(dtype=float)
    good_depths = good_units["depth_um"].to_numpy(dtype=float)
    rate_sums, _ = np.histogram(good_depths, bins=density_edges, weights=np.nan_to_num(good_rates))
    rate_counts, _ = np.histogram(good_depths[np.isfinite(good_rates)], bins=density_edges)
    binned_rates = np.divide(
        rate_sums, rate_counts, out=np.full_like(rate_sums, np.nan), where=rate_counts > 0
    )
    if np.isfinite(binned_rates).any():
        smooth_rate_sums = gaussian_filter1d(rate_sums, sigma=1.0)
        smooth_rate_counts = gaussian_filter1d(rate_counts.astype(float), sigma=1.0)
        smoothed_rates = np.divide(
            smooth_rate_sums,
            smooth_rate_counts,
            out=np.zeros_like(smooth_rate_sums),
            where=smooth_rate_counts > 0.08,
        )
    else:
        smoothed_rates = np.zeros_like(density_centers)
    good_amplitudes = pd.to_numeric(good_units["amplitude_median"], errors="coerce").to_numpy(dtype=float)
    amplitude_sums, _ = np.histogram(
        good_depths, bins=density_edges, weights=np.nan_to_num(np.abs(good_amplitudes))
    )
    amplitude_counts, _ = np.histogram(
        good_depths[np.isfinite(good_amplitudes)], bins=density_edges
    )
    smooth_amplitude_sums = gaussian_filter1d(amplitude_sums, sigma=1.0)
    smooth_amplitude_counts = gaussian_filter1d(amplitude_counts.astype(float), sigma=1.0)
    smoothed_amplitudes = np.divide(
        smooth_amplitude_sums,
        smooth_amplitude_counts,
        out=np.zeros_like(smooth_amplitude_sums),
        where=smooth_amplitude_counts > 0.08,
    )

    figure = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.34, 0.20, 0.20, 0.26],
        vertical_spacing=0.06,
    )
    figure.add_trace(
        go.Scatter(
            x=density_centers,
            y=smoothed_density,
            mode="lines",
            line={"color": "#76518F", "width": 3, "shape": "spline"},
            fill="tozeroy",
            fillcolor="rgba(118,81,143,0.15)",
            hovertemplate="%{x:.0f} µm from tip<br>%{y:.1f} good units/mm<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=density_centers,
            y=smoothed_rates,
            mode="lines",
            line={"color": "#4B91B5", "width": 2.5, "shape": "spline"},
            fill="tozeroy",
            fillcolor="rgba(75,145,181,0.12)",
            hovertemplate="%{x:.0f} µm from tip<br>%{y:.2f} Hz / good unit<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=density_centers,
            y=smoothed_amplitudes,
            mode="lines",
            line={"color": "#C48752", "width": 2.5, "shape": "spline"},
            fill="tozeroy",
            fillcolor="rgba(196,135,82,0.12)",
            hovertemplate="%{x:.0f} µm from tip<br>%{y:.2f} median amplitude (µV)<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    palette = ["#8EA8B8", "#A79ABC", "#85AA9D", "#B7A276", "#A98B86", "#8798B2"]
    for segment in segments.itertuples(index=False):
        region = segment.brain_region
        count = int(good_counts.get(region, 0))
        color = palette[sum(map(ord, region)) % len(palette)]
        midpoint = (segment.depth_min + segment.depth_max) / 2
        figure.add_trace(
            go.Scatter(
                x=[segment.depth_min, segment.depth_max],
                y=[0, 0],
                mode="lines",
                line={"color": color, "width": 34},
                hoverinfo="skip",
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        if segment.depth_max - segment.depth_min >= 150:
            figure.add_trace(
                go.Scatter(
                    x=[midpoint],
                    y=[0],
                    mode="text",
                    text=[f"<b>{region}</b>"],
                    textfont={"size": 10, "color": "#172033"},
                    customdata=[[region, count]],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>%{customdata[1]} good units"
                        "<br>%{x:.0f} µm from tip<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=4,
                col=1,
            )

    figure.update_yaxes(title=None, rangemode="tozero", row=1, col=1)
    figure.update_yaxes(title=None, rangemode="tozero", row=2, col=1)
    figure.update_yaxes(title="Amplitude (µV)", rangemode="tozero", row=3, col=1)
    figure.update_yaxes(visible=False, range=[-0.5, 0.5], fixedrange=True, row=4, col=1)
    figure.update_yaxes(title=None, rangemode="tozero", row=3, col=1)
    shared_depth_range = [depth_max + spacing, max(0, depth_min - spacing)]
    for row in (1, 2, 3):
        figure.update_xaxes(range=shared_depth_range, matches="x4", showgrid=False, row=row, col=1)
    for row, label in ((1, "Good units/mm"), (2, "Firing rate (Hz)"), (3, "Amplitude (µV)")):
        domain = getattr(figure.layout, f"yaxis{'' if row == 1 else row}").domain
        figure.add_annotation(
            x=-0.105, y=(domain[0] + domain[1]) / 2, xref="paper", yref="paper",
            text=label, textangle=-90, showarrow=False,
            font={"size": 11, "color": "#4B5563"},
        )
    figure.update_xaxes(
        title="Distance from tip (µm) · tip = 0",
        range=[depth_max + spacing, max(0, depth_min - spacing)],
        showgrid=False,
        row=4,
        col=1,
    )
    styled = style_detail_figure(figure, height=590)
    styled.update_layout(margin={"l": 88, "r": 28, "t": 20, "b": 48})
    return styled


def _build_probe_anatomy_figure_profile_legacy(channels, units):
    """Vertical probe profile, with the probe tip at the bottom."""
    channel_view = channels.dropna(subset=["depth_um"]).copy()
    if channel_view.empty:
        return None
    channel_view["brain_region"] = channel_view["brain_region"].fillna("Unknown").astype(str)
    channel_view = (
        channel_view.sort_values("depth_um")
        .groupby("depth_um", as_index=False)
        .agg(brain_region=("brain_region", lambda values: values.mode().iat[0]))
    )
    channel_view["segment"] = channel_view["brain_region"].ne(
        channel_view["brain_region"].shift()
    ).cumsum()
    spacing = float(channel_view["depth_um"].diff().median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 20.0
    segments = (
        channel_view.groupby(["segment", "brain_region"], as_index=False)
        .agg(depth_min=("depth_um", "min"), depth_max=("depth_um", "max"))
    )
    segments["depth_min"] -= spacing / 2
    segments["depth_max"] += spacing / 2

    good_units = units[(units["bombcell_label"] == "good") & units["depth_um"].notna()].copy()
    good_counts = good_units["brain_region"].astype(str).value_counts()
    depth_min = float(segments["depth_min"].min())
    depth_max = float(segments["depth_max"].max())
    edges = np.linspace(depth_min, depth_max, 121)
    centers = (edges[:-1] + edges[1:]) / 2
    depths = good_units["depth_um"].to_numpy(dtype=float)
    density_counts, _ = np.histogram(depths, bins=edges)
    density = gaussian_filter1d(density_counts.astype(float), sigma=1.15) / (np.diff(edges).mean() / 1000)

    def smoothed_good_unit_metric(column):
        values = pd.to_numeric(good_units[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        sums, _ = np.histogram(depths[finite], bins=edges, weights=np.abs(values[finite]))
        counts, _ = np.histogram(depths[finite], bins=edges)
        smooth_sums = gaussian_filter1d(sums.astype(float), sigma=1.0)
        smooth_counts = gaussian_filter1d(counts.astype(float), sigma=1.0)
        result = np.divide(
            smooth_sums, smooth_counts, out=np.zeros_like(smooth_sums), where=smooth_counts > 0.08
        )
        result[counts == 0] = 0
        return result

    rates = smoothed_good_unit_metric("firing_rate")
    amplitudes = smoothed_good_unit_metric("amplitude_median")
    figure = make_subplots(
        rows=1, cols=4, shared_yaxes=True,
        column_widths=[0.22, 0.26, 0.26, 0.26], horizontal_spacing=0.075,
    )
    palette = ["#8EA8B8", "#A79ABC", "#85AA9D", "#B7A276", "#A98B86", "#8798B2"]
    for segment in segments.itertuples(index=False):
        region = str(segment.brain_region)
        count = int(good_counts.get(region, 0))
        midpoint = (segment.depth_min + segment.depth_max) / 2
        color = palette[sum(map(ord, region)) % len(palette)]
        figure.add_trace(
            go.Scatter(
                x=[0, 0], y=[segment.depth_min, segment.depth_max], mode="lines",
                line={"color": color, "width": 42}, hoverinfo="skip", showlegend=False,
            ), row=1, col=1,
        )
        if segment.depth_max - segment.depth_min >= 150:
            figure.add_trace(
                go.Scatter(
                    x=[0], y=[midpoint], mode="text+markers", text=[f"<b>{region}</b>"],
                    textfont={"size": 10, "color": "#172033"},
                    marker={"size": 22, "color": "rgba(0,0,0,0)"},
                    customdata=[[region, count, midpoint]],
                    hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]} good units"
                                   "<br>%{customdata[2]:.0f} µm from tip<extra></extra>"),
                    showlegend=False,
                ), row=1, col=1,
            )

    curves = [
        (density, "#76518F", "rgba(118,81,143,0.15)", "Good units / mm", "%{x:.1f} good units/mm"),
        (rates, "#4B91B5", "rgba(75,145,181,0.12)", "Firing rate (Hz)", "%{x:.2f} Hz / good unit"),
        (amplitudes, "#C48752", "rgba(196,135,82,0.12)", "Amplitude (µV)", "%{x:.2f} µV"),
    ]
    for column, (values, color, fill, title, hover) in enumerate(curves, start=2):
        figure.add_trace(
            go.Scatter(
                x=values, y=centers, mode="lines",
                line={"color": color, "width": 2.7, "shape": "spline"},
                fill="tozerox", fillcolor=fill,
                hovertemplate=hover + "<br>%{y:.0f} µm from tip<extra></extra>",
                showlegend=False,
            ), row=1, col=column,
        )
        figure.update_xaxes(
            title=title, rangemode="tozero", showgrid=True, zeroline=False, row=1, col=column
        )

    depth_range = [max(0, depth_min - spacing), depth_max + spacing]
    figure.update_xaxes(
        title="Regions", range=[-0.7, 0.7], showticklabels=False, showgrid=False, row=1, col=1
    )
    figure.update_yaxes(
        title="Distance from probe tip (µm)", range=depth_range,
        showgrid=False, tickformat=".0f", row=1, col=1,
    )
    for column in range(2, 5):
        figure.update_yaxes(range=depth_range, showgrid=False, row=1, col=column)

    # Keep a little breathing room between the global metrics and the profile blocks.
    for column in range(1, 5):
        figure.update_yaxes(domain=[0.0, 0.91], row=1, col=column)
    figure.add_annotation(
        x=0, y=depth_range[0], xref="x", yref="y", text="TIP",
        showarrow=True, arrowhead=0, ay=22,
        font={"size": 10, "color": "#4B5563"}, arrowcolor="#4B5563",
    )
    styled = style_detail_figure(figure, height=650)
    styled.update_layout(margin={"l": 78, "r": 28, "t": 24, "b": 72})
    return styled


def _build_probe_anatomy_region_bars(channels, units):
    """Region-level good-unit summaries displayed as compact rectangular bars."""
    channel_view = channels.dropna(subset=["depth_um"]).copy()
    good_units = units[
        (units["bombcell_label"] == "good")
        & units["depth_um"].notna()
        & units["brain_region"].notna()
    ].copy()
    good_units["brain_region"] = good_units["brain_region"].astype(str)
    good_units = good_units[good_units["brain_region"].str.lower() != "unknown"]
    if channel_view.empty or good_units.empty:
        return None

    channel_view["brain_region"] = channel_view["brain_region"].fillna("Unknown").astype(str)
    channel_view = channel_view.sort_values("depth_um")
    # Remove tiny A-B-A islands from the channel annotation. These are usually
    # boundary artefacts and otherwise split one anatomical area into odd slivers.
    region_values = channel_view["brain_region"].to_numpy(copy=True)
    for _ in range(4):
        run_starts = np.r_[0, np.flatnonzero(region_values[1:] != region_values[:-1]) + 1]
        run_stops = np.r_[run_starts[1:], len(region_values)]
        run_labels = region_values[run_starts]
        changed = False
        for run_index in range(1, len(run_starts) - 1):
            if (
                run_labels[run_index - 1] == run_labels[run_index + 1]
                and run_stops[run_index] - run_starts[run_index] <= 5
            ):
                region_values[run_starts[run_index]:run_stops[run_index]] = run_labels[run_index - 1]
                changed = True
        if not changed:
            break
    channel_view["brain_region"] = region_values
    channel_view["segment"] = channel_view["brain_region"].ne(
        channel_view["brain_region"].shift()
    ).cumsum()
    spacing = float(channel_view["depth_um"].diff().median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 20.0
    segments = (
        channel_view.groupby(["segment", "brain_region"], as_index=False)
        .agg(depth_min=("depth_um", "min"), depth_max=("depth_um", "max"))
    )
    segments["span_um"] = segments["depth_max"] - segments["depth_min"] + spacing
    region_lengths = segments.groupby("brain_region")["span_um"].sum() / 1000

    good_units["firing_rate"] = pd.to_numeric(good_units["firing_rate"], errors="coerce")
    good_units["amplitude_abs"] = pd.to_numeric(
        good_units["amplitude_median"], errors="coerce"
    ).abs()
    summary = (
        good_units.groupby("brain_region", as_index=False)
        .agg(
            good_units=("unit_id", "count"),
            median_depth=("depth_um", "median"),
            firing_rate=("firing_rate", "median"),
            amplitude=("amplitude_abs", "median"),
        )
    )
    summary = summary[summary["good_units"] >= 10].copy()
    if summary.empty:
        return None
    summary["density"] = summary["good_units"] / summary["brain_region"].map(region_lengths)
    summary = summary.replace([np.inf, -np.inf], np.nan).sort_values("median_depth")
    region_order = summary["brain_region"].tolist()

    figure = make_subplots(
        rows=1, cols=3, shared_yaxes=True,
        column_widths=[0.34, 0.33, 0.33], horizontal_spacing=0.075,
    )
    specs = [
        ("density", "Good units / mm", "#80639A", ".1f"),
        ("firing_rate", "Median firing rate (Hz)", "#4B91B5", ".2f"),
        ("amplitude", "Median amplitude (µV)", "#C48752", ".1f"),
    ]
    customdata = np.column_stack([summary["good_units"], summary["median_depth"]])
    for column, (metric, title, color, value_format) in enumerate(specs, start=1):
        figure.add_trace(
            go.Bar(
                x=summary[metric], y=summary["brain_region"], orientation="h",
                marker={"color": color, "cornerradius": 5, "line": {"width": 0}},
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>" + title + ": %{x:" + value_format + "}"
                    "<br>%{customdata[0]:.0f} good units"
                    "<br>Median depth: %{customdata[1]:.0f} µm<extra></extra>"
                ),
                showlegend=False,
            ), row=1, col=column,
        )
        figure.update_xaxes(title=title, rangemode="tozero", showgrid=False, row=1, col=column)
        figure.update_yaxes(
            categoryorder="array", categoryarray=region_order,
            showgrid=False, title=None, row=1, col=column,
        )
    styled = style_detail_figure(figure, height=max(310, 54 * len(summary) + 105))
    styled.update_layout(margin={"l": 76, "r": 24, "t": 18, "b": 58}, bargap=0.28)
    return styled


def build_probe_anatomy_figure(channels, units):
    """Probe regions at left and aligned region-level metric blocks at right."""
    channel_view = channels.dropna(subset=["depth_um"]).copy()
    if channel_view.empty:
        return None
    channel_view["brain_region"] = channel_view["brain_region"].fillna("Unknown").astype(str)
    channel_view = channel_view.sort_values("depth_um")
    channel_view["segment"] = channel_view["brain_region"].ne(
        channel_view["brain_region"].shift()
    ).cumsum()
    spacing = float(channel_view["depth_um"].diff().median())
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 20.0
    segments = (
        channel_view.groupby(["segment", "brain_region"], as_index=False)
        .agg(depth_min=("depth_um", "min"), depth_max=("depth_um", "max"))
    )
    segments["depth_min"] -= spacing / 2
    segments["depth_max"] += spacing / 2
    segments["span_um"] = segments["depth_max"] - segments["depth_min"]

    good = units[
        (units["bombcell_label"] == "good")
        & units["depth_um"].notna()
        & units["brain_region"].notna()
    ].copy()
    good["brain_region"] = good["brain_region"].astype(str)
    good["firing_rate"] = pd.to_numeric(good["firing_rate"], errors="coerce")
    good["amplitude_abs"] = pd.to_numeric(good["amplitude_median"], errors="coerce").abs()
    region_span = segments.groupby("brain_region")["span_um"].sum() / 1000
    summary = good.groupby("brain_region", as_index=False).agg(
        good_units=("unit_id", "count"), median_depth=("depth_um", "median"),
        firing_rate=("firing_rate", "median"), amplitude=("amplitude_abs", "median"),
    )
    summary = summary[summary["good_units"] >= 10].copy()
    if summary.empty:
        return None
    summary["density"] = summary["good_units"] / summary["brain_region"].map(region_span)
    # A region can reappear in several tiny channel fragments. Keep its metric on
    # the largest contiguous fragment so the three profiles stay readable.
    primary_segment_indices = segments.groupby("brain_region")["span_um"].idxmax()
    primary_segments = segments.loc[primary_segment_indices].copy()
    metric_blocks = primary_segments.merge(summary, on="brain_region", how="inner")
    metric_blocks["mid_depth"] = (metric_blocks["depth_min"] + metric_blocks["depth_max"]) / 2
    metric_blocks["block_height"] = (metric_blocks["span_um"] * .82).clip(lower=spacing * 1.25)

    figure = make_subplots(
        rows=1, cols=4, shared_yaxes=True,
        column_widths=[0.20, 0.27, 0.27, 0.26], horizontal_spacing=0.075,
    )
    region_colors = {
        region: brain_region_color(region)
        for region in segments["brain_region"].drop_duplicates()
    }
    counts = good["brain_region"].value_counts()
    for segment in segments.itertuples(index=False):
        region = str(segment.brain_region)
        region_count = int(counts.get(region, 0))
        color = region_colors[region] if region_count >= 10 else "#C5CDD5"
        figure.add_trace(go.Bar(
            x=[1], y=[(segment.depth_min + segment.depth_max) / 2], orientation="h",
            width=[segment.span_um * .94],
            marker={"color": color, "cornerradius": 7, "line": {"color": PANEL_BG, "width": 1}},
            customdata=[[region, region_count]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]} good units<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        if segment.span_um >= 230:
            figure.add_annotation(
                x=.5, y=(segment.depth_min + segment.depth_max) / 2,
                text=f"<b>{region}</b>", showarrow=False,
                font={"size": 9, "color": "#172033"}, row=1, col=1,
            )

    specs = [
        ("density", "Good units / mm", "#80639A", ".1f"),
        ("firing_rate", "Median firing rate (Hz)", "#4B91B5", ".2f"),
        ("amplitude", "Median amplitude (µV)", "#C48752", ".1f"),
    ]
    customdata = np.column_stack([metric_blocks["brain_region"], metric_blocks["good_units"]])
    for column, (metric, title, color, fmt) in enumerate(specs, start=2):
        figure.add_trace(go.Bar(
            x=metric_blocks[metric], y=metric_blocks["mid_depth"], orientation="h",
            width=metric_blocks["block_height"],
            marker={"color": color, "opacity": 1.0, "cornerradius": 7,
                    "line": {"color": PANEL_BG, "width": 1}},
            customdata=customdata,
            hovertemplate=("<b>%{customdata[0]}</b><br>" + title + ": %{x:" + fmt + "}"
                           "<br>%{customdata[1]} good units<extra></extra>"),
            showlegend=False,
        ), row=1, col=column)
        figure.update_xaxes(
            title=title, rangemode="tozero", showgrid=True,
            gridcolor="rgba(94,112,132,.22)", gridwidth=1,
            zeroline=False,
            showline=True, linecolor="#657487", linewidth=1.6,
            row=1, col=column,
        )

    depth_range = [float(segments["depth_min"].min()), float(segments["depth_max"].max())]
    figure.update_xaxes(range=[0, 1], visible=False, row=1, col=1)
    figure.update_yaxes(
        title="Distance from probe tip (µm)", range=depth_range, showgrid=False, row=1, col=1
    )
    for column in range(2, 5):
        figure.update_yaxes(
            range=depth_range, showgrid=False, showline=True,
            linecolor="#657487", linewidth=1.6, ticks="outside",
            row=1, col=column,
        )
    for column in range(1, 5):
        figure.update_yaxes(domain=[0.0, 0.86], row=1, col=column)

    probe_length_mm = max(float(segments["span_um"].sum()) / 1000, 1e-9)
    global_values = [
        ("Good units / mm", len(good) / probe_length_mm, "#80639A", ".1f"),
        ("Median firing rate", good["firing_rate"].median(), "#4B91B5", ".2f"),
        ("Median amplitude", good["amplitude_abs"].median(), "#C48752", ".1f"),
    ]
    for column, (label, value, color, number_format) in enumerate(global_values, start=2):
        value_text = format(value, number_format) if np.isfinite(value) else "—"
        unit = " Hz" if label == "Median firing rate" else (" µV" if label == "Median amplitude" else "")
        figure.add_annotation(
            x=.5, y=1.15, xref=f"x{column} domain", yref="paper",
            text=(f"<span style='color:{color};font-size:16px'><b>{label}</b></span>"
                  f"<br><br><span style='color:#263548;font-size:18px;font-weight:400'>{value_text}{unit}</span>"),
            showarrow=False, align="center", bgcolor="rgba(0,0,0,0)",
            borderwidth=0, borderpad=0,
        )
    styled = style_detail_figure(figure, height=676)
    styled.update_layout(margin={"l": 78, "r": 28, "t": 112, "b": 72}, bargap=0)
    return styled


def build_activity_heatmap(activity_heatmap, bout_start_times=None, height=520):
    values = activity_heatmap["relative_activity"]
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.89, 0.11],
        vertical_spacing=0.008,
    )
    figure.add_trace(
        go.Heatmap(
            x=activity_heatmap["time_min"],
            y=np.arange(values.shape[0]),
            z=values,
            colorscale=[
                [0.0, "#111B2A"],
                [0.06, "#111B2A"],
                [0.25, "#3B0F70"],
                [0.52, "#8C2981"],
                [0.76, "#DE4968"],
                [0.90, "#FE9F6D"],
                [1.0, "#FCFDBF"],
            ],
            zmin=0,
            zmax=1,
            colorbar={
                "title": "activity",
                "thickness": 11,
                "outlinewidth": 0,
                "len": 0.80,
                "y": 0.59,
            },
            zsmooth=False,
            hovertemplate="Channel %{y:.0f}<br>%{x:.1f} min · activity %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    bout_times = np.asarray(bout_start_times if bout_start_times is not None else [], dtype=float)
    bout_times = bout_times[np.isfinite(bout_times)] / 60
    figure.add_trace(
        go.Bar(
            x=bout_times,
            y=np.full(len(bout_times), 0.22),
            base=0.68,
            width=0.009,
            marker={"color": "#C94A4A", "line": {"width": 0}},
            name="bout start",
            customdata=np.arange(1, len(bout_times) + 1),
            hovertemplate="Bout %{customdata}<br>%{x:.2f} min<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    figure.add_annotation(
        x=0.5,
        y=0.002,
        xref="paper",
        yref="paper",
        text="Bout start",
        showarrow=False,
        xanchor="center",
        font={"size": 12, "color": MUTED},
    )
    figure.update_xaxes(
        showticklabels=False, showgrid=False, fixedrange=True, row=1, col=1
    )
    figure.update_xaxes(
        title={"text": "Recording time (min)", "font": {"size": 12}, "standoff": 14},
        showgrid=False, fixedrange=True, row=2, col=1
    )
    figure.update_yaxes(
        title="Channel",
        tickmode="array",
        tickvals=[0, 64, 128, 192, 256, 320, 383],
        ticktext=["0", "64", "128", "192", "256", "320", "384"],
        range=[-0.5, 383.5],
        showticklabels=True,
        showgrid=False,
        fixedrange=True,
        row=1,
        col=1,
    )
    figure.update_yaxes(
        title="Bouts",
        visible=False,
        range=[0, 1],
        fixedrange=True,
        row=2,
        col=1,
    )
    styled = style_detail_figure(figure, height=height)
    styled.update_layout(
        dragmode=False,
        hovermode="closest",
        uirevision="ephys-activity-viewer",
        margin={"l": 58, "r": 62, "t": 18, "b": 58},
        showlegend=False,
    )
    return styled


def build_good_unit_raster(raster, bout_start_times=None):
    figure = go.Figure()
    figure.add_trace(
        go.Scattergl(
            x=raster["time_min"],
            y=raster["unit_row"],
            mode="markers",
            marker={"color": "#E7EEF5", "size": 1.5, "opacity": 0.62},
            hovertemplate="Good unit row %{y:.0f}<br>%{x:.2f} min<extra></extra>",
            showlegend=False,
        )
    )
    bout_times = np.asarray(bout_start_times if bout_start_times is not None else [], dtype=float)
    bout_times = bout_times[np.isfinite(bout_times)] / 60
    if len(bout_times):
        line_x = np.column_stack(
            [bout_times, bout_times, np.full(len(bout_times), np.nan)]
        ).ravel()
        line_y = np.tile([-0.5, raster["unit_count"] - 0.5, np.nan], len(bout_times))
        figure.add_trace(
            go.Scattergl(
                x=line_x,
                y=line_y,
                mode="lines",
                line={"color": "rgba(201,74,74,0.16)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.update_layout(
        height=390,
        margin={"l": 55, "r": 24, "t": 10, "b": 44},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#09111D",
        font={"color": NAVY, "size": 12},
        showlegend=False,
        hoverlabel={"bgcolor": NAVY, "font_color": "white"},
    )
    figure.update_xaxes(title="Recording time (min)", showgrid=False, fixedrange=True)
    figure.update_yaxes(
        title=f"{raster['unit_count']} good units",
        range=[-0.5, raster["unit_count"] - 0.5],
        showgrid=False,
        showticklabels=False,
        fixedrange=True,
    )
    return figure


def build_peri_bout_figure(peri_bout, height=300):
    time_s = peri_bout["time_s"]
    baseline_mask = (time_s >= -4) & (time_s <= -1)

    def to_zscore(values):
        baseline = values[baseline_mask]
        center = np.nanmean(baseline)
        spread = np.nanstd(baseline)
        return (values - center) / (spread if spread > 1e-9 else 1.0), spread

    mean_z, _ = to_zscore(np.asarray(peri_bout["mean_hz"], dtype=float))
    figure = go.Figure()
    for region, region_rate in peri_bout.get("regions", {}).items():
        region_z, _ = to_zscore(region_rate)
        region_z = gaussian_filter1d(region_z, sigma=1.0)
        figure.add_trace(
            go.Scatter(
                x=time_s,
                y=region_z,
                mode="lines",
                name=region,
                line={"color": brain_region_color(region), "width": 1.8},
                opacity=0.88,
                hovertemplate=(
                    f"<b>{region}</b><br>%{{x:.1f}} s · %{{y:.2f}} Hz / good unit"
                    "<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=time_s,
            y=mean_z,
            mode="lines",
            line={"color": "#172033", "width": 2.8, "dash": "dash"},
            hovertemplate="%{x:.1f} s · %{y:.2f} Hz / good unit<extra></extra>",
            showlegend=True,
            name="Global",
            visible=False,
        )
    )
    figure.add_vline(x=0, line={"color": "#C94A4A", "width": 2, "dash": "dot"})
    figure.add_annotation(
        x=0,
        y=1,
        xref="x",
        yref="paper",
        text="Arrival on site",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "#A84747"},
    )
    figure.update_xaxes(title="Time from arrival on site (s)", range=[-4, 8], fixedrange=True)
    figure.update_yaxes(title="Firing-rate z-score · baseline −4 to −1 s", fixedrange=True)
    styled = style_detail_figure(figure, height=height)
    styled.update_layout(
        margin={"l": 62, "r": 138, "t": 30, "b": 46},
        legend={
            "orientation": "v",
            "yanchor": "middle",
            "y": 0.5,
            "xanchor": "left",
            "x": 1.02,
            "title": None,
            "font": {"size": 11},
        },
        showlegend=True,
    )
    return styled


def build_lick_svm_figure(svm_view):
    figure = go.Figure()
    groups = [
        ("Non-rewarded lick", svm_view["nonrewarded"], "#E8899C"),
        ("Rewarded lick", svm_view["rewarded"], "#75C8A6"),
        ("Last lick", svm_view["last"], "#7EB7DE"),
    ]
    for name, points, color in groups:
        figure.add_trace(
            go.Scattergl(
                x=points[:, 0], y=points[:, 1], mode="markers", name=name,
                marker={"color": color, "size": 6, "opacity": 1.0},
                hovertemplate=f"<b>{name}</b><extra></extra>",
            )
        )
    all_points = np.vstack([points for _, points, _ in groups])
    x_min, x_max = np.percentile(all_points[:, 0], [1, 99])
    y_min, y_max = np.percentile(all_points[:, 1], [1, 99])
    x_padding, y_padding = 0.08 * (x_max - x_min), 0.08 * (y_max - y_min)
    figure.add_hline(
        y=svm_view["boundary"], line={"color": "#172033", "width": 2, "dash": "dash"}
    )
    figure.add_hline(
        y=svm_view["boundary"] - svm_view["margin"],
        line={"color": "rgba(75,85,99,0.40)", "width": 1, "dash": "dot"},
    )
    figure.add_hline(
        y=svm_view["boundary"] + svm_view["margin"],
        line={"color": "rgba(75,85,99,0.40)", "width": 1, "dash": "dot"},
    )
    figure.update_xaxes(
        title="Population-state axis", range=[x_min - x_padding, x_max + x_padding], fixedrange=True
    )
    figure.update_yaxes(
        title="SVM decision axis", range=[y_min - y_padding, y_max + y_padding], fixedrange=True
    )
    styled = style_detail_figure(figure, height=430)
    styled.update_layout(
        margin={"l": 62, "r": 24, "t": 20, "b": 62},
        legend={"orientation": "h", "y": -0.20, "x": 0, "font": {"size": 11}},
        plot_bgcolor="#172033",
        font={"color": "#E7EEF5", "size": 12},
    )
    styled.update_xaxes(gridcolor="rgba(226,232,240,0.16)")
    styled.update_yaxes(
        gridcolor="rgba(226,232,240,0.16)", scaleanchor="x", scaleratio=1
    )
    return styled


def build_qc_figure(units):
    grouped_labels = units["bombcell_label"].astype(str).replace(
        {"non_soma_good": "non_soma", "non_soma_mua": "non_soma"}
    )
    counts = grouped_labels.value_counts()
    good_count = int((units["bombcell_label"].astype(str) == "good").sum())
    colors = {
        "good": "#4B9B7B",
        "mua": "#C29A4A",
        "noise": "#B97070",
        "non_soma": "#8175AD",
        "unlabelled": "#9AA5B1",
    }
    figure = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.62,
            sort=False,
            marker={
                "colors": [colors.get(label, "#94A3B8") for label in counts.index],
                "line": {"color": PANEL_BG, "width": 3},
            },
            textinfo="label+percent",
            textposition="outside",
            automargin=True,
            hovertemplate="<b>%{label}</b><br>%{value} units · %{percent}<extra></extra>",
        )
    )
    for text, y, color, size in [
        (str(int(counts.sum())), .655, NAVY, 23),
        ("units", .585, NAVY, 10),
        (str(good_count), .415, "#4B9B7B", 23),
        ("good", .345, "#4B9B7B", 10),
    ]:
        figure.add_annotation(
            text=f"<b>{text}</b>" if size > 10 else text,
            x=.5, y=y, showarrow=False,
            font={"color": color, "size": size},
        )
    styled = style_detail_figure(figure, height=390)
    styled.update_layout(
        margin={"l": 70, "r": 70, "t": 34, "b": 76},
        uniformtext={"minsize": 11, "mode": "show"},
    )
    return styled


def build_snr_figure(units):
    values = pd.to_numeric(units.get("snr"), errors="coerce").dropna()
    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=30,
            marker={"color": "#0EA5E9"},
            hovertemplate="SNR %{x:.1f}<br>%{y} units<extra></extra>",
        )
    )
    figure.update_xaxes(title="SNR")
    figure.update_yaxes(title="Units")
    return style_detail_figure(figure)


def build_qc_scatter_figure(units):
    """Compact IBL-style view of recording stability versus refractory violations."""
    view = units.copy()
    view["presence_ratio"] = pd.to_numeric(view["presence_ratio"], errors="coerce")
    view["rp_contamination"] = pd.to_numeric(view["rp_contamination"], errors="coerce")
    view["snr"] = pd.to_numeric(view["snr"], errors="coerce")
    view["firing_rate"] = pd.to_numeric(view["firing_rate"], errors="coerce")
    view["qc_group"] = view["bombcell_label"].astype(str).replace(
        {"non_soma_good": "non_soma", "non_soma_mua": "non_soma"}
    )
    view = view.dropna(subset=["presence_ratio", "rp_contamination"])
    colors = {
        "good": "#4B9B7B", "mua": "#C29A4A", "noise": "#B97070",
        "non_soma": "#8175AD", "unlabelled": "#9AA5B1",
    }
    figure = go.Figure()
    for label, group in view.groupby("qc_group", sort=False):
        figure.add_trace(
            go.Scattergl(
                x=group["presence_ratio"], y=group["rp_contamination"],
                mode="markers", name=label,
                marker={"size": 7, "color": colors.get(label, "#94A3B8"), "opacity": 0.78},
                customdata=np.column_stack([group["unit_id"], group["snr"], group["firing_rate"]]),
                hovertemplate=(
                    "<b>Unit %{customdata[0]:.0f}</b><br>Presence ratio: %{x:.2f}"
                    "<br>RP contamination: %{y:.3f}<br>SNR: %{customdata[1]:.2f}"
                    "<br>Firing rate: %{customdata[2]:.2f} Hz<extra></extra>"
                ),
            )
        )
    figure.add_vline(x=0.9, line={"color": "rgba(75,85,99,.35)", "width": 1, "dash": "dot"})
    figure.add_hline(y=0.1, line={"color": "rgba(75,85,99,.35)", "width": 1, "dash": "dot"})
    figure.update_xaxes(title="Presence ratio", range=[0, 1.02])
    figure.update_yaxes(title="RP contamination", rangemode="tozero")
    styled = style_detail_figure(figure, height=390)
    styled.update_layout(
        margin={"l": 60, "r": 18, "t": 20, "b": 54}, showlegend=False,
    )
    return styled


DEFAULT_BOMBCELL_MUA_THRESHOLDS = {
    "amplitude_median": {"greater": 30.0, "less": None, "abs": True},
    "snr": {"greater": 5.0, "less": None},
    "amplitude_cutoff": {"greater": None, "less": 0.20},
    "num_spikes": {"greater": 300.0, "less": None},
    "rp_contamination": {"greater": None, "less": 0.10},
    "presence_ratio": {"greater": 0.50, "less": None},
    "drift_ptp": {"greater": None, "less": 100.0},
}

BOMBCELL_CRITERION_LABELS = {
    "amplitude_median": ("Amplitude median", "µV"),
    "snr": ("SNR", ""),
    "amplitude_cutoff": ("Amplitude cutoff", ""),
    "num_spikes": ("Spike count", ""),
    "rp_contamination": ("RP contamination", ""),
    "presence_ratio": ("Presence ratio", ""),
    "drift_ptp": ("Drift PTP", "µm"),
}


@st.cache_data(show_spinner=False, max_entries=4)
def load_bombcell_mua_thresholds(config_path: str, signature=None):
    """Read the active MUA threshold update from the ephys pipeline config."""
    del signature
    path = Path(config_path)
    if not path.is_file():
        return DEFAULT_BOMBCELL_MUA_THRESHOLDS.copy(), "built-in fallback"
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "update":
                continue
            target = ast.get_source_segment(source, node.func.value) or ""
            if "BOMBCELL_THRESHOLDS" not in target or "mua" not in target.lower():
                continue
            candidate = ast.literal_eval(node.args[0])
            if isinstance(candidate, dict):
                active = {
                    str(name): bounds for name, bounds in candidate.items()
                    if isinstance(bounds, dict)
                    and (bounds.get("greater") is not None or bounds.get("less") is not None)
                }
                if active:
                    return active, str(path)
    except (OSError, SyntaxError, ValueError):
        pass
    return DEFAULT_BOMBCELL_MUA_THRESHOLDS.copy(), "built-in fallback"


def format_bombcell_threshold(bounds, unit=""):
    lower, upper = bounds.get("greater"), bounds.get("less")
    prefix = "abs " if bounds.get("abs") else ""
    suffix = f" {unit}" if unit else ""
    if lower is not None and upper is not None:
        return f"{prefix}≥ {lower:g} and ≤ {upper:g}{suffix}"
    if lower is not None:
        return f"{prefix}≥ {lower:g}{suffix}"
    return f"{prefix}≤ {upper:g}{suffix}"


def build_qc_selectivity_figure(units):
    """Show independent rejection rates using the active ephys Bombcell config."""
    config_signature = source_signature(BOMBCELL_CONFIG) if BOMBCELL_CONFIG.is_file() else None
    thresholds, _ = load_bombcell_mua_thresholds(str(BOMBCELL_CONFIG), config_signature)
    eligible_units = units[units["bombcell_label"].astype(str) != "noise"]
    rows = []
    for column, bounds in thresholds.items():
        if column not in eligible_units.columns:
            continue
        label, unit = BOMBCELL_CRITERION_LABELS.get(
            column, (column.replace("_", " ").title(), "")
        )
        values = pd.to_numeric(eligible_units[column], errors="coerce").dropna()
        if values.empty:
            continue
        tested_values = values.abs() if bounds.get("abs") else values
        passed = pd.Series(True, index=values.index)
        if bounds.get("greater") is not None:
            passed &= tested_values >= float(bounds["greater"])
        if bounds.get("less") is not None:
            passed &= tested_values <= float(bounds["less"])
        passed_count = int(passed.sum())
        rows.append({
            "criterion": label, "passed": passed_count, "tested": int(len(values)),
            "passed_pct": 100 * passed_count / len(values),
            "threshold": format_bombcell_threshold(bounds, unit),
        })
    if not rows:
        return None
    view = pd.DataFrame(rows).sort_values(
        ["passed_pct", "criterion"], ascending=[False, True]
    ).reset_index(drop=True)
    colors = ["#7289B0"] * len(view)
    figure = go.Figure(go.Bar(
        x=view["passed_pct"], y=view["criterion"], orientation="h",
        marker={"color": colors, "cornerradius": 6, "line": {"width": 0}},
        text=[f"{value:.0f}%" for value in view["passed_pct"]],
        textposition="inside", insidetextanchor="end",
        textfont={"color": "#FFFFFF", "size": 11},
        customdata=np.column_stack([view["passed"], view["tested"], view["threshold"]]),
        hovertemplate=("<b>%{y}</b><br>%{customdata[0]} / %{customdata[1]} units pass"
                       "<br>Pass threshold: %{customdata[2]}<extra></extra>"),
    ))
    figure.update_xaxes(title="Units passing independently (%)", range=[0, 105], showgrid=False)
    figure.update_yaxes(
        title=None, showgrid=False, categoryorder="array",
        categoryarray=view["criterion"].tolist(), autorange="reversed",
    )
    styled = style_detail_figure(figure, height=390)
    styled.update_layout(margin={"l": 120, "r": 42, "t": 20, "b": 54})
    return styled


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def quiet_metric(label, value, accent="#D9E2EC"):
    st.markdown(
        f"""
        <div class="quiet-metric" style="--metric-accent: {accent};">
            <div class="quiet-metric-label">{label}</div>
            <div class="quiet-metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def set_ephys_session(state_key, session_date):
    st.session_state[state_key] = session_date


def open_mouse_view(mouse_id, view):
    st.session_state["global_mouse"] = mouse_id
    st.session_state[NAVIGATION_STATE_KEY] = view
    st.session_state["_scroll_to_top_requested"] = True


def set_global_view(view):
    st.session_state[NAVIGATION_STATE_KEY] = view
    st.session_state["_scroll_to_top_requested"] = True


def sync_navigation_mouse(widget_key):
    selected = st.session_state.get(widget_key)
    if selected is not None:
        st.session_state["global_mouse"] = selected


def is_visible_mouse(mouse_id):
    return str(mouse_id).strip().upper() not in EXCLUDED_MOUSE_IDS


def home_session_pie(protocol_counts, ephys_count, openfield_count=0, show_legend=False):
    training_1 = int(protocol_counts.get(1, 0))
    training_2 = int(protocol_counts.get(2, 0))
    task_total = int(protocol_counts.get(3, 0))
    flexifliping = int(protocol_counts.get(4, 0))
    slices = [
        ("Training 1", training_1, "#D2A72C", "Training 1"),
        ("Training 2", training_2, "#428CB3", "Training 2"),
        ("Task", task_total, "#CF725E", "Task"),
        ("Flexifliping", flexifliping, BEHAVIOR_PROTOCOL_COLORS[4], "Flexifliping"),
    ]
    total = sum(value for _, value, _, _ in slices)
    if total:
        cursor = 0.0
        gradient_parts = []
        for _, value, color, _ in slices:
            if value <= 0:
                continue
            next_cursor = cursor + value / total * 100
            gradient_parts.append(f"{color} {cursor:.2f}% {next_cursor:.2f}%")
            cursor = next_cursor
        gradient = "conic-gradient(" + ", ".join(gradient_parts) + ")"
    else:
        gradient = "#D7E1EA"
    tooltip = (
        f"Training 1: {training_1} · Training 2: {training_2} · "
        f"Task: {task_total} · Flexifliping: {flexifliping}"
    )
    legend_values = {"Training 1": training_1, "Training 2": training_2, "Task": task_total, "Flexifliping": flexifliping}
    legend = "".join(
        f'<span><i style="background:{color}"></i>{short} {legend_values[short]}</span>'
        for _, _, color, short in slices
    )
    wrapper_class = "home-global-pie" if show_legend else ""
    legend_html = f'<div class="home-pie-key">{legend}</div>' if show_legend else ""
    modality_items = []
    if int(ephys_count) > 0:
        modality_items.append(
            f'<div class="home-modality-count ephys" title="Ephys sessions">'
            f'<i><svg viewBox="0 0 48 48" fill="none"><path d="M7 25h8l4-12 7 23 5-16 3 5h7" stroke="currentColor" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg></i>'
            f'<b>{int(ephys_count)}</b></div>'
        )
    if int(openfield_count) > 0:
        modality_items.append(
            f'<div class="home-modality-count openfield" title="Openfield sessions">'
            f'<i><svg viewBox="0 0 48 48" fill="none"><rect x="7" y="7" width="34" height="34" rx="7" stroke="currentColor" stroke-width="3"/><path d="M13 24s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z" stroke="currentColor" stroke-width="3"/><circle cx="24" cy="24" r="4" fill="currentColor"/></svg></i>'
            f'<b>{int(openfield_count)}</b></div>'
        )
    modality_html = (
        '<div class="home-modality-counts">' + "".join(modality_items) + "</div>"
        if modality_items else ""
    )
    count_html = "" if show_legend else f'<span class="home-session-pie-count">{total}</span>'
    return (
        f'<div class="home-pie-composite {wrapper_class}">{modality_html}'
        f'<div title="{html.escape(tooltip)}">'
        f'<div class="home-session-pie" style="background:{gradient};">{count_html}</div>'
        f'{legend_html}</div></div>'
    )


def home_global_summary(mouse_count, openfield_count, ephys_count, protocol_counts):
    values = [
        ("Training 1", int(protocol_counts.get(1, 0)), "#D2A72C"),
        ("Training 2", int(protocol_counts.get(2, 0)), "#428CB3"),
        ("Task", int(protocol_counts.get(3, 0)), "#CF725E"),
        ("Flexifliping", int(protocol_counts.get(4, 0)), BEHAVIOR_PROTOCOL_COLORS[4]),
    ]
    total = sum(value for _, value, _ in values)
    if total:
        cursor, parts = 0.0, []
        for _, value, color in values:
            if value <= 0:
                continue
            end = cursor + value / total * 100
            parts.append(f"{color} {cursor:.2f}% {end:.2f}%")
            cursor = end
        gradient = "conic-gradient(" + ", ".join(parts) + ")"
    else:
        gradient = "#D7E1EA"
    legend = "".join(
        f'<span><i style="background:{color}"></i><b>{value}</b> {label}</span>'
        for label, value, color in values
    )
    openfield_icon = '<svg viewBox="0 0 48 48" fill="none"><rect x="7" y="7" width="34" height="34" rx="7" stroke="currentColor" stroke-width="3"/><path d="M13 24s4-7 11-7 11 7 11 7-4 7-11 7-11-7-11-7Z" stroke="currentColor" stroke-width="3"/><circle cx="24" cy="24" r="4" fill="currentColor"/></svg>'
    ephys_icon = '<svg viewBox="0 0 48 48" fill="none"><path d="M8 25h7l4-12 7 23 5-16 3 5h6" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    return (
        '<div class="home-summary-band">'
        f'<div class="home-summary-block"><div class="home-summary-mice"><strong>{int(mouse_count)}</strong><span>Mice</span></div></div>'
        f'<div class="home-summary-block"><div class="home-summary-pie" style="background:{gradient}"></div><div class="home-summary-legend">{legend}</div></div>'
        f'<div class="home-summary-block"><div class="home-summary-icon ephys">{ephys_icon}</div><div class="home-summary-value"><strong>{int(ephys_count)}</strong>Ephys</div></div>'
        f'<div class="home-summary-block"><div class="home-summary-icon openfield">{openfield_icon}</div><div class="home-summary-value"><strong>{int(openfield_count)}</strong>Openfield</div></div>'
        '</div>'
    )


def behavior_mouse_summary(mouse_id, mouse_df):
    protocol_counts = (
        pd.to_numeric(mouse_df.get("Protocol"), errors="coerce")
        .dropna().astype(int).value_counts().reindex([1, 2, 3, 4], fill_value=0)
    )
    values = [
        ("Training 1", int(protocol_counts.get(1, 0)), BEHAVIOR_PROTOCOL_COLORS[1]),
        ("Training 2", int(protocol_counts.get(2, 0)), BEHAVIOR_PROTOCOL_COLORS[2]),
        ("Task", int(protocol_counts.get(3, 0)), BEHAVIOR_PROTOCOL_COLORS[3]),
        ("Flexifliping", int(protocol_counts.get(4, 0)), BEHAVIOR_PROTOCOL_COLORS[4]),
    ]
    total = sum(value for _, value, _ in values)
    cursor, parts = 0.0, []
    for _, value, color in values:
        if value <= 0 or total <= 0:
            continue
        end = cursor + value / total * 100
        parts.append(f"{color} {cursor:.2f}% {end:.2f}%")
        cursor = end
    gradient = "conic-gradient(" + ", ".join(parts) + ")" if parts else "#D7E1EA"
    legend = "".join(
        f'<span><i style="background:{color}"></i><b>{value}</b> {label}</span>'
        for label, value, color in values
    )
    latest = mouse_df["Date"].max()
    latest_label = latest.strftime("%Y-%m-%d") if pd.notna(latest) else "-"
    return (
        '<div class="behavior-mouse-band">'
        f'<div class="behavior-mouse-block"><div class="behavior-mouse-primary"><strong>{html.escape(str(mouse_id))}</strong><span>MOUSE</span></div></div>'
        f'<div class="behavior-mouse-block"><div class="behavior-mouse-primary"><strong>{len(mouse_df)}</strong><span>SESSIONS</span></div></div>'
        f'<div class="behavior-mouse-block"><div class="behavior-mouse-pie" style="background:{gradient}"></div><div class="behavior-mouse-legend">{legend}</div></div>'
        f'<div class="behavior-mouse-block"><div class="behavior-mouse-primary"><strong>{latest_label}</strong><span>LATEST SESSION</span></div></div>'
        '</div>'
    )


def sync_ephys_probe(shared_key, widget_key):
    st.session_state[shared_key] = st.session_state[widget_key]


def render_probe_selector(section_name, probe_options, shared_key):
    if len(probe_options) < 2:
        return
    current = st.session_state.get(shared_key, probe_options[0])
    widget_key = f"{shared_key}_{re.sub(r'[^a-z0-9]+', '_', section_name.lower())}"
    st.session_state[widget_key] = current
    st.segmented_control(
        "Probe",
        probe_options,
        label_visibility="collapsed",
        width="content",
        key=widget_key,
        on_change=sync_ephys_probe,
        args=(shared_key, widget_key),
    )


def bout_simple_card(valid_bouts, total_bouts):
    valid = int(valid_bouts) if pd.notna(valid_bouts) else 0
    total = int(total_bouts) if pd.notna(total_bouts) else 0
    metric_card("Valid Bouts", f"{valid} / {total}")


def section(title, accent=NAVY):
    st.markdown(
        f"""
        <div class="section-block" style="--section-accent: {accent};">
            <hr class="section-separator">
            <div class="section-title">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def section_card(title):
    section_key = "ephys_section_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    with st.container(border=True, key=section_key):
        st.markdown(f'<div class="section-card-title">{title}</div>', unsafe_allow_html=True)
        yield
    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)


@contextmanager
def behavior_section_card(title):
    section_key = "behavior_section_" + re.sub(
        r"[^a-z0-9]+", "_", title.lower()
    ).strip("_")
    with st.container(border=True, key=section_key):
        st.markdown(f'<div class="section-card-title">{title}</div>', unsafe_allow_html=True)
        yield
    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)


def behavior_plot_card(title, figure):
    st.markdown(
        f'<div class="subplot-title behavior-subtitle">{title}</div>',
        unsafe_allow_html=True,
    )
    plot_card(figure)


def plot_card(image):
    with st.container(border=False):
        if isinstance(image, go.Figure):
            st.plotly_chart(
                image,
                width="stretch",
                config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False},
            )
        elif image:
            st.image(image, width="stretch")
        else:
            st.caption("Plot not available.")


def dataframe_card(dataframe):
    with st.container(border=False):
        st.dataframe(dataframe, width="stretch", hide_index=True)


def page_title(title):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


AGENT_DOMAIN_KNOWLEDGE = {
    "project": {
        "name": "INVIBE",
        "scope": (
            "Longitudinal mouse experiments combining a site-based licking task, "
            "open-field assays and electrophysiology. A mouse can have behavior-only, "
            "Openfield and/or Ephys sessions; absence from one dataset is not missingness "
            "in another unless the analysis explicitly requires paired observations."
        ),
    },
    "behavior_task": {
        "session_identity": ["Mouse_ID", "Date", "Version"],
        "protocols": {
            "1": "Training 1",
            "2": "Training 2",
            "3": "Task",
            "4": "Flexifliping",
        },
        "probability": (
            "Probas is the recorded condition and Proba_val is its normalized numeric form. "
            "Labels 90/30, 90/10 and 50/50 are conditions within Task (Protocol 3), not protocols."
        ),
        "events": {
            "arrival": "Entry into a reward-site zone; it is not itself a lick.",
            "site_deactivation": (
                "A change in site availability. Site state and licking are independent: "
                "licks may still occur after deactivation."
            ),
            "rewarded_lick": "A lick classified as rewarded by the acquisition pipeline.",
            "non_rewarded_lick": "A registered lick without reward in the applicable task state.",
            "invalid_lick": "A registered lick rejected by the task logic for that state/window.",
            "bout": (
                "A pipeline-defined lick bout. 'Correct Bouts' is the authoritative boolean "
                "mask; valid-bout count is its number of True values."
            ),
        },
        "derived_metrics": {
            "valid_bouts": "Count of True entries in Correct Bouts.",
            "total_bouts": "Number of detected bouts before validity filtering.",
            "rewarded_licks": "Count of Times Rewarded Licks.",
            "non_rewarded_licks": "Count of Times Non Rewarded Licks.",
            "invalid_licks": "Count of Times Invalid Licks.",
            "rewarded_licks_per_min": (
                "Rewarded-lick count divided by Session Dur in minutes; do not compare raw "
                "counts across unequal durations when a rate was requested."
            ),
        },
    },
    "openfield": {
        "assays": {
            "OFT": "Open Field Test; commonly interpreted with center/periphery, locomotion, rearing and grooming metrics.",
            "LDT": "Light-Dark Test; commonly interpreted with compartment occupancy, transitions and locomotion metrics.",
            "SIT": (
                "Social Interaction Test. Alone and Visitor recordings from the same mouse/date "
                "form one logical SIT session, not two sessions."
            ),
        },
        "tables": {
            "oft_sessions": "One row per OFT session when metrics are available.",
            "ldt_sessions": "One row per LDT session when metrics are available.",
            "paired_behavior_metrics": "Precomputed paired/cross-assay metrics where available.",
            "mouse_metrics": "Per-mouse summary metrics where available.",
        },
        "interpretation": (
            "Inspect the live schema before using an unfamiliar metric. Ratio fields are proportions "
            "or percentages according to the stored column; report the returned unit and never infer "
            "a missing SIT metric from video filenames."
        ),
    },
    "electrophysiology": {
        "session_identity": ["mouse", "date", "probe"],
        "unit_scope": (
            "A unit belongs to an insertion/probe and can be assigned a peak channel, anatomical "
            "depth, brain region and Bombcell label."
        ),
        "metrics": {
            "unit_count": "Number of units after the requested filters.",
            "num_spikes": "Number of detected spikes assigned to a unit.",
            "firing_rate": "Mean firing rate of a unit in Hz.",
            "presence_ratio": "Fraction of recording intervals in which the unit is present.",
            "snr": "Signal-to-noise ratio.",
            "rp_contamination": "Estimated refractory-period contamination.",
            "amplitude_cutoff": "Estimate of the fraction of spikes missing below the detection threshold.",
            "amplitude_median": "Median spike amplitude; preserve the unit returned by the tool.",
            "drift_ptp": "Peak-to-peak estimate of unit drift over the recording.",
        },
        "quality_rules": (
            "Bombcell labels and active thresholds are authoritative for unit quality. The dashboard's "
            "minimum of 10 good units per region is a visualization/inclusion rule, not an additional "
            "Bombcell classification criterion."
        ),
    },
    "joining_and_statistics": {
        "keys": (
            "Join datasets by normalized mouse ID. Add normalized date only for explicitly matched-session "
            "analyses. If dates do not match, aggregate within each mouse before a cross-dataset comparison."
        ),
        "repeated_measures": (
            "Sessions from the same mouse are repeated observations, not independent animals. Always report "
            "both the number of mice and sessions when available."
        ),
        "interpretation": (
            "Separate descriptive association from causality; state filters, aggregation, missingness and "
            "the analysis unit (unit, probe, session or mouse)."
        ),
    },
}


CHATBOT_TOOLS = [
    {
        "type": "function", "name": "get_domain_knowledge",
        "description": (
            "Return the authoritative INVIBE task ontology, protocol/event definitions, dataset "
            "relationships, metric meanings and interpretation rules. Call this before answering "
            "questions about experimental semantics or how datasets should be joined."
        ),
        "parameters": {
            "type": "object", "properties": {}, "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "get_data_catalog",
        "description": "List available mice and session counts across behavior, openfield and electrophysiology.",
        "parameters": {
            "type": "object", "properties": {
                "mouse": {"type": ["string", "null"], "description": "Optional mouse ID."},
            }, "required": ["mouse"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "get_behavior_timeseries",
        "description": (
            "Read a behavior metric across sessions and create a Plotly chart. "
            "Protocol mapping is exact: Training 1 = protocol 1, Training 2 = protocol 2, "
            "Task = protocol 3, and Flexifliping = protocol 4. For Task 90/30 use protocols=[3] and task_probability='90/30'. "
            "Use plot_type='density' for a pooled histogram/KDE."
        ),
        "parameters": {
            "type": "object", "properties": {
                "mice": {"type": "array", "items": {"type": "string"}, "description": "Empty means all mice."},
                "metric": {"type": "string", "enum": [
                    "valid_bouts", "total_bouts", "rewarded_licks",
                    "non_rewarded_licks", "invalid_licks",
                ]},
                "plot_type": {"type": "string", "enum": ["line", "bar", "density"]},
                "protocols": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4]}, "description": "Empty means all; 1=Training 1, 2=Training 2, 3=Task, 4=Flexifliping."},
                "task_probability": {"type": "string", "enum": ["all", "90/30", "90/10", "50/50"]},
                "aggregation": {"type": "string", "enum": ["individual", "mean", "sum"], "description": "How to combine mice for line or bar plots."},
            }, "required": ["mice", "metric", "plot_type", "protocols", "task_probability", "aggregation"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "get_behavior_session",
        "description": "Return scalar metadata and outcome counts for one behavioral session.",
        "parameters": {
            "type": "object", "properties": {
                "mouse": {"type": "string"}, "date": {"type": "string"},
            }, "required": ["mouse", "date"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "get_ephys_sessions",
        "description": "Return read-only ephys session, probe, alignment, good-unit and region summaries.",
        "parameters": {
            "type": "object", "properties": {"mouse": {"type": "string"}},
            "required": ["mouse"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "get_openfield_sessions",
        "description": "Return open-field session dates, files, and all available OFT session metrics.",
        "parameters": {
            "type": "object", "properties": {"mouse": {"type": "string"}},
            "required": ["mouse"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "query_assay_metrics",
        "description": (
            "Query and plot the SQLite behavioral assay metrics for OFT, LDT, paired OFT-LDT, "
            "or per-mouse summaries. Use inspect_data_schema(dataset='openfield') first when the "
            "exact metric name is unknown. Supports distributions, longitudinal plots and "
            "correlations between two metrics from the same table."
        ),
        "parameters": {
            "type": "object", "properties": {
                "table": {"type": "string", "enum": ["oft_sessions", "ldt_sessions", "paired_behavior_metrics", "mouse_metrics"]},
                "mice": {"type": "array", "items": {"type": "string"}, "description": "Empty means all mice."},
                "metric": {"type": "string", "description": "Exact numeric column name."},
                "secondary_metric": {"type": ["string", "null"], "description": "Optional exact numeric x-axis metric for correlations."},
                "date_from": {"type": ["string", "null"], "description": "Inclusive YYYY-MM-DD bound."},
                "date_to": {"type": ["string", "null"], "description": "Inclusive YYYY-MM-DD bound."},
                "group_by": {"type": "string", "enum": ["session", "mouse"]},
                "aggregation": {"type": "string", "enum": ["none", "mean", "median", "sum", "min", "max"]},
                "plot_type": {"type": "string", "enum": ["none", "line", "bar", "scatter", "histogram", "box"]},
            },
            "required": ["table", "mice", "metric", "secondary_metric", "date_from", "date_to", "group_by", "aggregation", "plot_type"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "correlate_behavior_assay",
        "description": (
            "Create a real Plotly scatter plot correlating one Behavior metric with one OFT/LDT "
            "metric after aggregating both datasets by mouse. Use this for cross-dataset questions; "
            "never attempt to draw the result as SVG or text."
        ),
        "parameters": {
            "type": "object", "properties": {
                "mice": {"type": "array", "items": {"type": "string"}, "description": "Empty means all mice."},
                "behavior_metric": {"type": "string", "description": "Derived behavior metric key or exact scalar numeric Behavior column."},
                "assay_table": {"type": "string", "enum": ["oft_sessions", "ldt_sessions", "paired_behavior_metrics", "mouse_metrics"]},
                "assay_metric": {"type": "string", "description": "Exact numeric OFT/LDT column."},
                "protocols": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4]}, "description": "Empty means all behavior protocols."},
                "task_probability": {"type": "string", "enum": ["all", "90/30", "90/10", "50/50"]},
                "behavior_aggregation": {"type": "string", "enum": ["mean", "median", "sum"]},
                "assay_aggregation": {"type": "string", "enum": ["mean", "median", "sum"]},
            },
            "required": ["mice", "behavior_metric", "assay_table", "assay_metric", "protocols", "task_probability", "behavior_aggregation", "assay_aggregation"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "inspect_data_schema",
        "description": (
            "Inspect the live read-only data dictionary before an unfamiliar analysis. "
            "Returns available behavior columns/derived metrics or ephys unit metrics."
        ),
        "parameters": {
            "type": "object", "properties": {
                "dataset": {"type": "string", "enum": ["behavior", "ephys", "openfield"]},
            }, "required": ["dataset"], "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "query_behavior_data",
        "description": (
            "General read-only behavior analysis. Filter any combination of mouse, date, "
            "protocol and task probability; aggregate a derived metric or any scalar numeric "
            "database column; and optionally produce a Plotly figure. Protocol mapping is exact: "
            "Training 1=1, Training 2=2, Task=3, Flexifliping=4. Use this for questions "
            "not covered exactly by get_behavior_timeseries."
        ),
        "parameters": {
            "type": "object", "properties": {
                "mice": {"type": "array", "items": {"type": "string"}, "description": "Empty means all mice."},
                "date_from": {"type": ["string", "null"], "description": "Inclusive YYYY-MM-DD bound."},
                "date_to": {"type": ["string", "null"], "description": "Inclusive YYYY-MM-DD bound."},
                "protocols": {"type": "array", "items": {"type": "integer", "enum": [1, 2, 3, 4]}, "description": "1=Training 1, 2=Training 2, 3=Task, 4=Flexifliping; empty means all."},
                "task_probability": {"type": "string", "enum": ["all", "90/30", "90/10", "50/50"]},
                "metric": {"type": "string", "description": "Derived metric key or exact scalar numeric column from inspect_data_schema."},
                "secondary_metric": {"type": ["string", "null"], "description": "Optional x metric for a correlation scatter plot."},
                "group_by": {"type": "array", "items": {"type": "string", "enum": ["session", "mouse", "date", "protocol", "task_probability"]}},
                "aggregation": {"type": "string", "enum": ["none", "count", "sum", "mean", "median", "min", "max", "std"]},
                "plot_type": {"type": "string", "enum": ["none", "line", "bar", "scatter", "histogram", "density", "box", "violin"]},
                "split_by": {"type": "string", "enum": ["none", "mouse", "protocol", "task_probability"]},
            },
            "required": ["mice", "date_from", "date_to", "protocols", "task_probability", "metric", "secondary_metric", "group_by", "aggregation", "plot_type", "split_by"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function", "name": "query_ephys_units",
        "description": (
            "General read-only query over electrophysiology unit/QC metrics. It can compare "
            "sessions, probes, regions and Bombcell labels and render distributions or summaries."
        ),
        "parameters": {
            "type": "object", "properties": {
                "mouse": {"type": "string"},
                "dates": {"type": "array", "items": {"type": "string"}, "description": "Empty means all dates for the mouse."},
                "probes": {"type": "array", "items": {"type": "string"}, "description": "Empty means all probes."},
                "good_units_only": {"type": "boolean"},
                "regions": {"type": "array", "items": {"type": "string"}, "description": "Empty means all regions."},
                "metric": {"type": "string", "enum": ["unit_count", "num_spikes", "firing_rate", "presence_ratio", "snr", "rp_contamination", "amplitude_cutoff", "amplitude_median", "drift_ptp"]},
                "group_by": {"type": "array", "items": {"type": "string", "enum": ["session", "probe", "region", "label"]}},
                "aggregation": {"type": "string", "enum": ["count", "sum", "mean", "median", "min", "max", "std"]},
                "minimum_units_per_region": {"type": "integer", "minimum": 0, "maximum": 1000},
                "plot_type": {"type": "string", "enum": ["none", "bar", "scatter", "histogram", "density", "box", "violin"]},
                "split_by": {"type": "string", "enum": ["none", "session", "probe", "region", "label"]},
            },
            "required": ["mouse", "dates", "probes", "good_units_only", "regions", "metric", "group_by", "aggregation", "minimum_units_per_region", "plot_type", "split_by"],
            "additionalProperties": False,
        },
    },
]

CHATBOT_METRIC_LABELS = {
    "valid_bouts": "Valid bouts",
    "total_bouts": "Total bouts",
    "rewarded_licks": "Rewarded licks",
    "non_rewarded_licks": "Non-rewarded licks",
    "invalid_licks": "Invalid licks",
}
CHATBOT_TOOL_VERSION = "cross-dataset-plots-v7"


def chatbot_api_key():
    key = os.getenv("OPENAI_API_KEY") or st.session_state.get("chatbot_api_key")
    if key:
        return key
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def chatbot_behavior_values(dataframe, metric):
    if metric == "valid_bouts":
        return dataframe.apply(count_valid_bouts, axis=1)
    if metric == "total_bouts":
        column = "Number of Bouts"
        if column not in dataframe.columns:
            return pd.Series(0, index=dataframe.index, dtype=float)
        return pd.to_numeric(dataframe[column], errors="coerce").fillna(0)
    if metric == "rewarded_licks":
        column = "Number of Rewarded Licks"
        if column not in dataframe.columns:
            return pd.Series(0, index=dataframe.index, dtype=float)
        return pd.to_numeric(dataframe[column], errors="coerce").fillna(0)
    lick_kind = {"non_rewarded_licks": "non_rewarded", "invalid_licks": "invalid"}[metric]
    return dataframe.apply(lambda row: count_licks(row, lick_kind), axis=1)


CHATBOT_DERIVED_BEHAVIOR_METRICS = {
    "valid_bouts": "Valid bouts",
    "total_bouts": "Total bouts",
    "rewarded_licks": "Rewarded licks",
    "non_rewarded_licks": "Non-rewarded licks",
    "invalid_licks": "Invalid licks",
}

CHATBOT_EPHYS_METRICS = [
    "num_spikes", "firing_rate", "presence_ratio", "snr",
    "rp_contamination", "amplitude_cutoff", "amplitude_median", "drift_ptp",
]


def chatbot_scalar_columns(dataframe):
    """Only expose compact scalar columns; object columns often contain large event arrays."""
    return [
        column for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[column])
        or pd.api.types.is_bool_dtype(dataframe[column])
        or pd.api.types.is_datetime64_any_dtype(dataframe[column])
    ]


def chatbot_resolve_behavior_metric(dataframe, requested):
    if requested in CHATBOT_DERIVED_BEHAVIOR_METRICS:
        return chatbot_behavior_values(dataframe, requested), CHATBOT_DERIVED_BEHAVIOR_METRICS[requested]
    matches = {str(column).casefold(): column for column in chatbot_scalar_columns(dataframe)}
    column = matches.get(str(requested).casefold())
    if column is None or column == "Date":
        raise ValueError(
            f"Unknown or non-scalar metric '{requested}'. Call inspect_data_schema first."
        )
    return pd.to_numeric(dataframe[column], errors="coerce"), str(column)


def chatbot_probability_label(values):
    numeric = pd.to_numeric(values, errors="coerce")
    labels = pd.Series("Other", index=values.index, dtype=object)
    labels[np.isclose(numeric, .30, equal_nan=False)] = "90/30"
    labels[np.isclose(numeric, .10, equal_nan=False)] = "90/10"
    labels[np.isclose(numeric, .50, equal_nan=False)] = "50/50"
    return labels


def chatbot_plot_figure(records, metric_label, plot_type, split_by="none", x_label=None):
    figure = go.Figure()
    split_column = None if split_by == "none" or split_by not in records.columns else split_by
    groups = [("All data", records)] if split_column is None else list(records.groupby(split_column, sort=True))
    colors = ["#2F91BB", "#7654A8", "#49A17D", "#DF8B3F", "#CE5877", "#5879B9"]
    for index, (group_name, group) in enumerate(groups):
        color = colors[index % len(colors)]
        name = str(group_name)
        if plot_type in {"histogram", "density"}:
            values = pd.to_numeric(group["value"], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            figure.add_trace(go.Histogram(
                x=values, name=name, histnorm="probability density" if plot_type == "density" else None,
                nbinsx=min(40, max(8, int(np.sqrt(len(values))))), opacity=.34,
                marker={"color": color, "line": {"width": 0}},
            ))
            if plot_type == "density" and len(values) >= 3 and np.nanstd(values) > 0:
                xs = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 240)
                try:
                    kde = gaussian_kde(values)
                    figure.add_trace(go.Scatter(
                        x=xs, y=kde(xs), mode="lines", name=name,
                        line={"color": color, "width": 2.8}, showlegend=split_column is not None,
                    ))
                except np.linalg.LinAlgError:
                    pass
        elif plot_type in {"box", "violin"}:
            trace_class = go.Box if plot_type == "box" else go.Violin
            kwargs = {"y": group["value"], "name": name, "marker_color": color, "boxpoints": "outliers"}
            if plot_type == "violin":
                kwargs = {"y": group["value"], "name": name, "line_color": color, "box_visible": True, "meanline_visible": True}
            figure.add_trace(trace_class(**kwargs))
        else:
            x = group["x"] if "x" in group else np.arange(len(group))
            if plot_type == "bar":
                figure.add_trace(go.Bar(x=x, y=group["value"], name=name, marker={"color": color, "cornerradius": 5}))
            else:
                figure.add_trace(go.Scatter(
                    x=x, y=group["value"], name=name,
                    mode="lines+markers" if plot_type == "line" else "markers",
                    line={"color": color, "width": 2.5}, marker={"color": color, "size": 7},
                ))
    if plot_type in {"histogram", "density"}:
        figure.update_xaxes(title=metric_label)
        figure.update_yaxes(title="Density" if plot_type == "density" else "Sessions / units")
        figure.update_layout(barmode="overlay")
    elif plot_type in {"box", "violin"}:
        figure.update_yaxes(title=metric_label)
    else:
        figure.update_yaxes(title=metric_label, rangemode="tozero" if plot_type == "bar" else "normal")
        if x_label:
            figure.update_xaxes(title=x_label)
    return _finish_behavior_plot(figure, height=410, showlegend=split_column is not None)


def chatbot_group_behavior(records, group_by, aggregation):
    group_map = {
        "mouse": "mouse", "date": "date", "protocol": "protocol",
        "task_probability": "task_probability", "session": "session",
    }
    columns = [group_map[item] for item in group_by]
    if aggregation == "none":
        grouped = records.copy()
    else:
        value_columns = [column for column in ["value", "secondary_value"] if column in records]
        if aggregation == "count":
            grouped = records.groupby(columns, dropna=False, as_index=False)["value"].count()
        else:
            grouped = records.groupby(columns, dropna=False, as_index=False)[value_columns].agg(aggregation)
    x_column = next((column for column in ["date", "session", "mouse", "protocol", "task_probability"] if column in grouped), None)
    if x_column:
        grouped["x"] = grouped[x_column]
    return grouped


@st.cache_data(show_spinner="Reading ephys unit metrics…", max_entries=128)
def load_chatbot_ephys_units(mouse, date, probe, locations_path, session_path, signature):
    del signature
    session_dir = Path(session_path)
    probe_dir = casefold_path(session_dir, "alf", probe)
    location_file = Path(locations_path)
    channel_rows = []
    if location_file.suffix.lower() == ".json" and location_file.exists():
        with location_file.open(encoding="utf-8") as stream:
            locations = json.load(stream)
        for name, location in locations.items():
            suffix = name.rsplit("_", 1)[-1]
            if suffix.isdigit() and isinstance(location, dict):
                channel_rows.append({
                    "channel_id": int(suffix),
                    "depth_um": pd.to_numeric(location.get("axial"), errors="coerce"),
                    "brain_region": str(location.get("brain_region", "Unknown")),
                })
    channels = pd.DataFrame(channel_rows)
    if not channels.empty:
        channels = channels.drop_duplicates("channel_id").set_index("channel_id")
    metrics_path = casefold_path(probe_dir, "clusters.metrics.csv")
    labels_path = casefold_path(session_dir, "bombcell", probe, "bombcell_labels.csv")
    peak_path = casefold_path(probe_dir, "clusters.channels.npy")
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    labels = pd.read_csv(labels_path, index_col=0) if labels_path.exists() else pd.DataFrame()
    peaks = np.asarray(np.load(peak_path), dtype=int) if peak_path.exists() else np.array([], dtype=int)
    unit_count = max(len(metrics), len(labels), len(peaks))
    units = pd.DataFrame({"unit_id": np.arange(unit_count, dtype=int)})
    if len(peaks):
        peak_series = pd.Series(peaks)
        units["peak_channel"] = units["unit_id"].map(peak_series)
        if not channels.empty:
            units["brain_region"] = units["peak_channel"].map(channels["brain_region"])
            units["depth_um"] = units["peak_channel"].map(channels["depth_um"])
    if "bombcell_label" in labels:
        labels.index = pd.to_numeric(labels.index, errors="coerce")
        units["bombcell_label"] = units["unit_id"].map(labels["bombcell_label"])
    available = [column for column in ["cluster_id", *CHATBOT_EPHYS_METRICS] if column in metrics]
    if "cluster_id" in available:
        units = units.merge(metrics[available], left_on="unit_id", right_on="cluster_id", how="left").drop(columns="cluster_id")
    else:
        for column in available:
            units[column] = metrics[column].reindex(units.index)
    for column in CHATBOT_EPHYS_METRICS:
        if column not in units:
            units[column] = np.nan
    spikes_path = casefold_path(probe_dir, "spikes.times.npy")
    if units["firing_rate"].isna().all() and spikes_path.exists():
        spike_times = np.load(spikes_path, mmap_mode="r")
        if len(spike_times) > 1:
            duration_s = float(spike_times[-1]) - float(spike_times[0])
            if duration_s > 0:
                units["firing_rate"] = pd.to_numeric(units["num_spikes"], errors="coerce") / duration_s
    units["amplitude_median"] = pd.to_numeric(
        units["amplitude_median"], errors="coerce"
    ).abs()
    units["mouse"], units["session"], units["probe"] = mouse, date, probe
    units["region"] = units.get("brain_region", pd.Series("Unknown", index=units.index)).fillna("Unknown").astype(str)
    units["label"] = units.get("bombcell_label", pd.Series("unlabelled", index=units.index)).fillna("unlabelled").astype(str)
    return units


def chatbot_behavior_figure(records, metric, plot_type, aggregation="individual"):
    if plot_type == "density":
        values = records["value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        figure = go.Figure()
        figure.add_trace(go.Histogram(
            x=values, histnorm="probability density", nbinsx=min(35, max(8, int(np.sqrt(len(values))))),
            name="Sessions", marker={"color": "rgba(66,140,179,.42)", "line": {"width": 0}},
            hovertemplate=f"{CHATBOT_METRIC_LABELS[metric]} %{{x:.1f}}<br>Density %{{y:.4f}}<extra></extra>",
        ))
        if len(values) >= 3 and np.nanstd(values) > 0:
            xs = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 240)
            kde = gaussian_kde(values)
            figure.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines", name="Density",
                line={"color": "#7654A8", "width": 2.7},
                hovertemplate=f"{CHATBOT_METRIC_LABELS[metric]} %{{x:.1f}}<br>Density %{{y:.4f}}<extra></extra>",
            ))
        figure.update_xaxes(title=CHATBOT_METRIC_LABELS[metric])
        figure.update_yaxes(title="Density", rangemode="tozero")
        return _finish_behavior_plot(figure, height=410, showlegend=False)

    if aggregation in {"mean", "sum"}:
        aggregator = "mean" if aggregation == "mean" else "sum"
        records = (
            records.groupby("date", as_index=False)["value"].agg(aggregator)
            .assign(mouse=f"All mice ({aggregation})", protocol="combined")
        )
    figure = go.Figure()
    colors = _behavior_date_colors(max(1, len(records["mouse"].unique())))
    for index, (mouse, mouse_records) in enumerate(records.groupby("mouse", sort=True)):
        mode = "lines+markers" if plot_type == "line" else None
        trace_class = go.Scatter if plot_type == "line" else go.Bar
        trace_kwargs = {
            "x": mouse_records["date"], "y": mouse_records["value"],
            "name": mouse, "customdata": mouse_records[["protocol"]].to_numpy(),
            "hovertemplate": "%{x|%Y-%m-%d}<br>%{y:.0f}<br>Protocol %{customdata[0]}<extra></extra>",
        }
        if plot_type == "line":
            trace_kwargs.update(mode=mode, line={"color": colors[index], "width": 2.4}, marker={"size": 6})
        else:
            trace_kwargs.update(marker={"color": colors[index], "cornerradius": 5})
        figure.add_trace(trace_class(**trace_kwargs))
    figure.update_xaxes(title="Session date")
    figure.update_yaxes(title=CHATBOT_METRIC_LABELS[metric], rangemode="tozero")
    return _finish_behavior_plot(figure, height=410, showlegend=len(records["mouse"].unique()) > 1)


def assay_metric_figure(records, metric, plot_type, secondary_metric=None):
    label = metric_display_name(metric)
    figure = go.Figure()
    if plot_type == "scatter" and secondary_metric:
        figure.add_trace(go.Scatter(
            x=records["secondary_value"], y=records["value"], mode="markers",
            marker={"size": 9, "color": "#428CB3", "line": {"color": "#FFFFFF", "width": 1}},
            customdata=records[["mouse", "date_label"]].to_numpy(),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
        ))
        figure.update_xaxes(title=metric_display_name(secondary_metric))
        figure.update_yaxes(title=label)
    elif plot_type == "histogram":
        figure.add_trace(go.Histogram(
            x=records["value"], marker={"color": "#428CB3", "cornerradius": 5},
            hovertemplate="%{x:.3g}<br>n=%{y}<extra></extra>",
        ))
        figure.update_xaxes(title=label)
        figure.update_yaxes(title="Sessions")
    elif plot_type == "box":
        for mouse, rows in records.groupby("mouse", sort=True):
            figure.add_trace(go.Box(
                x=[mouse] * len(rows), y=rows["value"], name=mouse,
                marker={"color": "#428CB3"}, boxpoints="all", jitter=.25,
                hovertemplate=f"{mouse}<br>%{{y:.3g}}<extra></extra>",
            ))
        figure.update_xaxes(title="Mouse")
        figure.update_yaxes(title=label)
    else:
        trace_type = go.Bar if plot_type == "bar" else go.Scatter
        for index, (mouse, rows) in enumerate(records.groupby("mouse", sort=True)):
            color = _behavior_date_colors(max(1, records["mouse"].nunique()))[index]
            if plot_type == "bar":
                figure.add_trace(trace_type(
                    x=rows["x"], y=rows["value"], name=mouse,
                    marker={"color": color, "cornerradius": 5},
                ))
            else:
                figure.add_trace(trace_type(
                    x=rows["x"], y=rows["value"], name=mouse, mode="lines+markers",
                    line={"color": color, "width": 2.4}, marker={"size": 7},
                ))
        figure.update_xaxes(title="Session" if "date" in records else "Mouse")
        figure.update_yaxes(title=label)
    return _finish_behavior_plot(
        figure, height=410, showlegend=records["mouse"].nunique() > 1,
        margin={"l": 58, "r": 26, "t": 28, "b": 54},
    )


def execute_chatbot_tool(name, arguments):
    figures = []
    behavior_df = None
    if BEHAVIOR_DB.exists():
        behavior_df = load_behavior_database(
            str(BEHAVIOR_DB), source_signature(BEHAVIOR_DB)
        )
        behavior_df = behavior_df[behavior_df["Mouse_ID"].apply(is_visible_mouse)]

    if name == "get_domain_knowledge":
        result = AGENT_DOMAIN_KNOWLEDGE
    elif name == "inspect_data_schema":
        dataset = arguments["dataset"]
        if dataset == "behavior":
            if behavior_df is None:
                raise ValueError("Behavior database is unavailable.")
            scalar_columns = chatbot_scalar_columns(behavior_df)
            result = {
                "dataset": "behavior",
                "rows": int(len(behavior_df)),
                "mice": sorted(behavior_df["Mouse_ID"].dropna().astype(str).unique().tolist()),
                "derived_metrics": CHATBOT_DERIVED_BEHAVIOR_METRICS,
                "protocol_mapping": {
                    "1": "Training 1", "2": "Training 2", "3": "Task",
                    "4": "Flexifliping",
                },
                "scalar_columns": [
                    {"name": str(column), "dtype": str(behavior_df[column].dtype)}
                    for column in scalar_columns
                ],
                "dimensions": ["session", "mouse", "date", "protocol", "task_probability"],
                "note": "Array/object event columns are intentionally excluded from the generic engine.",
            }
        elif dataset == "ephys":
            inventory = discover_ibl_channel_locations(EPHYS_ROOT)
            result = {
                "dataset": "ephys", "unit_metrics": ["unit_count", *CHATBOT_EPHYS_METRICS],
                "dimensions": ["session", "probe", "region", "label"],
                "mice": sorted(inventory["mouse"].astype(str).unique().tolist()) if not inventory.empty else [],
                "insertions": int(len(inventory)),
            }
        else:
            inventory = discover_openfield_sessions(OPENFIELD_ROOT)
            assay_tables = {}
            for table_name in ("oft_sessions", "ldt_sessions", "paired_behavior_metrics", "mouse_metrics"):
                table = behavior_metrics_table(table_name)
                assay_tables[table_name] = {
                    "rows": int(len(table)),
                    "columns": [
                        {"name": str(column), "dtype": str(table[column].dtype)}
                        for column in table.columns
                    ],
                    "numeric_metrics": [
                        str(column) for column in table.select_dtypes(include=[np.number]).columns
                    ],
                }
            result = {
                "dataset": "openfield", "dimensions": ["mouse", "date"],
                "available_metrics": ["sessions", "videos", "tracking_files", "OFT metrics", "LDT metrics"],
                "mice": sorted(inventory["mouse"].astype(str).unique().tolist()) if not inventory.empty else [],
                "sqlite_database": str(BEHAVIOR_METRICS_DB),
                "tables": assay_tables,
            }
    elif name == "query_assay_metrics":
        table_name = arguments["table"]
        source = behavior_metrics_table(table_name)
        if source.empty:
            raise ValueError(f"The assay table '{table_name}' is unavailable or empty.")
        selected = source[source["mouse"].astype(str).apply(is_visible_mouse)].copy()
        mice = [mouse for mouse in arguments["mice"] if is_visible_mouse(mouse)]
        if mice:
            selected = selected[selected["mouse"].astype(str).isin(mice)]
        date_column = next((column for column in ("oft_date", "ldt_date") if column in selected.columns), None)
        if date_column:
            if arguments["date_from"]:
                selected = selected[selected[date_column] >= pd.Timestamp(arguments["date_from"])]
            if arguments["date_to"]:
                selected = selected[selected[date_column] <= pd.Timestamp(arguments["date_to"])]
        metric = arguments["metric"]
        secondary = arguments.get("secondary_metric")
        if metric not in selected.columns or not pd.api.types.is_numeric_dtype(selected[metric]):
            raise ValueError(f"'{metric}' is not an available numeric metric in {table_name}.")
        if secondary and (
            secondary not in selected.columns or not pd.api.types.is_numeric_dtype(selected[secondary])
        ):
            raise ValueError(f"'{secondary}' is not an available numeric metric in {table_name}.")
        records = pd.DataFrame({
            "mouse": selected["mouse"].astype(str),
            "value": pd.to_numeric(selected[metric], errors="coerce"),
        })
        if date_column:
            records["date"] = pd.to_datetime(selected[date_column], errors="coerce")
            records["date_label"] = records["date"].dt.strftime("%Y-%m-%d")
            records["x"] = records["date"]
        else:
            records["date_label"] = "summary"
            records["x"] = records["mouse"]
        if secondary:
            records["secondary_value"] = pd.to_numeric(selected[secondary], errors="coerce")
        records = records.dropna(subset=["value"] + (["secondary_value"] if secondary else []))
        if records.empty:
            raise ValueError("No assay rows match these filters.")
        raw_records = records.copy()
        if arguments["group_by"] == "mouse" and arguments["aggregation"] != "none":
            aggregation = arguments["aggregation"]
            columns = ["value"] + (["secondary_value"] if secondary else [])
            records = records.groupby("mouse", as_index=False)[columns].agg(aggregation)
            records["date_label"] = "summary"
            records["x"] = records["mouse"]
        plot_type = arguments["plot_type"]
        if plot_type != "none":
            figures.append(assay_metric_figure(records, metric, plot_type, secondary))
        sample = records.head(200).copy()
        if "date" in sample:
            sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
        result = {
            "dataset": table_name, "metric": metric, "secondary_metric": secondary,
            "matched_rows": int(len(raw_records)), "result_rows": int(len(records)),
            "statistics": {
                "mean": float(raw_records["value"].mean()),
                "median": float(raw_records["value"].median()),
                "std": float(raw_records["value"].std()) if len(raw_records) > 1 else 0.0,
                "min": float(raw_records["value"].min()),
                "max": float(raw_records["value"].max()),
            },
            "records": sample.where(pd.notna(sample), None).to_dict("records"),
            "records_truncated": len(records) > 200, "chart_attached": bool(figures),
        }
    elif name == "correlate_behavior_assay":
        if behavior_df is None:
            raise ValueError("Behavior database is unavailable.")
        behavior = behavior_df.copy()
        mice = [mouse for mouse in arguments["mice"] if is_visible_mouse(mouse)]
        if mice:
            behavior = behavior[behavior["Mouse_ID"].astype(str).isin(mice)]
        protocols = arguments["protocols"]
        if protocols:
            behavior = behavior[pd.to_numeric(behavior["Protocol"], errors="coerce").isin(protocols)]
        probability = arguments["task_probability"]
        probability_values = {"90/30": .30, "90/10": .10, "50/50": .50}
        if probability in probability_values:
            behavior = behavior[np.isclose(
                pd.to_numeric(behavior["Proba_val"], errors="coerce"),
                probability_values[probability], equal_nan=False,
            )]
        behavior_values, behavior_label = chatbot_resolve_behavior_metric(
            behavior, arguments["behavior_metric"]
        )
        behavior_records = pd.DataFrame({
            "mouse": behavior["Mouse_ID"].astype(str),
            "behavior_value": pd.to_numeric(behavior_values, errors="coerce"),
        }).dropna()
        behavior_summary = behavior_records.groupby("mouse", as_index=False)["behavior_value"].agg(
            arguments["behavior_aggregation"]
        )

        assay = behavior_metrics_table(arguments["assay_table"])
        if assay.empty:
            raise ValueError(f"The assay table '{arguments['assay_table']}' is unavailable.")
        assay = assay[assay["mouse"].astype(str).apply(is_visible_mouse)].copy()
        if mice:
            assay = assay[assay["mouse"].astype(str).isin(mice)]
        assay_metric = arguments["assay_metric"]
        if assay_metric not in assay.columns or not pd.api.types.is_numeric_dtype(assay[assay_metric]):
            raise ValueError(f"'{assay_metric}' is not an available numeric assay metric.")
        assay["assay_value"] = pd.to_numeric(assay[assay_metric], errors="coerce")
        assay_summary = assay.dropna(subset=["assay_value"]).groupby(
            "mouse", as_index=False
        )["assay_value"].agg(arguments["assay_aggregation"])
        paired = behavior_summary.merge(assay_summary, on="mouse", how="inner").dropna()
        if paired.empty:
            raise ValueError("No mice have both matching Behavior and assay measurements.")

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=paired["assay_value"], y=paired["behavior_value"], mode="markers",
            marker={"size": 11, "color": "#428CB3", "line": {"color": "#FFFFFF", "width": 1.4}},
            customdata=paired[["mouse"]].to_numpy(),
            hovertemplate="%{customdata[0]}<br>x=%{x:.3g}<br>y=%{y:.3g}<extra></extra>",
        ))
        if len(paired) >= 3 and paired["assay_value"].nunique() > 1:
            slope, intercept = np.polyfit(paired["assay_value"], paired["behavior_value"], 1)
            x_line = np.linspace(paired["assay_value"].min(), paired["assay_value"].max(), 100)
            figure.add_trace(go.Scatter(
                x=x_line, y=slope * x_line + intercept, mode="lines",
                line={"color": "#CF725E", "width": 2, "dash": "dash"},
                name="Linear fit", hoverinfo="skip",
            ))
        assay_label = metric_display_name(assay_metric)
        figure.update_xaxes(title=assay_label)
        figure.update_yaxes(title=behavior_label)
        figures.append(_finish_behavior_plot(
            figure, height=430, showlegend=len(figure.data) > 1,
            margin={"l": 62, "r": 28, "t": 28, "b": 58},
        ))
        correlations = {}
        if len(paired) >= 3:
            correlations = {
                "pearson_r": float(paired[["assay_value", "behavior_value"]].corr(method="pearson").iloc[0, 1]),
                "spearman_r": float(paired[["assay_value", "behavior_value"]].corr(method="spearman").iloc[0, 1]),
            }
        result = {
            "dataset": "behavior_x_assay", "behavior_metric": behavior_label,
            "assay_metric": assay_metric, "assay_table": arguments["assay_table"],
            "matched_mice": int(len(paired)), "matched_behavior_sessions": int(len(behavior_records)),
            "correlations": correlations, "filters": {
                "mice": mice or "all", "protocols": protocols or "all",
                "task_probability": probability,
                "behavior_aggregation": arguments["behavior_aggregation"],
                "assay_aggregation": arguments["assay_aggregation"],
            },
            "records": paired.where(pd.notna(paired), None).to_dict("records"),
            "chart_attached": True,
        }
    elif name == "query_behavior_data":
        if behavior_df is None:
            raise ValueError("Behavior database is unavailable.")
        selected = behavior_df.copy()
        mice = [mouse for mouse in arguments["mice"] if is_visible_mouse(mouse)]
        if mice:
            selected = selected[selected["Mouse_ID"].astype(str).isin(mice)]
        protocols = arguments["protocols"]
        if protocols:
            selected = selected[pd.to_numeric(selected["Protocol"], errors="coerce").isin(protocols)]
        probability = arguments["task_probability"]
        probability_values = {"90/30": .30, "90/10": .10, "50/50": .50}
        if probability in probability_values:
            selected = selected[np.isclose(
                pd.to_numeric(selected["Proba_val"], errors="coerce"),
                probability_values[probability], equal_nan=False,
            )]
        if arguments["date_from"]:
            selected = selected[selected["Date"] >= pd.Timestamp(arguments["date_from"])]
        if arguments["date_to"]:
            selected = selected[selected["Date"] <= pd.Timestamp(arguments["date_to"])]
        values, metric_label = chatbot_resolve_behavior_metric(selected, arguments["metric"])
        secondary_metric = arguments.get("secondary_metric")
        if secondary_metric:
            secondary_values, secondary_label = chatbot_resolve_behavior_metric(selected, secondary_metric)
        else:
            secondary_values, secondary_label = None, None
        records = pd.DataFrame({
            "mouse": selected["Mouse_ID"].astype(str),
            "date": pd.to_datetime(selected["Date"], errors="coerce"),
            "protocol": pd.to_numeric(selected["Protocol"], errors="coerce")
                .map(PROTOCOL_LABELS).fillna("Unknown"),
            "task_probability": chatbot_probability_label(selected["Proba_val"]),
            "value": pd.to_numeric(values, errors="coerce"),
        }).dropna(subset=["date", "value"]).sort_values(["date", "mouse"])
        if secondary_values is not None:
            records["secondary_value"] = pd.to_numeric(secondary_values, errors="coerce")
            records = records.dropna(subset=["secondary_value"])
        records["session"] = records["mouse"] + " · " + records["date"].dt.strftime("%Y-%m-%d")
        if records.empty:
            raise ValueError("No behavior rows match these filters.")
        group_by = arguments["group_by"] or ["session"]
        grouped = chatbot_group_behavior(records, group_by, arguments["aggregation"])
        plot_type = arguments["plot_type"]
        if plot_type != "none":
            correlation_plot = plot_type == "scatter" and secondary_label is not None
            plot_records = records if plot_type in {"histogram", "density", "box", "violin"} or correlation_plot else grouped
            if correlation_plot:
                plot_records = plot_records.copy()
                plot_records["x"] = plot_records["secondary_value"]
            figures.append(chatbot_plot_figure(
                plot_records, metric_label, plot_type, arguments["split_by"], secondary_label
            ))
        sample = grouped.head(200).copy()
        for column in sample.select_dtypes(include=["datetime", "datetimetz"]).columns:
            sample[column] = sample[column].dt.strftime("%Y-%m-%d")
        statistics = {
            "mean": float(records["value"].mean()), "median": float(records["value"].median()),
            "std": float(records["value"].std()) if len(records) > 1 else 0.0,
            "min": float(records["value"].min()), "max": float(records["value"].max()),
        }
        if secondary_label is not None and len(records) >= 3:
            statistics["pearson_r"] = float(records[["secondary_value", "value"]].corr(method="pearson").iloc[0, 1])
            statistics["spearman_r"] = float(records[["secondary_value", "value"]].corr(method="spearman").iloc[0, 1])
        result = {
            "dataset": "behavior", "metric": metric_label, "secondary_metric": secondary_label,
            "matched_sessions": int(len(records)), "result_rows": int(len(grouped)),
            "statistics": statistics,
            "query": {key: arguments[key] for key in [
                "mice", "date_from", "date_to", "protocols", "task_probability",
                "group_by", "aggregation", "plot_type", "split_by",
            ]},
            "protocol_filter_labels": [PROTOCOL_LABELS[value] for value in protocols]
                if protocols else ["Training 1", "Training 2", "Task", "Flexifliping"],
            "records": sample.where(pd.notna(sample), None).to_dict("records"),
            "records_truncated": len(grouped) > 200, "chart_attached": bool(figures),
        }
    elif name == "query_ephys_units":
        inventory = discover_ibl_channel_locations(EPHYS_ROOT)
        inventory = inventory[inventory["mouse"].astype(str) == arguments["mouse"]]
        if arguments["dates"]:
            inventory = inventory[inventory["date"].isin(arguments["dates"])]
        if arguments["probes"]:
            inventory = inventory[inventory["probe"].isin(arguments["probes"])]
        if inventory.empty:
            raise ValueError("No ephys insertions match these filters.")
        unit_tables = []
        for row in inventory.itertuples(index=False):
            unit_tables.append(load_chatbot_ephys_units(
                row.mouse, row.date, row.probe, row.path, row.session_dir,
                probe_analysis_signature(Path(row.path), row.session_dir, row.probe),
            ))
        units = pd.concat(unit_tables, ignore_index=True)
        if arguments["good_units_only"]:
            units = units[units["label"].str.casefold() == "good"]
        if arguments["regions"]:
            wanted = {region.casefold() for region in arguments["regions"]}
            units = units[units["region"].str.casefold().isin(wanted)]
        minimum = arguments["minimum_units_per_region"]
        if minimum:
            region_counts = units.groupby("region")["unit_id"].count()
            units = units[units["region"].isin(region_counts[region_counts >= minimum].index)]
        if units.empty:
            raise ValueError("No units match these filters.")
        metric = arguments["metric"]
        units["value"] = 1.0 if metric == "unit_count" else pd.to_numeric(units[metric], errors="coerce")
        units = units.dropna(subset=["value"])
        if units.empty:
            raise ValueError(f"The metric '{metric}' is unavailable for the matching units.")
        group_by = arguments["group_by"] or ["session", "probe"]
        aggregation = "count" if metric == "unit_count" else arguments["aggregation"]
        if aggregation == "count":
            grouped = units.groupby(group_by, dropna=False, as_index=False)["value"].count()
        else:
            grouped = units.groupby(group_by, dropna=False, as_index=False)["value"].agg(aggregation)
        grouped["x"] = grouped[group_by[0]].astype(str)
        plot_type = arguments["plot_type"]
        if plot_type != "none":
            plot_records = units if plot_type in {"histogram", "density", "box", "violin"} else grouped
            figures.append(chatbot_plot_figure(
                plot_records, metric.replace("_", " ").title(), plot_type, arguments["split_by"]
            ))
        sample = grouped.head(200).where(pd.notna(grouped.head(200)), None)
        result = {
            "dataset": "ephys", "mouse": arguments["mouse"], "metric": metric,
            "matched_units": int(len(units)), "matched_insertions": int(len(inventory)),
            "result_rows": int(len(grouped)),
            "statistics": {
                "mean": float(units["value"].mean()), "median": float(units["value"].median()),
                "min": float(units["value"].min()), "max": float(units["value"].max()),
            },
            "query": arguments, "records": sample.to_dict("records"),
            "records_truncated": len(grouped) > 200, "chart_attached": bool(figures),
        }
    elif name == "get_data_catalog":
        requested_mouse = arguments.get("mouse")
        behavior_rows = behavior_df
        if requested_mouse and behavior_rows is not None:
            behavior_rows = behavior_rows[behavior_rows["Mouse_ID"] == requested_mouse]
        ephys = discover_ibl_channel_locations(EPHYS_ROOT)
        openfield = discover_openfield_sessions(OPENFIELD_ROOT)
        if requested_mouse:
            ephys = ephys[ephys["mouse"] == requested_mouse] if not ephys.empty else ephys
            openfield = openfield[openfield["mouse"] == requested_mouse] if not openfield.empty else openfield
        result = {
            "scope": requested_mouse or "all mice",
            "behavior_sessions": int(len(behavior_rows)) if behavior_rows is not None else 0,
            "behavior_mice": sorted(behavior_rows["Mouse_ID"].unique().tolist()) if behavior_rows is not None else [],
            "ephys_sessions": int(ephys[["mouse", "date"]].drop_duplicates().shape[0]) if not ephys.empty else 0,
            "ephys_probes": int(len(ephys)),
            "openfield_sessions": int(len(openfield)),
        }
    elif name == "get_behavior_timeseries":
        if behavior_df is None:
            raise ValueError("Behavior database is unavailable.")
        mice = [mouse for mouse in arguments["mice"] if is_visible_mouse(mouse)]
        if not mice:
            mice = sorted(behavior_df["Mouse_ID"].dropna().unique().tolist())
        metric = arguments["metric"]
        selected = behavior_df[behavior_df["Mouse_ID"].isin(mice)].copy()
        protocols = arguments.get("protocols", [])
        if protocols:
            selected = selected[selected["Protocol"].isin(protocols)]
        probability = arguments.get("task_probability", "all")
        probability_values = {"90/30": .30, "90/10": .10, "50/50": .50}
        if probability in probability_values:
            selected = selected[
                np.isclose(
                    pd.to_numeric(selected["Proba_val"], errors="coerce"),
                    probability_values[probability], equal_nan=False,
                )
            ]
        selected["value"] = chatbot_behavior_values(selected, metric)
        records = pd.DataFrame({
            "mouse": selected["Mouse_ID"].astype(str), "date": selected["Date"],
            "protocol": selected["Protocol"].astype(str), "value": selected["value"].astype(float),
        }).dropna(subset=["date", "value"]).sort_values(["mouse", "date"])
        if records.empty:
            raise ValueError("No matching behavior sessions were found.")
        aggregation = arguments.get("aggregation", "individual")
        figures.append(chatbot_behavior_figure(
            records, metric, arguments["plot_type"], aggregation
        ))
        result = {
            "metric": CHATBOT_METRIC_LABELS[metric], "mice": mice,
            "sessions": int(len(records)),
            "filters": {
                "protocols": protocols or "all",
                "task_probability": probability,
                "aggregation": aggregation,
                "plot_type": arguments["plot_type"],
            },
            "mean": float(records["value"].mean()),
            "minimum": float(records["value"].min()),
            "maximum": float(records["value"].max()),
            "latest_values": records.groupby("mouse").tail(1)[["mouse", "date", "value"]]
                .assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
            "chart_attached": True,
        }
    elif name == "get_behavior_session":
        if behavior_df is None:
            raise ValueError("Behavior database is unavailable.")
        rows = behavior_df[
            (behavior_df["Mouse_ID"] == arguments["mouse"])
            & (behavior_df["Date"].dt.strftime("%Y-%m-%d") == arguments["date"])
        ]
        if rows.empty:
            raise ValueError("Behavior session not found.")
        row = rows.iloc[-1]
        total_bouts = pd.to_numeric(row.get("Number of Bouts"), errors="coerce")
        rewarded_licks = pd.to_numeric(
            row.get("Number of Rewarded Licks"), errors="coerce"
        )
        result = {
            "mouse": arguments["mouse"], "date": arguments["date"],
            "version": str(row.get("Version", "")),
            "protocol": int(row["Protocol"]) if pd.notna(row.get("Protocol")) else None,
            "protocol_label": PROTOCOL_LABELS.get(int(row["Protocol"]), "Unknown")
                if pd.notna(row.get("Protocol")) else None,
            "probabilities": str(row.get("Probas", "")),
            "valid_bouts": int(count_valid_bouts(row)),
            "total_bouts": int(total_bouts) if pd.notna(total_bouts) else 0,
            "rewarded_licks": int(rewarded_licks) if pd.notna(rewarded_licks) else 0,
            "non_rewarded_licks": int(count_licks(row, "non_rewarded")),
            "invalid_licks": int(count_licks(row, "invalid")),
        }
    elif name == "get_ephys_sessions":
        inventory = discover_ibl_channel_locations(EPHYS_ROOT)
        inventory = inventory[inventory["mouse"] == arguments["mouse"]]
        if inventory.empty:
            result = {"mouse": arguments["mouse"], "sessions": []}
        else:
            rows = inventory[["mouse", "date", "probe", "path", "session_dir", "source"]].to_dict("records")
            tracks = load_probe_tracks(rows, channel_locations_signature(inventory))
            result = {"mouse": arguments["mouse"], "sessions": [{
                "date": track["date"], "probe": track["probe"], "source": track["source"],
                "good_units": int(track.get("good_units", 0)),
                "total_units": int(track.get("total_units", 0)),
                "regions": track.get("regions", []),
            } for track in tracks]}
    elif name == "get_openfield_sessions":
        inventory = discover_openfield_sessions(OPENFIELD_ROOT)
        inventory = inventory[inventory["mouse"] == arguments["mouse"]]
        metrics = behavior_metrics_table("oft_sessions")
        if not metrics.empty:
            metrics = metrics[metrics["mouse"].astype(str) == arguments["mouse"]].copy()
            metrics["date"] = metrics["oft_date"].dt.strftime("%Y-%m-%d")
            metric_columns = [column for column in metrics.columns if column.startswith("OFT_")]
            metrics_by_date = metrics.set_index("date")[metric_columns].to_dict("index")
        else:
            metrics_by_date = {}
        result = {"mouse": arguments["mouse"], "sessions": [{
            "date": row.date, "videos": len(row.videos), "tracking_files": len(row.csv_files),
            "video_names": [Path(path).name for path in row.videos],
            "metrics": metrics_by_date.get(row.date, {}),
        } for row in inventory.itertuples(index=False)]}
    else:
        raise ValueError(f"Unsupported read-only tool: {name}")
    return json.dumps(result, ensure_ascii=False, default=str), figures


def run_chatbot_turn(
    api_key, messages, selected_mouse=None, analysis_mode="Advanced",
    use_astra=False,
):
    client = OpenAI(api_key=api_key)
    default_model = "gpt-5.6-sol" if analysis_mode == "Advanced" else "gpt-5.6-terra"
    model = "gpt-6-astra" if use_astra else os.getenv("OPENAI_CHAT_MODEL", default_model)
    domain_knowledge = json.dumps(
        AGENT_DOMAIN_KNOWLEDGE, ensure_ascii=False, separators=(",", ":")
    )
    instructions = (
        "You are Agent, a rigorous scientific data analyst embedded in the INVIBE dashboard. "
        "Answer in the user's language. You have read-only analytical tools for Behavior, Openfield "
        "and electrophysiology. For every factual statement about local data, call a tool; never "
        "estimate or invent a value. For an unfamiliar variable, first call inspect_data_schema. "
        "Use this behavioral vocabulary exactly: Training 1 means Protocol 1; Training 2 means "
        "Protocol 2; Task means Protocol 3; Flexifliping means Protocol 4. Task and Flexifliping "
        "are session types, not synonyms for all behavior. "
        "A request for training means Protocols 1 and 2 unless the user specifies one stage. "
        "A request for Task must always filter protocols=[3], and a request for Flexifliping must "
        "always filter protocols=[4]. Never call Protocol 3 'Training 3'. "
        "Probability labels such as 90/30, 90/10 and 50/50 are Task conditions and are applied "
        "in addition to the Protocol 3 filter. "
        "Prefer query_behavior_data and query_ephys_units for flexible filtering, grouping, summary "
        "statistics and plots. Use query_assay_metrics for OFT and LDT metrics, including time in "
        "center/periphery, distance, speed, rearing, grooming, dark-box behavior and paired z-scores. "
        "For every Behavior versus OFT/LDT comparison, use correlate_behavior_assay so the app "
        "receives a real Plotly figure. Never write or embed SVG, HTML, Mermaid, ASCII charts, image "
        "markup or plotting code in the textual answer. If no plotting tool supports a requested "
        "figure, explain the limitation instead of simulating a plot in text. "
        "Break complex requests into several tool calls when useful. Check that "
        "the returned sample size and filters match the question before interpreting a result. "
        "Clearly distinguish observations from hypotheses, flag missing data and avoid causal claims "
        "from descriptive analyses. In the final answer, briefly state the population, filters, n, "
        "aggregation and metric used. If a chart_attached field is true, say that the chart is shown "
        "without recreating it in prose. Never claim to modify files, databases, code or the app: no "
        "write capability exists. Do not request arbitrary Python or shell execution. "
        "Treat the following INVIBE domain dictionary as authoritative. Use its definitions "
        "consistently, and call get_domain_knowledge whenever the user asks about task semantics, "
        "database relationships or interpretation rules: "
        f"{domain_knowledge} "
        f"The mouse currently selected in the navigation is {selected_mouse or 'none'}."
    )
    conversation = [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-16:]
    ]
    figures = []
    for _ in range(10):
        response = client.responses.create(
            model=model, instructions=instructions, input=conversation,
            tools=CHATBOT_TOOLS, store=False,
            reasoning={"effort": "high" if analysis_mode == "Advanced" else "medium"},
        )
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text or "No textual response was returned.", figures
        conversation.extend(
            item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            for item in response.output
        )
        for call in calls:
            try:
                arguments = json.loads(call.arguments)
                output, new_figures = execute_chatbot_tool(call.name, arguments)
                figures.extend(new_figures)
            except Exception as error:
                output = json.dumps({"error": str(error)}, ensure_ascii=False)
            conversation.append({
                "type": "function_call_output", "call_id": call.call_id,
                "output": output,
            })
    return "The analysis required too many tool calls. Please make the question more specific.", figures


def render_chatbot_view(selected_mouse=None):
    with st.container(border=True, key="chatbot_shell"):
        api_key = chatbot_api_key()
        if OpenAI is None:
            st.error("The OpenAI SDK is not installed. Run: pip install openai")
            return
        if not api_key:
            entered_key = st.text_input(
                "OpenAI API key", type="password",
                placeholder="sk-...", key="chatbot_api_key_input",
                help="Stored only in this Streamlit session.",
            )
            if entered_key:
                st.session_state["chatbot_api_key"] = entered_key.strip()
                st.rerun()
            st.caption("You can also set OPENAI_API_KEY on the server.")
            return

        if st.session_state.get("chatbot_tool_version") != CHATBOT_TOOL_VERSION:
            st.session_state["chatbot_tool_version"] = CHATBOT_TOOL_VERSION
            st.session_state["chatbot_messages"] = []
        analysis_mode = "Advanced"
        with st.container(key="chatbot_controls"):
            spacer_col, clear_col, astra_col = st.columns([7.4, 1, 1.8], gap="small")
            with clear_col:
                if st.button("Clear", key="chatbot_clear", width="stretch"):
                    st.session_state["chatbot_messages"] = []
                    st.rerun()
            with astra_col:
                st.toggle(
                    "GPT-6 Astra", key="chatbot_astra", value=False,
                )
        with st.container(height=430, border=False, key="chatbot_history"):
            for message_index, message in enumerate(st.session_state["chatbot_messages"]):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    for figure_index, figure in enumerate(message.get("figures", [])):
                        st.plotly_chart(
                            figure, width="stretch",
                            config={
                                "displayModeBar": "hover", "displaylogo": False,
                                "modeBarButtonsToRemove": [
                                    "select2d", "lasso2d", "zoom2d", "pan2d",
                                    "zoomIn2d", "zoomOut2d", "autoScale2d",
                                ],
                                "toImageButtonOptions": {
                                    "format": "png", "filename": "cazette_plot",
                                    "scale": 2,
                                },
                            },
                            key=f"chatbot_plot_{message_index}_{figure_index}",
                        )
            if st.session_state.pop("chatbot_pending_response", False):
                if st.session_state["chatbot_messages"] and st.session_state["chatbot_messages"][-1]["role"] == "user":
                    with st.spinner("Analyzing local data…"):
                        try:
                            answer, figures = run_chatbot_turn(
                                api_key, st.session_state["chatbot_messages"], selected_mouse,
                                analysis_mode, st.session_state.get("chatbot_astra", False),
                            )
                        except Exception as error:
                            answer, figures = f"OpenAI API error: {error}", []
                    st.session_state["chatbot_messages"].append({
                        "role": "assistant", "content": answer, "figures": figures,
                    })
                    st.rerun()
        prompt = st.chat_input("Ask about your data…")
        if prompt:
            st.session_state["chatbot_messages"].append({"role": "user", "content": prompt, "figures": []})
            st.session_state["chatbot_pending_response"] = True
            st.rerun()


def render_home_view():
    if not BEHAVIOR_DB.exists():
        st.error(f"Behavior database not found: {BEHAVIOR_DB}")
        return

    dataframe = load_behavior_database(
        str(BEHAVIOR_DB), source_signature(BEHAVIOR_DB)
    )
    dataframe = dataframe[dataframe["Mouse_ID"].apply(is_visible_mouse)].copy()
    if dataframe.empty:
        st.warning("No behavioral metadata found.")
        return

    ephys_inventory = (
        discover_ibl_channel_locations(EPHYS_ROOT)
        if EPHYS_ROOT.exists() else pd.DataFrame()
    )
    if not ephys_inventory.empty:
        ephys_inventory = ephys_inventory[
            ephys_inventory["mouse"].apply(is_visible_mouse)
        ].copy()
    openfield_inventory = discover_openfield_sessions(OPENFIELD_ROOT)
    ephys_by_mouse = {}
    ephys_latest_by_mouse = {}
    if not ephys_inventory.empty:
        ephys_by_mouse = (
            ephys_inventory.groupby("mouse")["date"].nunique().astype(int).to_dict()
        )
        dated_ephys = ephys_inventory.assign(
            _session_date=pd.to_datetime(ephys_inventory["date"], errors="coerce")
        )
        ephys_latest_by_mouse = (
            dated_ephys.groupby("mouse")["_session_date"].max().to_dict()
        )
    openfield_by_mouse = {}
    openfield_latest_by_mouse = {}
    if not openfield_inventory.empty:
        openfield_by_mouse = (
            openfield_inventory.groupby("mouse").size().astype(int).to_dict()
        )
        dated_openfield = openfield_inventory.assign(
            _session_date=pd.to_datetime(openfield_inventory["date"], errors="coerce")
        )
        openfield_latest_by_mouse = (
            dated_openfield.groupby("mouse")["_session_date"].max().to_dict()
        )

    behavior_latest_by_mouse = (
        dataframe.groupby("Mouse_ID")["Date"].max().to_dict()
    )
    behavior_mice = set(dataframe["Mouse_ID"].dropna().unique().tolist())
    ephys_mice = set(ephys_inventory["mouse"].unique().tolist()) if not ephys_inventory.empty else set()
    openfield_mice = set(openfield_inventory["mouse"].unique().tolist()) if not openfield_inventory.empty else set()
    mouse_ids = sorted(behavior_mice | ephys_mice | openfield_mice)
    latest_by_mouse = {}
    for mouse_id in mouse_ids:
        candidates = [
            value for value in (
                behavior_latest_by_mouse.get(mouse_id),
                ephys_latest_by_mouse.get(mouse_id),
                openfield_latest_by_mouse.get(mouse_id),
            )
            if pd.notna(value)
        ]
        latest_by_mouse[mouse_id] = max(candidates) if candidates else pd.NaT
    mouse_ids.sort(
        key=lambda mouse_id: (
            pd.Timestamp(latest_by_mouse[mouse_id]).value
            if pd.notna(latest_by_mouse[mouse_id]) else -1
        ),
        reverse=True,
    )
    total_ephys_sessions = int(
        ephys_inventory[["mouse", "date"]].drop_duplicates().shape[0]
    ) if not ephys_inventory.empty else 0
    total_openfield_sessions = int(len(openfield_inventory))
    global_protocol_counts = (
        pd.to_numeric(dataframe["Protocol"], errors="coerce")
        .dropna().astype(int).value_counts().to_dict()
    )
    with st.container(border=True, key="home_summary"):
        st.markdown(
            home_global_summary(
                len(mouse_ids), total_openfield_sessions,
                total_ephys_sessions, global_protocol_counts,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    with st.container(border=True, key="home_mouse_grid"):
        st.markdown('<div class="section-card-title">Mice</div>', unsafe_allow_html=True)
        card_columns = st.columns(4, gap="medium")
        for index, mouse_id in enumerate(mouse_ids):
            mouse_rows = dataframe[dataframe["Mouse_ID"] == mouse_id]
            mouse_latest = latest_by_mouse[mouse_id]
            latest_label = (
                mouse_latest.strftime("%Y-%m-%d") if pd.notna(mouse_latest) else "-"
            )
            behavior_count = len(mouse_rows)
            ephys_count = int(ephys_by_mouse.get(mouse_id, 0))
            openfield_count = int(openfield_by_mouse.get(mouse_id, 0))
            protocol_counts = (
                pd.to_numeric(mouse_rows["Protocol"], errors="coerce")
                .dropna().astype(int).value_counts().to_dict()
            )
            pie_html = home_session_pie(
                protocol_counts, ephys_count, openfield_count
            )
            safe_mouse = re.sub(r"[^a-zA-Z0-9_-]+", "_", mouse_id)
            with card_columns[index % 4]:
                with st.container(border=True, key=f"home_mouse_card_{safe_mouse}"):
                    st.markdown(
                        f'<div class="home-mouse-overview">'
                        f'<div class="home-mouse-copy">'
                        f'<div class="home-mouse-name">{html.escape(mouse_id)}</div>'
                        f'<div class="home-mouse-meta">{latest_label}</div>'
                        f'</div>{pie_html}</div>',
                        unsafe_allow_html=True,
                    )
                    behavior_column, ephys_column = st.columns(2, gap="small")
                    with behavior_column:
                        st.button(
                            f"Behavior **{behavior_count}**", key=f"home_behavior_{safe_mouse}",
                            on_click=open_mouse_view, args=(mouse_id, "Overview"),
                            width="stretch", disabled=behavior_count == 0,
                        )
                    with ephys_column:
                        st.button(
                            f"Ephys **{ephys_count}**", key=f"home_ephys_{safe_mouse}",
                            on_click=open_mouse_view,
                            args=(mouse_id, "Electrophysiology"),
                            width="stretch", disabled=ephys_count == 0,
                        )
                    openfield_column, info_column = st.columns(2, gap="small")
                    with openfield_column:
                        st.button(
                            f"Openfield **{openfield_count}**", key=f"home_openfield_{safe_mouse}",
                            on_click=open_mouse_view, args=(mouse_id, "Openfield"),
                            width="stretch", disabled=openfield_count == 0,
                        )
                    with info_column:
                        st.button(
                            "Info", key=f"home_info_{safe_mouse}",
                            on_click=open_mouse_view, args=(mouse_id, "Info"),
                            width="stretch",
                        )


def render_behavior_view(view_mode, mouse_id=None):
    if not BEHAVIOR_DB.exists():
        st.error(f"Behavior database not found: {BEHAVIOR_DB}")
        st.info("Set APP_BEHAVIOR_DB to use another Feather file.")
        return

    signature = source_signature(BEHAVIOR_DB)
    dataframe = load_behavior_database(str(BEHAVIOR_DB), signature)
    if dataframe.empty:
        st.warning("No behavioral metadata found.")
        return

    mouse_options = sorted(dataframe["Mouse_ID"].dropna().unique().tolist())
    if mouse_id not in mouse_options:
        latest_per_mouse = dataframe.groupby("Mouse_ID", dropna=False)["Date"].max().dropna()
        mouse_id = latest_per_mouse.idxmax() if not latest_per_mouse.empty else mouse_options[0]

    mouse_df = dataframe[dataframe["Mouse_ID"] == mouse_id].copy()

    with st.container(border=True, key="behavior_summary"):
        st.markdown(behavior_mouse_summary(mouse_id, mouse_df), unsafe_allow_html=True)

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    if view_mode == "Overview":
        with behavior_section_card("Training progression"):
            st.markdown(
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10%;padding:10px 0 0 0;">'
                '<div class="subplot-title behavior-subtitle">Valid bouts</div>'
                '<div class="subplot-title behavior-subtitle">Rewarded licks / min</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            plot_card(
                build_behavior_plot(str(BEHAVIOR_DB), signature, mouse_id, "training_outcomes")
            )

        with behavior_section_card("Performance structure"):
            col1, col2 = st.columns(2, gap="large")
            with col1:
                behavior_plot_card(
                    "Failure distribution",
                    build_behavior_plot(
                        str(BEHAVIOR_DB), signature, mouse_id, "histogram_kde_failures"
                    ),
                )
            with col2:
                behavior_plot_card(
                    "Failures across sessions",
                    build_behavior_plot(
                        str(BEHAVIOR_DB), signature, mouse_id, "kde_failures_by_session"
                    ),
                )
            behavior_plot_card(
                "Rewards and failures",
                build_behavior_plot(str(BEHAVIOR_DB), signature, mouse_id, "regression"),
            )

    mouse_df["label"] = (
        mouse_df["Date"].dt.strftime("%Y-%m-%d")
        + " · "
        + mouse_df["Protocol"].apply(
            lambda value: PROTOCOL_LABELS.get(int(value), f"Protocol {int(value)}")
            if pd.notna(value)
            else "-"
        )
        + " · v"
        + mouse_df["Version"].astype(str)
    )
    session_labels = mouse_df["label"].tolist()
    session_state_key = f"behavior_session_{mouse_id}"
    if st.session_state.get(session_state_key) not in session_labels:
        st.session_state[session_state_key] = session_labels[-1]

    with behavior_section_card("Session focus"):
        st.markdown('<div class="subplot-title anatomy-subtitle">Session</div>', unsafe_allow_html=True)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        selected_label = st.selectbox(
            "Session", session_labels, key=session_state_key,
            label_visibility="collapsed",
        )
        row = mouse_df[mouse_df["label"] == selected_label].iloc[0]
        session_key = (row["Date"].strftime("%Y-%m-%d"), str(row["Version"]))
        duration_value = pd.to_numeric(row.get("Session Dur"), errors="coerce")
        duration_minutes = float(duration_value) / 60 if pd.notna(duration_value) else np.nan
        valid = int(count_valid_bouts(row))
        total_value = pd.to_numeric(row.get("Number of Bouts"), errors="coerce")
        total = int(total_value) if pd.notna(total_value) else 0
        rewarded_value = pd.to_numeric(row.get("Number of Rewarded Licks"), errors="coerce")
        rewarded = float(rewarded_value) if pd.notna(rewarded_value) else 0.0
        rewarded_per_minute = rewarded / duration_minutes if duration_minutes > 0 else np.nan
        correct = int(np.asarray(row.get("Correct Bouts", []), dtype=bool).sum())
        success_rate = 100 * correct / valid if valid else np.nan
        st.markdown('<div class="subplot-title behavior-subtitle">Stats</div>', unsafe_allow_html=True)
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            quiet_metric("Protocol", PROTOCOL_LABELS.get(int(row["Protocol"]), "-") if pd.notna(row["Protocol"]) else "-", "#4D9FC7")
        with m2:
            quiet_metric("Probabilities", row.get("Probas", "-"), "#80639A")
        with m3:
            quiet_metric("Duration", f"{duration_minutes:.1f} min" if np.isfinite(duration_minutes) else "-", "#C48752")
        with m4:
            quiet_metric("Valid bouts", f"{valid} / {total}", "#4F9D7D")
        with m5:
            quiet_metric("Success rate", f"{success_rate:.1f}%" if np.isfinite(success_rate) else "-", "#4D9FC7")
        with m6:
            quiet_metric("Rewarded licks / min", f"{rewarded_per_minute:.2f}" if np.isfinite(rewarded_per_minute) else "-", "#4F9D7D")

        st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
        replay_row = load_behavior_video_session(
            str(BEHAVIOR_DB), signature, mouse_id, session_key[0], session_key[1]
        )
        replay_payload = (
            build_behavior_replay_payload(replay_row, mouse_id, session_key[0])
            if replay_row is not None else None
        )
        if replay_payload is None:
            st.info("Position and speed signals are not available for this session.")
        else:
            components.html(
                build_behavior_replay_html(replay_payload, behavior_only=True),
                height=455,
                scrolling=False,
            )

        st.markdown('<div class="subplot-title behavior-subtitle">Behavioral timeline</div>', unsafe_allow_html=True)
        plot_card(
            build_behavior_plot(
                str(BEHAVIOR_DB), signature, mouse_id, "session_timeline", session_key
            )
        )

        col1, col2 = st.columns(2, gap="large")
        with col1:
            behavior_plot_card(
                "Rewards vs failures",
                build_behavior_plot(
                    str(BEHAVIOR_DB), signature, mouse_id,
                    "session_rewards_failures", session_key,
                ),
            )
        with col2:
            behavior_plot_card(
                "Failure distribution",
                build_behavior_plot(
                    str(BEHAVIOR_DB), signature, mouse_id,
                    "session_failure_distribution", session_key,
                ),
            )


def _render_openfield_view_legacy(mouse=None, inventory=None):
    if inventory is None:
        inventory = discover_openfield_sessions(OPENFIELD_ROOT)
    if inventory.empty:
        st.warning(f"No open-field sessions found below {OPENFIELD_ROOT}")
        return

    mouse_options = sorted(inventory["mouse"].unique().tolist())
    if mouse not in mouse_options:
        mouse = mouse_options[0]
    mouse_inventory = inventory[inventory["mouse"] == mouse].copy().sort_values("date")
    session_options = mouse_inventory["date"].tolist()
    session_key = f"openfield_session_{mouse}"
    if st.session_state.get(session_key) not in session_options:
        st.session_state[session_key] = session_options[-1]

    with st.container(border=True, key="behavior_summary"):
        columns = st.columns(3)
        with columns[0]:
            quiet_metric("Mouse", mouse, "#278DBB")
        with columns[1]:
            quiet_metric("Openfield sessions", len(session_options), "#CF725E")
        with columns[2]:
            quiet_metric("Latest session", session_options[-1], "#31956E")

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    with behavior_section_card("Openfield session"):
        selected_date = st.selectbox(
            "Session", session_options, key=session_key,
            label_visibility="collapsed",
        )
        selected_row = mouse_inventory[mouse_inventory["date"] == selected_date].iloc[-1]
        info_columns = st.columns(3)
        with info_columns[0]:
            quiet_metric("Date", selected_date, "#278DBB")
        with info_columns[1]:
            quiet_metric("Videos", len(selected_row["videos"]), "#7654A8")
        with info_columns[2]:
            quiet_metric("Tracking files", len(selected_row["csv_files"]), "#31956E")

    oft_metrics = behavior_metrics_table("oft_sessions")
    selected_oft_metrics = pd.DataFrame()
    if not oft_metrics.empty:
        selected_oft_metrics = oft_metrics[
            (oft_metrics["mouse"].astype(str) == mouse)
            & (oft_metrics["oft_date"].dt.strftime("%Y-%m-%d") == selected_date)
        ]
    if not selected_oft_metrics.empty:
        metric_row = selected_oft_metrics.iloc[-1]
        with behavior_section_card("OFT metrics"):
            metric_specs = [
                ("Time in center", "OFT_ratio_time_center", "%", "#80639A"),
                ("Distance / min", "OFT_distance_cm_per_min", " cm", "#278DBB"),
                ("Center entries / min", "OFT_entries_center_per_min", "", "#31956E"),
                ("Median speed", "OFT_median_speed_cm_s", " cm/s", "#CF725E"),
                ("Rearing / min", "OFT_rearing_per_min", "", "#D18432"),
                ("Grooming time", "OFT_grooming_time_ratio", "%", "#6C7890"),
            ]
            metric_columns = st.columns(3, gap="medium")
            for index, (label, column, suffix, color) in enumerate(metric_specs):
                raw_value = pd.to_numeric(metric_row.get(column), errors="coerce")
                if pd.isna(raw_value):
                    value = "–"
                elif suffix == "%":
                    value = f"{100 * float(raw_value):.1f}%"
                elif suffix == " cm":
                    value = f"{float(raw_value):.1f} cm"
                else:
                    value = f"{float(raw_value):.2f}{suffix}"
                with metric_columns[index % 3]:
                    quiet_metric(label, value, color)

    ldt_metrics = behavior_metrics_table("ldt_sessions")
    mouse_ldt = (
        ldt_metrics[ldt_metrics["mouse"].astype(str) == mouse].copy()
        if not ldt_metrics.empty else pd.DataFrame()
    )
    if not mouse_ldt.empty:
        mouse_ldt = mouse_ldt.sort_values("ldt_date")
        ldt_dates = mouse_ldt["ldt_date"].dt.strftime("%Y-%m-%d").tolist()
        oft_timestamp = pd.Timestamp(selected_date)
        default_ldt = mouse_ldt.iloc[
            (mouse_ldt["ldt_date"] - oft_timestamp).abs().argmin()
        ]["ldt_date"].strftime("%Y-%m-%d")
        ldt_key = f"ldt_session_{mouse}"
        if st.session_state.get(ldt_key) not in ldt_dates:
            st.session_state[ldt_key] = default_ldt
        with behavior_section_card("Light-dark test"):
            selected_ldt_date = st.selectbox(
                "LDT session", ldt_dates, key=ldt_key, label_visibility="collapsed"
            )
            ldt_row = mouse_ldt[
                mouse_ldt["ldt_date"].dt.strftime("%Y-%m-%d") == selected_ldt_date
            ].iloc[-1]
            ldt_specs = [
                ("Time in dark", "LDT_dark_time_ratio", "%", "#80639A"),
                ("Dark entries / min", "LDT_dark_entries_per_min", "", "#278DBB"),
                ("Time in box", "LDT_in_box_time_ratio", "%", "#31956E"),
                ("Box exits / min", "LDT_box_exits_per_min", "", "#CF725E"),
            ]
            ldt_columns = st.columns(4, gap="medium")
            for column_area, (label, column, suffix, color) in zip(ldt_columns, ldt_specs):
                raw_value = pd.to_numeric(ldt_row.get(column), errors="coerce")
                if pd.isna(raw_value):
                    value = "–"
                elif suffix == "%":
                    value = f"{100 * float(raw_value):.1f}%"
                else:
                    value = f"{float(raw_value):.2f}"
                with column_area:
                    quiet_metric(label, value, color)

    videos = list(selected_row["videos"])
    if videos:
        with behavior_section_card("Video"):
            preferred_order = {"overlay": 0, "skeleton": 1, "clean": 2}
            videos.sort(key=lambda path: next(
                (rank for token, rank in preferred_order.items() if token in Path(path).stem.lower()),
                3,
            ))
            if selected_assay == "SIT":
                video_labels = {}
                for path in videos:
                    stem = Path(path).stem.lower()
                    phase = "Visitor" if "visitor" in stem else "Alone"
                    if "bis" in stem:
                        phase += " bis"
                    original_phase = phase
                    suffix = 2
                    while phase in video_labels:
                        phase = f"{original_phase} {suffix}"
                        suffix += 1
                    video_labels[phase] = path
            else:
                video_labels = {Path(path).name: path for path in videos}
            selected_video_label = st.selectbox(
                "Video source", list(video_labels), key=f"openfield_video_{mouse}_{selected_date}"
            )
            selected_video = Path(video_labels[selected_video_label])
            playback_video = cached_browser_video(selected_video)
            if playback_video is not None and playback_video.exists():
                st.video(str(playback_video), format="video/mp4")
            else:
                st.caption(
                    "No clean-skeleton video is available for this session."
                )
                if False and st.button(
                    "Prepare and play video", type="primary",
                    key=f"prepare_openfield_video_{mouse}_{selected_date}_{selected_video_label}",
                ):
                    try:
                        with st.spinner("Preparing the browser-compatible video…"):
                            prepare_browser_video(selected_video)
                        st.rerun()
                    except Exception as error:
                        st.error(f"Video conversion failed: {error}")
    else:
        st.info("No video found for this session.")


OPENFIELD_ASSAY_CONFIG = {
    "OFT": {"table": "oft_sessions", "date": "oft_date", "color": "#4D9FC7"},
    "LDT": {"table": "ldt_sessions", "date": "ldt_date", "color": "#80639A"},
    "SIT": {"table": "sit_sessions", "date": "sit_date", "color": "#D99A48"},
}


def openfield_summary(mouse_id, mouse_inventory):
    counts = (
        mouse_inventory.drop_duplicates(["assay", "date"])["assay"].value_counts()
    )
    blocks = [("SIT", "#D99A48"), ("OFT", "#4D9FC7"), ("LDT", "#80639A")]
    assay_html = "".join(
        '<div class="behavior-mouse-block">'
        '<div class="behavior-mouse-primary">'
        f'<strong style="color:{color}">{int(counts.get(assay, 0))}</strong>'
        f'<span>{assay} SESSIONS</span></div></div>'
        for assay, color in blocks
    )
    return (
        '<div class="behavior-mouse-band">'
        '<div class="behavior-mouse-block"><div class="behavior-mouse-primary">'
        f'<strong>{html.escape(str(mouse_id))}</strong><span>MOUSE</span>'
        f'</div></div>{assay_html}</div>'
    )


def openfield_metric_columns(frame, assay):
    prefix = f"{assay}_"
    return [
        column for column in frame.columns
        if str(column).startswith(prefix)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def format_openfield_metric(metric_name, raw_value):
    value = pd.to_numeric(raw_value, errors="coerce")
    if pd.isna(value):
        return "–"
    metric_lower = str(metric_name).lower()
    if "ratio" in metric_lower:
        return f"{100 * float(value):.1f}%"
    if "cm_per_min" in metric_lower:
        return f"{float(value):.1f} cm/min"
    if "cm_s" in metric_lower:
        return f"{float(value):.2f} cm/s"
    return f"{float(value):.2f}"


def render_openfield_metric_grid(values, metric_columns, assay):
    if not metric_columns:
        st.info(f"No {assay} metric is available yet.")
        return
    color = OPENFIELD_ASSAY_CONFIG[assay]["color"]
    cards = "".join(
        '<div class="openfield-metric-card">'
        f'<div class="openfield-metric-label" style="--metric-accent:{color}">'
        f'{html.escape(metric_display_name(metric_name))}</div>'
        f'<div class="openfield-metric-value">{html.escape(format_openfield_metric(metric_name, values.get(metric_name)))}</div>'
        '</div>'
        for metric_name in metric_columns
    )
    st.markdown(
        f'<div class="openfield-metric-grid">{cards}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_openfield_behavior_predictions(csv_path, file_size, modified_ns):
    frame = pd.read_csv(csv_path)
    if "label" not in frame:
        return pd.DataFrame(columns=["behavior", "frames", "percent", "confidence"])
    labels = frame["label"].astype(str).str.strip()
    if "motion_state" in frame:
        background = labels.str.casefold().eq("background")
        labels = labels.mask(
            background,
            frame["motion_state"].astype(str).str.strip().str.title(),
        )
    labels = labels.str.replace("_", " ", regex=False).str.title()
    confidence = pd.to_numeric(frame.get("score"), errors="coerce")
    summarized = (
        pd.DataFrame({"behavior": labels, "confidence": confidence})
        .dropna(subset=["behavior"])
        .groupby("behavior", as_index=False)
        .agg(frames=("behavior", "size"), confidence=("confidence", "mean"))
    )
    summarized["percent"] = 100 * summarized["frames"] / summarized["frames"].sum()
    return summarized.sort_values("percent")


def build_openfield_behavior_prediction_plot(csv_path):
    path = Path(csv_path)
    stats = path.stat()
    summary = load_openfield_behavior_predictions(
        str(path), stats.st_size, stats.st_mtime_ns
    )
    if summary.empty:
        return None
    palette = ["#80639A", "#4D9FC7", "#4F9D7D", "#D99A48", "#D06A7D", "#718094"]
    colors = [palette[index % len(palette)] for index in range(len(summary))]
    figure = go.Figure(go.Bar(
        x=summary["percent"], y=summary["behavior"], orientation="h",
        marker={"color": colors, "cornerradius": 5},
        customdata=np.column_stack((summary["confidence"], summary["frames"])),
        hovertemplate=(
            "%{y}<br>%{x:.1f}% of frames<br>"
            "Mean confidence %{customdata[0]:.2f}<extra></extra>"
        ),
    ))
    figure.update_layout(
        height=330, margin={"l": 8, "r": 18, "t": 8, "b": 42},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": "#223248", "size": 11},
        showlegend=False,
    )
    figure.update_xaxes(
        title="Time (% of frames)", rangemode="tozero",
        gridcolor="rgba(115,137,157,.16)", zeroline=False,
    )
    figure.update_yaxes(title=None, showgrid=False)
    return figure


@st.cache_data(show_spinner=False)
def load_openfield_keypoint_reconstruction(
    tracking_csv, tracking_size, tracking_mtime,
    behavior_csv, behavior_size, behavior_mtime, fps,
):
    tracking = pd.read_csv(tracking_csv, header=[0, 1, 2])
    bodyparts = [
        "nose", "neck", "torso", "tail_base", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
    ]
    scorer_names = {str(column[0]) for column in tracking.columns[1:]}
    scorer = next(iter(scorer_names), None)
    if scorer is None:
        return None
    frame_column = tracking.columns[0]
    frame_numbers = pd.to_numeric(tracking[frame_column], errors="coerce")
    effective_fps = max(float(fps or 25.0), 1.0)
    step = max(1, int(round(effective_fps / 12.0)))
    sample_indices = np.arange(0, len(tracking), step, dtype=int)
    sampled_frames = frame_numbers.iloc[sample_indices].fillna(
        pd.Series(sample_indices, index=frame_numbers.index[sample_indices])
    ).astype(int).to_numpy()
    points = {}
    all_x, all_y = [], []
    for bodypart in bodyparts:
        lookup = {
            str(column[2]).lower(): column
            for column in tracking.columns
            if str(column[0]) == scorer and str(column[1]).lower() == bodypart
        }
        if "x" not in lookup or "y" not in lookup:
            continue
        x_values = pd.to_numeric(tracking[lookup["x"]], errors="coerce")
        y_values = pd.to_numeric(tracking[lookup["y"]], errors="coerce")
        if "likelihood" in lookup:
            likelihood = pd.to_numeric(
                tracking[lookup["likelihood"]], errors="coerce"
            )
            valid = likelihood >= .35
            x_values = x_values.where(valid)
            y_values = y_values.where(valid)
        x_values = x_values.interpolate(limit_direction="both")
        y_values = y_values.interpolate(limit_direction="both")
        sampled_x = x_values.iloc[sample_indices].to_numpy(dtype=float)
        sampled_y = y_values.iloc[sample_indices].to_numpy(dtype=float)
        points[bodypart] = np.column_stack((sampled_x, sampled_y)).round(2).tolist()
        all_x.append(sampled_x)
        all_y.append(sampled_y)
    if not points:
        return None

    predictions = pd.read_csv(behavior_csv)
    predictions["frame"] = pd.to_numeric(predictions["frame"], errors="coerce")
    predictions = predictions.dropna(subset=["frame"]).copy()
    predictions["frame"] = predictions["frame"].astype(int)
    labels = predictions["label"].astype(str).str.strip()
    if "motion_state" in predictions:
        labels = labels.mask(
            labels.str.casefold().eq("background"),
            predictions["motion_state"].astype(str).str.strip().str.title(),
        )
    predictions["display_label"] = (
        labels.str.replace("_", " ", regex=False).str.title()
    )
    predictions["score"] = pd.to_numeric(predictions.get("score"), errors="coerce")
    prediction_lookup = (
        predictions.drop_duplicates("frame", keep="last")
        .set_index("frame").sort_index()
    )
    sampled_predictions = prediction_lookup.reindex(sampled_frames, method="nearest")
    behavior_labels = sampled_predictions["display_label"].fillna("Unknown").tolist()
    behavior_scores = sampled_predictions["score"].fillna(0).round(3).tolist()
    summary = (
        predictions.groupby("display_label", as_index=False)
        .size().rename(columns={"size": "frames"})
    )
    summary["percent"] = 100 * summary["frames"] / max(summary["frames"].sum(), 1)
    summary = summary.sort_values("percent", ascending=False)

    x_values = np.concatenate(all_x)
    y_values = np.concatenate(all_y)
    finite_x, finite_y = x_values[np.isfinite(x_values)], y_values[np.isfinite(y_values)]
    if not len(finite_x) or not len(finite_y):
        return None
    bounds = [
        float(np.nanpercentile(finite_x, .5)), float(np.nanpercentile(finite_x, 99.5)),
        float(np.nanpercentile(finite_y, .5)), float(np.nanpercentile(finite_y, 99.5)),
    ]
    return {
        "fps": effective_fps,
        "sampleFps": effective_fps / step,
        "frames": sampled_frames.tolist(),
        "time": (sampled_frames / effective_fps).round(3).tolist(),
        "duration": float(sampled_frames[-1] / effective_fps),
        "points": points,
        "labels": behavior_labels,
        "scores": behavior_scores,
        "bounds": bounds,
        "summary": [
            {"label": row.display_label, "percent": round(float(row.percent), 1)}
            for row in summary.itertuples(index=False)
        ],
    }


def build_openfield_reconstruction_html(payload):
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    colors_json = json.dumps(OPENFIELD_BEHAVIOR_COLORS, separators=(",", ":"))
    return f"""
    <!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
      *{{box-sizing:border-box}} html,body{{margin:0;background:transparent;color:#223248;font-family:Inter,Segoe UI,sans-serif}}
      .shell{{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,.75fr);gap:24px;align-items:start}}
      .title{{font-size:13px;font-weight:700;color:#344256;margin:0 0 14px}}
      .stage{{height:430px;border-radius:15px;overflow:hidden;background:#111B2A;box-shadow:inset 0 0 0 1px rgba(255,255,255,.06)}}
      canvas{{width:100%;height:100%;display:block}}
      .right{{height:430px;display:grid;grid-template-rows:auto 1fr;gap:18px}}
      .live{{min-height:112px;background:#F3F6F8;border-radius:14px;padding:16px 18px;display:flex;flex-direction:column;justify-content:center}}
      .eyebrow{{font-size:10px;font-weight:800;letter-spacing:.11em;color:#718094;margin-bottom:12px}}
      .live-row{{display:flex;align-items:center;gap:11px;min-width:0}}
      .pill{{display:inline-flex;align-items:center;justify-content:center;min-height:37px;padding:8px 15px;border-radius:11px;color:white;font-size:17px;font-weight:750;line-height:1;transition:background .12s ease;white-space:nowrap}}
      .confidence{{font-size:11px;color:#718094;white-space:nowrap}}
      .profile{{background:#F3F6F8;border-radius:14px;padding:15px 17px;overflow:hidden}}
      .bars{{display:flex;flex-direction:column;gap:9px;margin-top:13px}}
      .bar-row{{display:grid;grid-template-columns:88px minmax(0,1fr) 38px;align-items:center;gap:8px;font-size:10px;color:#536477}}
      .bar-label{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:right}}
      .track{{height:8px;border-radius:5px;background:#E1E7EC;overflow:hidden}}
      .fill{{height:100%;border-radius:5px}}
      .value{{font-variant-numeric:tabular-nums}}
      .controls{{grid-column:1/-1;display:grid;grid-template-columns:40px minmax(0,1fr) 92px;gap:13px;align-items:center;margin-top:4px}}
      button{{width:40px;height:38px;border:0;border-radius:11px;background:#223248;color:white;font-size:15px;cursor:pointer}}
      input[type=range]{{width:100%;accent-color:#F04F47}}
      .clock{{font-size:11px;font-weight:650;color:#536477;text-align:right;font-variant-numeric:tabular-nums}}
      @media(max-width:760px){{.shell{{grid-template-columns:1fr}}.right{{height:auto;grid-template-columns:1fr 1fr;grid-template-rows:auto}}.stage{{height:340px}}.controls{{grid-column:1}}}}
    </style></head><body>
    <div class="shell">
      <div><div class="title">Keypoint reconstruction</div><div class="stage"><canvas id="arena"></canvas></div></div>
      <div><div class="title">Behavior prediction</div><div class="right">
        <div class="live"><div class="eyebrow">LIVE BEHAVIOR</div><div class="live-row"><span class="pill" id="pill">–</span><span class="confidence" id="confidence">–</span></div></div>
        <div class="profile"><div class="eyebrow">SESSION PROFILE</div><div class="bars" id="bars"></div></div>
      </div></div>
      <div class="controls"><button id="play">▶</button><input id="seek" type="range" min="0" max="1000" value="0"><div class="clock" id="clock">00:00 / 00:00</div></div>
    </div>
    <script>
      const D={data_json}, COLORS={colors_json};
      const canvas=document.getElementById('arena'),ctx=canvas.getContext('2d'),pill=document.getElementById('pill'),confidence=document.getElementById('confidence'),seek=document.getElementById('seek'),clock=document.getElementById('clock'),play=document.getElementById('play');
      let playing=false,current=0,last=performance.now();
      const colorFor=l=>COLORS[l]||'#80639A';
      function resize(){{const r=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(r.width*d);canvas.height=Math.round(r.height*d);ctx.setTransform(d,0,0,d,0,0)}}
      function fmt(t){{t=Math.max(0,t);return String(Math.floor(t/60)).padStart(2,'0')+':'+String(Math.floor(t%60)).padStart(2,'0')}}
      function indexAt(t){{let lo=0,hi=D.time.length-1;while(lo<hi){{const m=Math.ceil((lo+hi)/2);if(D.time[m]<=t)lo=m;else hi=m-1}}return lo}}
      function lerp(a,b,q){{return a+(b-a)*q}}
      function sample(name,i,q){{const a=D.points[name]?.[i],b=D.points[name]?.[Math.min(i+1,D.time.length-1)]||a;if(!a)return null;return [lerp(a[0],b[0],q),lerp(a[1],b[1],q)]}}
      function rounded(x,y,w,h,r){{ctx.beginPath();ctx.roundRect(x,y,w,h,r);return ctx}}
      function draw(t){{
        const w=canvas.clientWidth,h=canvas.clientHeight;ctx.clearRect(0,0,w,h);const grad=ctx.createLinearGradient(0,0,w,h);grad.addColorStop(0,'#0C1624');grad.addColorStop(1,'#18283A');ctx.fillStyle=grad;ctx.fillRect(0,0,w,h);
        const pad=27,b=D.bounds,bw=Math.max(1,b[1]-b[0]),bh=Math.max(1,b[3]-b[2]),scale=Math.min((w-pad*2)/bw,(h-pad*2)/bh),ox=(w-bw*scale)/2-b[0]*scale,oy=(h-bh*scale)/2-b[2]*scale;
        const map=p=>p?[ox+p[0]*scale,oy+p[1]*scale]:null;ctx.fillStyle='#24384B';rounded(pad,pad,w-pad*2,h-pad*2,24).fill();ctx.strokeStyle='rgba(145,166,185,.20)';ctx.lineWidth=1.5;ctx.setLineDash([7,7]);ctx.strokeRect(w*.32,h*.32,w*.36,h*.36);ctx.setLineDash([]);
        const i=indexAt(t),next=Math.min(i+1,D.time.length-1),span=Math.max(.001,D.time[next]-D.time[i]),q=Math.max(0,Math.min(1,(t-D.time[i])/span));
        const P=name=>map(sample(name,i,q)),torso=P('torso'),neck=P('neck'),nose=P('nose'),tail=P('tail_base');if(!torso||!neck||!nose||!tail)return;
        ctx.save();ctx.globalAlpha=.18;ctx.strokeStyle='#70C7D8';ctx.lineWidth=2;ctx.beginPath();for(let k=Math.max(0,i-90);k<=i;k+=3){{const p=map(D.points.torso?.[k]);if(p)k===Math.max(0,i-90)?ctx.moveTo(...p):ctx.lineTo(...p)}}ctx.stroke();ctx.restore();
        const angle=Math.atan2(nose[1]-tail[1],nose[0]-tail[0]),bodyLen=Math.max(34,Math.min(74,Math.hypot(neck[0]-tail[0],neck[1]-tail[1]))),bodyW=bodyLen*.43,headLen=Math.max(16,Math.min(30,Math.hypot(nose[0]-neck[0],nose[1]-neck[1])));
        ctx.save();ctx.translate(torso[0],torso[1]);ctx.rotate(angle);ctx.lineCap='round';ctx.strokeStyle='#AFC0CF';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(-bodyLen*.48,0);ctx.bezierCurveTo(-bodyLen*.72,-bodyW*.15,-bodyLen*.83,bodyW*.50,-bodyLen*1.08,bodyW*.38+Math.sin(t*3)*3);ctx.stroke();
        ctx.shadowColor='rgba(72,184,208,.35)';ctx.shadowBlur=18;ctx.fillStyle='#DDE7EE';ctx.beginPath();ctx.ellipse(-bodyLen*.05,0,bodyLen*.48,bodyW*.52,0,0,Math.PI*2);ctx.fill();ctx.shadowBlur=7;ctx.fillStyle='#F5F8FA';ctx.beginPath();ctx.ellipse(bodyLen*.43,0,headLen*.62,headLen*.46,0,0,Math.PI*2);ctx.fill();ctx.restore();
        for(const earName of ['left_ear','right_ear']){{const e=P(earName);if(e){{ctx.fillStyle='#D8A5B3';ctx.beginPath();ctx.arc(e[0],e[1],Math.max(4,bodyW*.16),0,Math.PI*2);ctx.fill();ctx.strokeStyle='#F0CDD5';ctx.stroke()}}}}
        ctx.fillStyle='#D98296';ctx.beginPath();ctx.arc(nose[0],nose[1],3.2,0,Math.PI*2);ctx.fill();const eyeX=lerp(neck[0],nose[0],.58),eyeY=lerp(neck[1],nose[1],.58);ctx.fillStyle='#172234';ctx.beginPath();ctx.arc(eyeX+Math.cos(angle+Math.PI/2)*5,eyeY+Math.sin(angle+Math.PI/2)*5,2.1,0,Math.PI*2);ctx.fill();
        for(const foot of ['left_shoulder','right_shoulder','left_hip','right_hip']){{const p=P(foot);if(p){{ctx.fillStyle='rgba(199,212,222,.86)';ctx.beginPath();ctx.ellipse(p[0],p[1],5,2.7,angle,0,Math.PI*2);ctx.fill()}}}}
        const label=D.labels[i]||'Unknown';pill.textContent=label;pill.style.background=colorFor(label);confidence.textContent=Math.round((D.scores[i]||0)*100)+'% confidence';
      }}
      D.summary.forEach(item=>{{const row=document.createElement('div');row.className='bar-row';row.innerHTML=`<div class="bar-label">${{item.label}}</div><div class="track"><div class="fill" style="width:${{item.percent}}%;background:${{colorFor(item.label)}}"></div></div><div class="value">${{item.percent.toFixed(1)}}%</div>`;document.getElementById('bars').appendChild(row)}});
      function tick(now){{if(playing){{current+=Math.min(.1,(now-last)/1000);if(current>=D.duration){{current=0;playing=false;play.textContent='▶'}}}}last=now;seek.value=Math.round(current/D.duration*1000);clock.textContent=fmt(current)+' / '+fmt(D.duration);draw(current);requestAnimationFrame(tick)}}
      play.onclick=()=>{{playing=!playing;play.textContent=playing?'❚❚':'▶';last=performance.now()}};seek.oninput=()=>{{current=Number(seek.value)/1000*D.duration;draw(current)}};window.addEventListener('resize',()=>{{resize();draw(current)}});resize();requestAnimationFrame(tick);
    </script></body></html>
    """


@st.cache_data(show_spinner=False)
def is_browser_compatible_video(video_path, file_size, modified_ns):
    source = Path(video_path)
    if imageio_ffmpeg is None:
        return source.suffix.lower() == ".webm"
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(source)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    codec_match = re.search(r"Video:\s*([^,\s]+)", result.stderr, re.IGNORECASE)
    codec = codec_match.group(1).lower() if codec_match else ""
    return (
        source.suffix.lower() == ".mp4" and codec in {"h264", "av1"}
    ) or (
        source.suffix.lower() == ".webm" and codec in {"vp8", "vp9", "av1"}
    )


OPENFIELD_BEHAVIOR_COLORS = {
    "Static": "#6C7890",
    "Walking": "#4D9FC7",
    "Running": "#2B7FA8",
    "Rearing": "#80639A",
    "Wall Rearing": "#B06A9A",
    "Grooming": "#4F9D7D",
    "Freezing": "#D99A48",
}


@st.cache_data(show_spinner=False)
def load_openfield_behavior_sidecar(csv_path, file_size, modified_ns, fps):
    frame = pd.read_csv(csv_path)
    if frame.empty or "frame" not in frame or "label" not in frame:
        return None
    labels = frame["label"].astype(str).str.strip()
    if "motion_state" in frame:
        labels = labels.mask(
            labels.str.casefold().eq("background"),
            frame["motion_state"].astype(str).str.strip().str.title(),
        )
    labels = labels.str.replace("_", " ", regex=False).str.title()
    timeline = pd.DataFrame({
        "frame": pd.to_numeric(frame["frame"], errors="coerce"),
        "label": labels,
        "score": pd.to_numeric(frame.get("score"), errors="coerce"),
    }).dropna(subset=["frame", "label"]).sort_values("frame")
    if timeline.empty:
        return None
    timeline["frame"] = timeline["frame"].astype(int)
    timeline["score_step"] = (timeline["score"] * 100).round()
    timeline["run"] = (
        timeline["label"].ne(timeline["label"].shift())
        | timeline["score_step"].ne(timeline["score_step"].shift())
    ).cumsum()
    runs = timeline.groupby("run", as_index=False).agg(
        start_frame=("frame", "min"), end_frame=("frame", "max"),
        label=("label", "first"), score=("score_step", "first"),
    )
    effective_fps = max(float(fps or 25.0), 1.0)
    summary = (
        timeline["label"].value_counts(normalize=True).mul(100)
        .rename_axis("label").reset_index(name="percent")
    )
    return {
        "runs": [
            {
                "start": round(float(row.start_frame) / effective_fps, 3),
                "end": round(float(row.end_frame + 1) / effective_fps, 3),
                "label": str(row.label),
                "score": int(row.score) if pd.notna(row.score) else None,
            }
            for row in runs.itertuples(index=False)
        ],
        "summary": [
            {"label": str(row.label), "percent": round(float(row.percent), 1)}
            for row in summary.itertuples(index=False)
        ],
    }


def build_openfield_behavior_sidecar_html(payload):
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    colors_json = json.dumps(OPENFIELD_BEHAVIOR_COLORS, separators=(",", ":"))
    bars = "".join(
        f'<div class="bar-row"><span>{html.escape(item["label"])}</span>'
        f'<div class="track"><i style="width:{item["percent"]:.1f}%;background:{OPENFIELD_BEHAVIOR_COLORS.get(item["label"], "#718094")}"></i></div>'
        f'<b>{item["percent"]:.1f}%</b></div>'
        for item in payload["summary"]
    )
    return f"""
    <!doctype html><html><head><meta charset="utf-8"><style>
      *{{box-sizing:border-box}}html,body{{margin:0;background:transparent;color:#223248;font-family:Inter,Segoe UI,sans-serif}}
      .panel{{height:520px;display:flex;flex-direction:column;gap:14px;padding:2px 0}}
      .live,.profile{{background:#F3F6F8;border-radius:15px}}
      .live{{height:108px;min-height:108px;padding:16px 22px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;transition:background .12s ease}}
      .eyebrow{{font-size:11px;font-weight:750;letter-spacing:.1em;color:#718094;margin-bottom:15px}}
      .current{{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:11px}}
      .pill{{color:#fff;font-size:29px;font-weight:750;line-height:1.05;text-shadow:0 1px 2px rgba(20,35,50,.16)}}
      .confidence{{font-size:12px;color:rgba(255,255,255,.86);white-space:nowrap}}
      .profile{{flex:1;padding:20px 24px;min-height:0}}
      .profile-body{{display:flex;flex-direction:column;justify-content:center;gap:13px;height:calc(100% - 28px)}}
      .bar-row{{display:grid;grid-template-columns:92px minmax(0,1fr) 43px;gap:10px;align-items:center;font-size:11px;color:#536477}}
      .bar-row>span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right}}
      .bar-row>b{{font-size:10px;font-weight:650;color:#718094;font-variant-numeric:tabular-nums}}
      .track{{height:13px;border-radius:7px;background:#E1E7EC;overflow:hidden}}
      .track i{{display:block;height:100%;min-width:2px;border-radius:7px}}
      @media(max-width:600px){{.panel{{height:auto}}}}
    </style></head><body><div class="panel">
      <div class="live" id="live"><div class="current"><span class="pill" id="pill">Ready</span><span class="confidence" id="confidence"></span></div></div>
      <div class="profile"><div class="eyebrow">SESSION PROFILE</div><div class="profile-body">{bars}</div></div>
    </div><script>
      const D={data_json}, COLORS={colors_json}, live=document.getElementById('live'), pill=document.getElementById('pill'), confidence=document.getElementById('confidence');
      let video=null,lastIndex=-1;
      function locateVideo(){{try{{const items=window.parent.document.querySelectorAll('video');video=items.length?items[items.length-1]:null}}catch(error){{video=null}}}}
      function indexAt(t){{let lo=0,hi=D.runs.length-1,best=0;while(lo<=hi){{const mid=(lo+hi)>>1;if(D.runs[mid].start<=t){{best=mid;lo=mid+1}}else hi=mid-1}}return best}}
      function update(){{if(!video||!video.isConnected)locateVideo();if(!video||!D.runs.length)return;const index=indexAt(video.currentTime||0);if(index===lastIndex)return;lastIndex=index;const item=D.runs[index];pill.textContent=item.label;live.style.background=COLORS[item.label]||'#718094';confidence.textContent=item.score===null?'':item.score+'% confidence'}}
      locateVideo();update();setInterval(update,100);
    </script></body></html>
    """


def ass_color(hex_color):
    color = hex_color.lstrip("#")
    red, green, blue = color[0:2], color[2:4], color[4:6]
    return f"&H00{blue}{green}{red}&"


def ass_timestamp(seconds):
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours}:{minutes:02d}:{remainder:05.2f}"


def write_behavior_prediction_ass(csv_path, output_path, fps):
    frame = pd.read_csv(csv_path)
    if frame.empty or "frame" not in frame or "label" not in frame:
        return False
    labels = frame["label"].astype(str).str.strip()
    if "motion_state" in frame:
        labels = labels.mask(
            labels.str.casefold().eq("background"),
            frame["motion_state"].astype(str).str.strip().str.title(),
        )
    labels = labels.str.replace("_", " ", regex=False).str.title()
    frame_numbers = pd.to_numeric(frame["frame"], errors="coerce")
    scores = (
        pd.to_numeric(frame["score"], errors="coerce")
        if "score" in frame
        else pd.Series(np.nan, index=frame.index)
    )
    valid = frame_numbers.notna() & labels.notna()
    timeline = pd.DataFrame({
        "frame": frame_numbers[valid].astype(int),
        "label": labels[valid],
        "score": scores[valid],
    }).sort_values("frame")
    if timeline.empty:
        return False
    # Update the confidence badge in readable 5-point steps while preserving
    # exact frame timing for every behavior transition.
    timeline["score_step"] = (timeline["score"] * 20).round() / 20
    timeline["run"] = (
        timeline["label"].ne(timeline["label"].shift())
        | timeline["score_step"].ne(timeline["score_step"].shift())
    ).cumsum()
    runs = timeline.groupby("run", as_index=False).agg(
        first_frame=("frame", "min"), last_frame=("frame", "max"),
        label=("label", "first"), score=("score_step", "first"),
    )
    total_end = ass_timestamp((runs["last_frame"].max() + 1) / max(float(fps or 25.0), 1.0))
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 720
PlayResY: 480
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Header,Arial,17,&H00718094,&H00718094,&H00F3F6F8,&H00F3F6F8,1,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1
Style: Label,Arial,25,&H00FFFFFF,&H00FFFFFF,&H0080639A,&H0080639A,1,0,0,0,100,100,0,0,3,8,0,5,0,0,0,1
Style: Confidence,Arial,14,&H00718094,&H00718094,&H00F3F6F8,&H00F3F6F8,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    dialogue = [
        f"Dialogue: 0,0:00:00.00,{total_end},Header,,0,0,0,,"
        r"{\an5\pos(600,45)}LIVE BEHAVIOR"
    ]
    effective_fps = max(float(fps or 25.0), 1.0)
    for row in runs.itertuples(index=False):
        start = ass_timestamp(row.first_frame / effective_fps)
        end = ass_timestamp((row.last_frame + 1) / effective_fps)
        label = str(row.label).replace("\\", "").replace("{", "").replace("}", "")
        confidence = f"{100 * row.score:.0f}% confidence" if pd.notna(row.score) else ""
        behavior_color = ass_color(OPENFIELD_BEHAVIOR_COLORS.get(label, "#80639A"))
        dialogue.append(
            f"Dialogue: 0,{start},{end},Label,,0,0,0,,"
            rf"{{\an5\pos(560,92)\3c{behavior_color}}}{label}"
        )
        dialogue.append(
            f"Dialogue: 0,{start},{end},Confidence,,0,0,0,,"
            rf"{{\an5\pos(670,92)}}{confidence}"
        )
    Path(output_path).write_text(header + "\n".join(dialogue), encoding="utf-8")
    return True


def write_behavior_summary_panel(csv_path, output_path):
    path = Path(csv_path)
    stats = path.stat()
    summary = load_openfield_behavior_predictions(
        str(path), stats.st_size, stats.st_mtime_ns
    )
    if summary.empty:
        return False
    figure, axis = plt.subplots(figsize=(3.0, 3.15), dpi=100)
    figure.patch.set_facecolor("#F3F6F8")
    axis.set_facecolor("#F3F6F8")
    colors = [
        OPENFIELD_BEHAVIOR_COLORS.get(label, "#80639A")
        for label in summary["behavior"]
    ]
    axis.barh(summary["behavior"], summary["percent"], color=colors, height=.58)
    axis.set_title("SESSION PROFILE", loc="left", fontsize=9, fontweight="bold", color="#718094", pad=10)
    axis.set_xlabel("Time (% of frames)", fontsize=8, color="#718094", labelpad=7)
    axis.tick_params(axis="x", colors="#718094", labelsize=7, length=0)
    axis.tick_params(axis="y", colors="#223248", labelsize=8, length=0)
    axis.grid(axis="x", color="#DDE5EB", linewidth=.7)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(False)
    maximum = max(float(summary["percent"].max()), 1.0)
    axis.set_xlim(0, maximum * 1.16)
    for y_value, percent in enumerate(summary["percent"]):
        axis.text(
            percent + maximum * .025, y_value, f"{percent:.1f}%",
            va="center", ha="left", fontsize=7, color="#536477",
        )
    figure.tight_layout(pad=1.2)
    figure.savefig(output_path, facecolor="#F3F6F8", bbox_inches="tight", pad_inches=.08)
    plt.close(figure)
    return True


def cached_browser_video(video_path, predictions_csv=None):
    """Return the source when compatible, otherwise its persistent H.264 proxy path."""
    source = Path(video_path)
    try:
        stats = source.stat()
        signature = f"{source.resolve()}|{stats.st_size}|{stats.st_mtime_ns}"
    except OSError:
        return None
    if predictions_csv:
        prediction_path = Path(predictions_csv)
        try:
            prediction_stats = prediction_path.stat()
            signature += (
                f"|{prediction_path.resolve()}|{prediction_stats.st_size}"
                f"|{prediction_stats.st_mtime_ns}|realtime-fast-v4"
            )
        except OSError:
            predictions_csv = None
    if not predictions_csv and is_browser_compatible_video(
        str(source), stats.st_size, stats.st_mtime_ns
    ):
        return source
    cache_root = Path(os.getenv("LOCALAPPDATA", str(Path.cwd()))) / "INVIBE_dashboard" / "video_cache"
    return cache_root / f"{hashlib.sha1(signature.encode('utf-8')).hexdigest()}.mp4"


@st.cache_resource(show_spinner=False)
def fast_h264_encoder_arguments():
    """Prefer a working GPU encoder and fall back to the fastest CPU preset."""
    if imageio_ffmpeg is None:
        return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    candidates = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "30", "-b:v", "0"]),
        ("h264_qsv", ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "30"]),
    ]
    for encoder, arguments in candidates:
        probe = subprocess.run(
            [
                executable, "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=size=64x64:rate=1",
                "-frames:v", "1", "-c:v", encoder, "-f", "null", "-",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if probe.returncode == 0:
            return arguments
    return ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]


def prepare_browser_video(video_path, predictions_csv=None, fps=None):
    """Create a compact H.264 proxy for AVI recordings used by the web player."""
    if imageio_ffmpeg is None:
        raise RuntimeError("imageio-ffmpeg is not installed. Run: pip install imageio-ffmpeg")
    source = Path(video_path)
    output = cached_browser_video(source, predictions_csv)
    if output is None:
        raise FileNotFoundError(source)
    if output.exists() and output.stat().st_size > 0:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial.mp4")
    video_filter = "scale=480:-2:force_original_aspect_ratio=decrease,fps=12"
    subtitle_path = None
    panel_path = None
    if predictions_csv:
        subtitle_path = output.with_suffix(".ass")
        panel_path = output.with_suffix(".summary.png")
        has_subtitles = write_behavior_prediction_ass(
            predictions_csv, subtitle_path, fps
        )
        has_panel = write_behavior_summary_panel(predictions_csv, panel_path)
    else:
        has_subtitles = has_panel = False
    if has_subtitles and has_panel:
        filter_complex = (
            f"[0:v]{video_filter},pad=720:ih:0:0:color=0xF3F6F8[base];"
            "[1:v]scale=240:245[chart];"
            "[base][chart]overlay=480:H-h-10[charted];"
            f"[charted]ass={subtitle_path.name}[outv]"
        )
        encoder_arguments = fast_h264_encoder_arguments()
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source),
            "-loop", "1", "-framerate", "12", "-i", panel_path.name,
            "-filter_complex", filter_complex, "-map", "[outv]", "-an",
            *encoder_arguments,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest",
            str(temporary),
        ]
    else:
        encoder_arguments = fast_h264_encoder_arguments()
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source),
            "-map", "0:v:0", "-an", "-vf", video_filter,
            *encoder_arguments,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ]
    try:
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True, cwd=str(output.parent),
        )
        temporary.replace(output)
        if subtitle_path is not None:
            subtitle_path.unlink(missing_ok=True)
        if panel_path is not None:
            panel_path.unlink(missing_ok=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        if subtitle_path is not None:
            subtitle_path.unlink(missing_ok=True)
        if panel_path is not None:
            panel_path.unlink(missing_ok=True)
        raise
    return output


def openfield_session_metrics(mouse, assay, selected_date):
    config = OPENFIELD_ASSAY_CONFIG[assay]
    frame = behavior_metrics_table(config["table"])
    if frame.empty or "mouse" not in frame or config["date"] not in frame:
        return frame, None
    selected = frame[
        (frame["mouse"].astype(str) == str(mouse))
        & (frame[config["date"]].dt.strftime("%Y-%m-%d") == selected_date)
    ]
    return frame, selected.iloc[-1] if not selected.empty else None


def render_openfield_view(mouse=None, inventory=None):
    if inventory is None:
        inventory = discover_openfield_sessions(OPENFIELD_ROOT)
    if inventory.empty:
        st.warning(f"No OFT, LDT or SIT sessions found below {OPENFIELD_ROOT}")
        return

    mouse_options = sorted(inventory["mouse"].unique().tolist())
    if mouse not in mouse_options:
        mouse = mouse_options[0]
    mouse_inventory = (
        inventory[inventory["mouse"] == mouse]
        .copy().sort_values(["date", "assay"])
    )

    with st.container(border=True, key="behavior_summary"):
        st.markdown(openfield_summary(mouse, mouse_inventory), unsafe_allow_html=True)

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    with behavior_section_card("Global metrics"):
        metric_tabs = st.tabs(["OFT", "LDT", "SIT"])
        for metric_tab, assay in zip(metric_tabs, ("OFT", "LDT", "SIT")):
            with metric_tab:
                config = OPENFIELD_ASSAY_CONFIG[assay]
                frame = behavior_metrics_table(config["table"])
                mouse_frame = (
                    frame[frame["mouse"].astype(str) == str(mouse)].copy()
                    if not frame.empty and "mouse" in frame else pd.DataFrame()
                )
                metric_columns = openfield_metric_columns(mouse_frame, assay)
                if mouse_frame.empty or not metric_columns:
                    if assay == "SIT":
                        st.info(
                            "SIT videos are indexed, but no SIT metrics table is present "
                            "in behavior_metrics.sqlite yet."
                        )
                    else:
                        st.info(f"No {assay} metrics are available for {mouse}.")
                else:
                    mean_values = mouse_frame[metric_columns].apply(
                        pd.to_numeric, errors="coerce"
                    ).mean()
                    render_openfield_metric_grid(mean_values, metric_columns, assay)

    with behavior_section_card("Session focus"):
        mouse_inventory["label"] = (
            mouse_inventory["assay"] + "  |  " + mouse_inventory["date"]
        )
        session_labels = mouse_inventory["label"].tolist()
        session_key = f"openfield_session_{mouse}"
        if st.session_state.get(session_key) not in session_labels:
            st.session_state[session_key] = session_labels[-1]
        selected_label = st.selectbox(
            "Session", session_labels, key=session_key,
            label_visibility="collapsed",
        )
        selected_row = mouse_inventory[
            mouse_inventory["label"] == selected_label
        ].iloc[-1]
        selected_assay = str(selected_row["assay"])
        selected_date = str(selected_row["date"])
        metric_frame, metric_row = openfield_session_metrics(
            mouse, selected_assay, selected_date
        )

        st.markdown(
            '<div class="subplot-title behavior-subtitle">Infos</div>',
            unsafe_allow_html=True,
        )
        stat_columns = st.columns(5, gap="medium")
        duration = pd.to_numeric(
            metric_row.get("duration_min") if metric_row is not None else np.nan,
            errors="coerce",
        )
        fps = pd.to_numeric(
            metric_row.get("fps") if metric_row is not None else np.nan,
            errors="coerce",
        )
        stat_specs = [
            ("Test", selected_assay, OPENFIELD_ASSAY_CONFIG[selected_assay]["color"]),
            ("Date", selected_date, "#4D9FC7"),
            ("Duration", f"{duration:.1f} min" if pd.notna(duration) else "–", "#80639A"),
            ("Frame rate", f"{fps:.1f} fps" if pd.notna(fps) else "–", "#4F9D7D"),
            (
                "Phases" if selected_assay == "SIT" else "Videos",
                str(len(selected_row["videos"])), "#D99A48",
            ),
        ]
        for column_area, (label, value, color) in zip(stat_columns, stat_specs):
            with column_area:
                quiet_metric(label, value, color)

        session_metric_columns = openfield_metric_columns(metric_frame, selected_assay)
        if metric_row is not None and session_metric_columns:
            st.markdown(
                '<div class="subplot-title behavior-subtitle">Session metrics</div>',
                unsafe_allow_html=True,
            )
            render_openfield_metric_grid(
                metric_row, session_metric_columns, selected_assay
            )

        videos = list(selected_row["videos"])
        browser_video = next(
            (
                Path(path) for path in videos
                if Path(path).stem.lower().endswith("_clean_skeleton_h264")
            ),
            None,
        )
        prediction_csv = next(
            (
                path for path in selected_row["csv_files"]
                if "behavior" in Path(path).stem.lower()
            ),
            None,
        )
        if browser_video is not None and browser_video.exists():
            st.markdown(
                '<div class="subplot-title behavior-subtitle">Video</div>',
                unsafe_allow_html=True,
            )
            video_column, behavior_column = st.columns([1, 1], gap="large")
            with video_column:
                st.video(str(browser_video), format="video/mp4")
            with behavior_column:
                if prediction_csv:
                    prediction_path = Path(prediction_csv)
                    prediction_stats = prediction_path.stat()
                    sidecar_payload = load_openfield_behavior_sidecar(
                        str(prediction_path), prediction_stats.st_size,
                        prediction_stats.st_mtime_ns,
                        float(fps) if pd.notna(fps) else 50.0,
                    )
                    if sidecar_payload:
                        components.html(
                            build_openfield_behavior_sidecar_html(sidecar_payload),
                            height=530, scrolling=False,
                        )
        if False and videos:
            clean_skeleton = next(
                (
                    Path(path) for path in videos
                    if "clean" in Path(path).stem.lower()
                    and "skeleton" in Path(path).stem.lower()
                ),
                None,
            )
            # SIT files are still raw: keep Alone + Visitor as one session and
            # display Visitor by default until a clean-skeleton export exists.
            if clean_skeleton is None and selected_assay == "SIT":
                clean_skeleton = next(
                    (Path(path) for path in videos if "visitor" in Path(path).stem.lower()),
                    Path(videos[0]),
                )
            selected_video = clean_skeleton
            prediction_csv = next(
                (
                    path for path in selected_row["csv_files"]
                    if "behavior" in Path(path).stem.lower()
                ),
                None,
            )
            playback_video = (
                cached_browser_video(selected_video, prediction_csv)
                if selected_video else None
            )
            if selected_video is not None and (
                playback_video is None or not playback_video.exists()
            ):
                try:
                    with st.spinner("Preparing the session video..."):
                        playback_video = prepare_browser_video(
                            selected_video, prediction_csv,
                            float(fps) if pd.notna(fps) else 25.0,
                        )
                except Exception as error:
                    playback_video = None
                    st.error(f"Video preparation failed: {error}")
            if playback_video is not None and playback_video.exists():
                left_space, video_column, right_space = st.columns([1, 3, 1])
                with video_column:
                    st.video(str(playback_video), format="video/mp4")
            else:
                st.caption(
                    "No clean-skeleton video is available for this session."
                )
                if False and st.button(
                    "Prepare and play video", type="primary",
                    key=f"prepare_openfield_video_{mouse}_{selected_date}_{selected_video_label}",
                ):
                    try:
                        with st.spinner("Preparing the browser-compatible video…"):
                            prepare_browser_video(selected_video)
                        st.rerun()
                    except Exception as error:
                        st.error(f"Video conversion failed: {error}")


def render_ephys_view(mouse=None, inventory=None):
    if not EPHYS_ROOT.exists():
        st.error(f"Electrophysiology root not found: {EPHYS_ROOT}")
        st.info("Set APP_EPHYS_ROOT to the folder containing the mouse directories.")
        return

    if inventory is None:
        inventory = discover_ibl_channel_locations(EPHYS_ROOT)
    if inventory.empty:
        st.warning(f"No IBL or Brainreg probe trajectories found below {EPHYS_ROOT}")
        return

    mouse_options = sorted(inventory["mouse"].unique().tolist())
    if mouse not in mouse_options:
        mouse = mouse_options[0]

    mouse_inventory = inventory[inventory["mouse"] == mouse].copy()
    session_options = sorted(mouse_inventory["date"].unique().tolist())
    selected_sessions = session_options
    selected_inventory = mouse_inventory

    track_rows = selected_inventory[
        ["mouse", "date", "probe", "path", "session_dir", "source"]
    ].to_dict("records")
    tracks = load_probe_tracks(track_rows, channel_locations_signature(selected_inventory))
    resolved_probe_targets = resolve_probe_targets(tracks)

    behavior_counts = {}
    behavior_bout_starts = {}
    behavior_on_site_arrivals = {}
    if BEHAVIOR_DB.exists():
        try:
            counts_df = load_behavior_session_counts(
                str(BEHAVIOR_DB),
                source_signature(BEHAVIOR_DB),
            )
            counts_df = counts_df[counts_df["Mouse_ID"] == mouse]
            behavior_counts = dict(zip(counts_df["Date"], counts_df["valid_bouts"]))
            behavior_bout_starts = dict(
                zip(counts_df["Date"], counts_df["bout_start_times"])
            )
            behavior_on_site_arrivals = dict(
                zip(counts_df["Date"], counts_df["on_site_arrival_times"])
            )
        except (OSError, ValueError, KeyError):
            behavior_counts = {}
            behavior_bout_starts = {}
            behavior_on_site_arrivals = {}

    atlas_mesh_path = ATLAS_DIR / "meshes" / "997.obj"
    annotation_path = ATLAS_DIR / "annotation.tiff"
    if not atlas_mesh_path.exists():
        st.error(f"Allen atlas mesh not found: {atlas_mesh_path}")
        st.info("Set APP_ATLAS_DIR to the local allen_mouse_25um_v1.2 atlas directory.")
        return
    brain_mesh = load_obj_mesh(str(atlas_mesh_path), atlas_mesh_signature(atlas_mesh_path))
    region_meshes = {}
    for region, structure_id in {"MOs": 993, "STR": 477, "CP": 672}.items():
        region_mesh_path = ATLAS_DIR / "meshes" / f"{structure_id}.obj"
        if region_mesh_path.exists():
            region_meshes[region] = load_obj_mesh(
                str(region_mesh_path),
                atlas_mesh_signature(region_mesh_path),
            )

    selected_key = f"selected_ephys_session_{mouse}"
    if st.session_state.get(selected_key) not in selected_sessions:
        st.session_state[selected_key] = selected_sessions[-1]
    atlas_session = st.session_state[selected_key]
    atlas_probe_options = sorted(
        selected_inventory[selected_inventory["date"] == atlas_session]["probe"].unique().tolist()
    )
    atlas_probe_key = f"detail_probe_{atlas_session}"
    atlas_selected_probe = st.session_state.get(
        atlas_probe_key, atlas_probe_options[0] if atlas_probe_options else None
    )

    top_panel = st.container(border=True, key="atlas_session_panel")
    top_panel.markdown(
        '<div class="section-card-title">Probe selection</div>',
        unsafe_allow_html=True,
    )
    atlas_col, cards_col = top_panel.columns([1.62, 1.38], gap="large")
    with atlas_col:
        with st.container(border=False):
            atlas_figure = build_probe_figure(
                tracks,
                brain_mesh,
                region_meshes,
                selected_session=st.session_state[selected_key],
                selected_probe=atlas_selected_probe,
                annotation_path=str(annotation_path) if annotation_path.exists() else None,
                height=430,
            )
            st.plotly_chart(
                atlas_figure,
                width="stretch",
                key=f"probe_atlas_view_{mouse}_{atlas_session}_{atlas_selected_probe}",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                },
            )

    with cards_col:
        with st.container(border=False, key="session_cards"):
            card_columns = st.columns(2, gap="small")
            for card_index, session_date in enumerate(selected_sessions):
                card_color = session_color(session_date)
                is_selected = st.session_state[selected_key] == session_date
                button_key = f"session_option_{mouse}_{session_date}"
                card_background = "#FFF4F1" if is_selected else "#F3F5F7"
                card_shadow = (
                    "0 0 0 1px rgba(240,79,71,.14), "
                    "0 7px 16px rgba(224,76,59,.09)"
                    if is_selected
                    else "0 1px 3px rgba(34,50,72,0.04)"
                )
                st.markdown(
                    f"""
                    <style>
                    .st-key-{button_key} button {{
                        position: relative !important;
                        width: 100% !important;
                        min-height: 78px !important;
                        background: {card_background} !important;
                        border: 1.5px solid {"#F04F47" if is_selected else "#DDE3E8"} !important;
                        border-radius: 12px !important;
                        padding: 7px 25px 7px 10px !important;
                        margin: 0 0 8px 0 !important;
                        box-shadow: {card_shadow} !important;
                        color: {NAVY} !important;
                        justify-content: center !important;
                        text-align: center !important;
                        white-space: pre-line !important;
                        line-height: 1.24 !important;
                        font-size: .82rem !important;
                        font-weight: 450 !important;
                    }}
                    .st-key-{button_key} button::before {{
                        content: "";
                        position: absolute;
                        top: 0;
                        right: 0;
                        width: 22px;
                        height: 22px;
                        border-radius: 0 10px 0 8px;
                        background: {card_color};
                        box-shadow: none;
                    }}
                    .st-key-{button_key} button:hover {{
                        background: #EEF3F7 !important;
                        border-color: #AEBCC9 !important;
                        box-shadow: 0 6px 15px rgba(34,50,72,.09) !important;
                        transform: translateY(-1px);
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                session_tracks = [track for track in tracks if track["date"] == session_date]
                probes = sorted(track["probe"] for track in session_tracks)
                insertion = "Double insertion" if len(probes) > 1 else "Single insertion"
                session_good = sum(track["good_units"] for track in session_tracks)
                valid_bouts = behavior_counts.get(session_date)
                good_value = f"{session_good}" if session_good else "—"
                bouts_value = f"{int(valid_bouts)}" if pd.notna(valid_bouts) else "—"
                insertion_type = "Double insertion" if len(probes) > 1 else "Single insertion"
                label = (
                    f"{mouse} · {session_date}  \n"
                    f"{insertion_type}  \n"
                    f"{good_value} good units  {bouts_value} valid bouts"
                )
                with card_columns[card_index % 2]:
                    st.button(
                        label,
                        key=button_key,
                        type="secondary",
                        width="stretch",
                        on_click=set_ephys_session,
                        args=(selected_key, session_date),
                    )
        focus_session = st.session_state[selected_key]
        session_inventory = selected_inventory[selected_inventory["date"] == focus_session]
        probe_options = sorted(session_inventory["probe"].unique().tolist())
        shared_probe_key = f"detail_probe_{focus_session}"
        if st.session_state.get(shared_probe_key) not in probe_options:
            st.session_state[shared_probe_key] = probe_options[0]
        focus_probe = st.session_state[shared_probe_key]

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    focus_probe = focus_probe or probe_options[0]
    selected_row = session_inventory[session_inventory["probe"] == focus_probe].iloc[0]
    locations_path = Path(selected_row["path"])
    behavior_row = None
    behavior_database_index = None
    behavior_version = None
    selected_behavior_arrivals = behavior_on_site_arrivals.get(
        focus_session, np.array([])
    )
    if BEHAVIOR_DB.exists():
        behavior_signature = source_signature(BEHAVIOR_DB)
        behavior_database_index, behavior_version = ephys_behavior_selection(
            selected_row["session_dir"], focus_probe
        )
        if behavior_database_index is not None or behavior_version is not None:
            behavior_row = load_behavior_video_session(
                str(BEHAVIOR_DB),
                behavior_signature,
                mouse,
                focus_session,
                behavior_version,
                behavior_database_index,
            )
            if behavior_row is not None:
                correct_bouts = np.asarray(
                    behavior_row.get("Correct Bouts", []), dtype=bool
                )
                selected_behavior_arrivals = arrivals_from_traveling_epochs(
                    behavior_row.get("Traveling Epochs"), len(correct_bouts)
                )
                if not np.isfinite(selected_behavior_arrivals).any():
                    selected_behavior_arrivals = replay_numeric_array(
                        behavior_row.get("Onsite Acc Start Time")
                    )
                if len(correct_bouts) == len(selected_behavior_arrivals):
                    selected_behavior_arrivals = selected_behavior_arrivals[correct_bouts]
                selected_behavior_arrivals = selected_behavior_arrivals[
                    np.isfinite(selected_behavior_arrivals)
                ]
    analysis = load_probe_analysis(
        str(locations_path),
        probe_analysis_signature(
            locations_path,
            selected_row["session_dir"],
            focus_probe,
        ),
        selected_row["source"],
        selected_row["session_dir"],
        focus_probe,
        selected_behavior_arrivals,
    )
    units = analysis["units"]
    good_count = int((units["bombcell_label"] == "good").sum())
    good_percent = 100 * good_count / len(units) if len(units) else 0
    alignment = analysis["alignment"]

    with section_card("Replay"):
        if not BEHAVIOR_DB.exists():
            st.info("Behavioral database unavailable for this replay.")
        else:
            behavior_signature = source_signature(BEHAVIOR_DB)
            # New alignments identify the exact behavior row chosen from all
            # same-day candidates. Only legacy JSON files fall back to the
            # historical "highest Version" behavior.
            if behavior_row is None and behavior_database_index is None and behavior_version is None:
                video_index = load_behavior_video_index(str(BEHAVIOR_DB), behavior_signature)
                session_match = video_index[
                    (video_index["Mouse_ID"] == mouse) & (video_index["Date"] == focus_session)
                ]
                if not session_match.empty:
                    behavior_version = str(session_match.iloc[-1]["Version"])
                    behavior_row = load_behavior_video_session(
                        str(BEHAVIOR_DB), behavior_signature, mouse,
                        focus_session, behavior_version,
                    )
            if behavior_row is None:
                st.info("No behavioral recording matches this ephys session.")
            else:
                replay_payload = build_behavior_replay_payload(
                    behavior_row, mouse, focus_session
                ) if behavior_row is not None else None
                if replay_payload is None:
                    st.info("Position and speed signals are not available for this session.")
                else:
                    probe_activity_items = []
                    for probe_row in session_inventory.itertuples(index=False):
                        probe_name = str(probe_row.probe)
                        if probe_name == focus_probe:
                            probe_analysis = analysis
                        else:
                            probe_locations_path = Path(probe_row.path)
                            probe_analysis = load_probe_analysis(
                                str(probe_locations_path),
                                probe_analysis_signature(
                                    probe_locations_path,
                                    probe_row.session_dir,
                                    probe_name,
                                ),
                                probe_row.source,
                                probe_row.session_dir,
                                probe_name,
                                selected_behavior_arrivals,
                            )
                        probe_activity_items.append(
                            {
                                "probe": probe_name,
                                "regional_activity": probe_analysis.get("regional_activity"),
                                "atlas_regions": sorted({
                                    str(region)
                                    for region in probe_analysis["channels"]["brain_region"].dropna()
                                    if str(region).lower() != "unknown"
                                }),
                                "alignment": probe_analysis.get("alignment", {}),
                            }
                        )
                    replay_payload = attach_ephys_to_replay(
                        replay_payload, probe_activity_items,
                    )
                    structures_path = ATLAS_DIR / "structures.csv"
                    replay_regions = tuple(sorted({
                        region
                        for item in probe_activity_items
                        for region in (item.get("regional_activity") or {}).get("regions", {})
                    }))
                    if annotation_path.exists() and structures_path.exists():
                        session_track_coordinates = [
                            track["coordinates"] for track in tracks
                            if track["date"] == focus_session and len(track["coordinates"])
                        ]
                        default_ap_frame = (
                            float(np.nanmean(np.concatenate(session_track_coordinates)[:, 0])) / 25
                            if session_track_coordinates else 264.0
                        )
                        replay_payload["atlasSlices"] = build_replay_atlas_slices(
                            str(annotation_path), source_signature(annotation_path),
                            str(structures_path), source_signature(structures_path),
                            replay_regions, default_ap_frame,
                        )
                    components.html(
                        build_behavior_replay_html(replay_payload),
                        height=880,
                        scrolling=False,
                    )

    with section_card("Anatomy"):
        render_probe_selector("Anatomy", probe_options, shared_probe_key)
        probe_profile = build_probe_anatomy_figure(analysis["channels"], units)
        if probe_profile is None:
            st.info("Channel anatomy is not available for this probe.")
        else:
            st.markdown('<div class="subplot-title anatomy-subtitle">Probe anatomy</div>', unsafe_allow_html=True)
            st.plotly_chart(probe_profile, width="stretch", config={"displayModeBar": False})

    with section_card("Activity"):
        render_probe_selector("Activity", probe_options, shared_probe_key)
        with st.container(border=False, key="activity_viewer"):
            if analysis["activity_heatmap"] is None:
                st.info("Good-unit spike clusters are not available for this probe.")
            else:
                activity_left, activity_right = st.columns([1.22, 1], gap="large")
                with activity_left:
                    st.markdown('<div class="subplot-title">Good-unit activity</div>', unsafe_allow_html=True)
                    st.plotly_chart(
                        build_activity_heatmap(
                            analysis["activity_heatmap"],
                            behavior_bout_starts.get(focus_session, np.array([])),
                            height=430,
                        ),
                        width="stretch",
                        key=f"activity_chart_{mouse}_{focus_session}_{focus_probe}",
                        config={
                            "displayModeBar": False,
                            "displaylogo": False,
                            "scrollZoom": False,
                        },
                    )
                with activity_right:
                    if analysis["peri_bout"] is None:
                        st.info("Arrival-aligned activity is unavailable for this probe.")
                    else:
                        st.markdown(
                            '<div class="subplot-title">Arrival on site</div>',
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            build_peri_bout_figure(analysis["peri_bout"], height=430),
                            width="stretch",
                            key=f"peri_bout_chart_{mouse}_{focus_session}_{focus_probe}",
                            config={"displayModeBar": False, "displaylogo": False, "scrollZoom": False},
                        )

    with section_card("Quality control"):
        render_probe_selector("Quality control", probe_options, shared_probe_key)
        qc_left, qc_right = st.columns(2, gap="large")
        with qc_left:
            st.markdown('<div class="subplot-title">Bombcell classification</div>', unsafe_allow_html=True)
            st.plotly_chart(build_qc_figure(units), width="stretch", config={"displayModeBar": False})
        with qc_right:
            st.markdown('<div class="subplot-title">QC criterion selectivity</div>', unsafe_allow_html=True)
            qc_selectivity = build_qc_selectivity_figure(units)
            if qc_selectivity is None:
                st.info("QC metrics are unavailable for this probe.")
            else:
                st.plotly_chart(
                    qc_selectivity, width="stretch", config={"displayModeBar": False}
                )

        st.markdown('<div class="subplot-title">Unit explorer</div>', unsafe_allow_html=True)
        display_columns = [
            column
            for column in [
                "unit_id",
                "bombcell_label",
                "brain_region",
                "depth_um",
                "num_spikes",
                "firing_rate",
                "snr",
                "presence_ratio",
                "rp_contamination",
                "amplitude_median",
                "drift_ptp",
            ]
            if column in units.columns
        ]
        dataframe_card(units[display_columns])

    if False:  # Behavioral replay is rendered before probe-dependent sections.
        if not BEHAVIOR_DB.exists():
            st.info("Behavioral database unavailable for this replay.")
        else:
            behavior_signature = source_signature(BEHAVIOR_DB)
            video_index = load_behavior_video_index(str(BEHAVIOR_DB), behavior_signature)
            session_match = video_index[
                (video_index["Mouse_ID"] == mouse) & (video_index["Date"] == focus_session)
            ]
            if session_match.empty:
                st.info("No behavioral recording matches this ephys session.")
            else:
                version = str(session_match.iloc[-1]["Version"])
                behavior_row = load_behavior_video_session(
                    str(BEHAVIOR_DB), behavior_signature, mouse, focus_session, version
                )
                replay_payload = build_behavior_replay_payload(
                    behavior_row, mouse, focus_session
                ) if behavior_row is not None else None
                if replay_payload is None:
                    st.info("Position and speed signals are not available for this session.")
                else:
                    probe_activity_items = []
                    for probe_row in session_inventory.itertuples(index=False):
                        probe_name = str(probe_row.probe)
                        if probe_name == focus_probe:
                            probe_analysis = analysis
                        else:
                            probe_locations_path = Path(probe_row.path)
                            probe_analysis = load_probe_analysis(
                                str(probe_locations_path),
                                probe_analysis_signature(
                                    probe_locations_path,
                                    probe_row.session_dir,
                                    probe_name,
                                ),
                                probe_row.source,
                                probe_row.session_dir,
                                probe_name,
                                behavior_on_site_arrivals.get(focus_session, np.array([])),
                            )
                        probe_activity_items.append(
                            {
                                "probe": probe_name,
                                "regional_activity": probe_analysis.get("regional_activity"),
                                "alignment": probe_analysis.get("alignment", {}),
                            }
                        )
                    replay_payload = attach_ephys_to_replay(
                        replay_payload, probe_activity_items,
                    )
                    components.html(
                        build_behavior_replay_html(replay_payload),
                        height=880 + 165 * max(1, len(probe_activity_items)),
                        scrolling=False,
                    )
                    st.caption(
                        "Measured position and speed · good-unit firing rates transformed with "
                        "t_behavior = a × t_ephys + b · site deactivation at the triggering lick."
                    )


inject_css()

try:
    if st.session_state.get(NAVIGATION_STATE_KEY) not in VIEW_OPTIONS:
        legacy_view = st.session_state.get("global_view")
        st.session_state[NAVIGATION_STATE_KEY] = (
            legacy_view if legacy_view in VIEW_OPTIONS
            else DEFAULT_VIEW if DEFAULT_VIEW in VIEW_OPTIONS
            else VIEW_OPTIONS[0]
        )
    prepared_inventory = None
    prepared_openfield_inventory = None
    default_navigation_mouse = None
    selected_view = st.session_state[NAVIGATION_STATE_KEY]
    navigation_shell = st.container(border=False, key="navigation_fixed_shell")
    with navigation_shell:
        navigation_bar_column, mouse_column = st.columns(
            [6, 1], gap="medium", vertical_alignment="center",
        )
        with navigation_bar_column:
            with st.container(border=True, key="app_navigation"):
                navigation_columns = st.columns(6, gap=None, vertical_alignment="center")
                for navigation_column, view in zip(navigation_columns, VIEW_OPTIONS):
                    with navigation_column:
                        st.button(
                            VIEW_LABELS[view], key=f"global_nav_{view.lower()}",
                            type="primary" if selected_view == view else "secondary",
                            width="stretch", on_click=set_global_view, args=(view,),
                        )

    if selected_view == "Electrophysiology":
        prepared_inventory = discover_ibl_channel_locations(EPHYS_ROOT)
        mouse_options = (
            sorted(
                mouse for mouse in prepared_inventory["mouse"].unique().tolist()
                if is_visible_mouse(mouse)
            )
            if not prepared_inventory.empty else []
        )
    elif selected_view == "Openfield":
        prepared_openfield_inventory = discover_openfield_sessions(OPENFIELD_ROOT)
        mouse_options = (
            sorted(prepared_openfield_inventory["mouse"].unique().tolist())
            if not prepared_openfield_inventory.empty else []
        )
    elif BEHAVIOR_DB.exists():
        navigation_dataframe = load_behavior_database(
            str(BEHAVIOR_DB), source_signature(BEHAVIOR_DB)
        )
        mouse_options = sorted(
            mouse for mouse in navigation_dataframe["Mouse_ID"].dropna().unique().tolist()
            if is_visible_mouse(mouse)
        )
        latest_per_mouse = navigation_dataframe.groupby(
            "Mouse_ID", dropna=False
        )["Date"].max().dropna()
        default_navigation_mouse = (
            latest_per_mouse.idxmax() if not latest_per_mouse.empty
            else mouse_options[0] if mouse_options else None
        )
    else:
        mouse_options = []

    if mouse_options and default_navigation_mouse not in mouse_options:
        default_navigation_mouse = mouse_options[0]

    selected_mouse = None
    with mouse_column:
        if selected_view in {"Home", "Chatbot"}:
            pass
        elif mouse_options:
            global_mouse = st.session_state.get("global_mouse")
            if global_mouse not in mouse_options:
                global_mouse = default_navigation_mouse
                st.session_state["global_mouse"] = global_mouse
            navigation_mouse_key = f"navigation_mouse_{selected_view.lower()}"
            if st.session_state.get(navigation_mouse_key) != global_mouse:
                st.session_state[navigation_mouse_key] = global_mouse
            selected_mouse = st.selectbox(
                "Mouse", mouse_options, key=navigation_mouse_key,
                label_visibility="collapsed",
                on_change=sync_navigation_mouse, args=(navigation_mouse_key,),
            )
            st.session_state["global_mouse"] = selected_mouse
        else:
            st.markdown(
                '<div style="height:58px;display:flex;align-items:center;'
                'justify-content:center;color:#748091;font-size:.80rem;">No mouse</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="navigation-flow-spacer"></div>', unsafe_allow_html=True)

    if st.session_state.pop("_scroll_to_top_requested", False):
        components.html(
            """
            <script>
            (() => {
              const doc = window.parent.document;
              const app = doc.querySelector('[data-testid="stAppViewContainer"]');
              if (app) app.scrollTop = 0;
              doc.documentElement.scrollTop = 0;
              doc.body.scrollTop = 0;
              window.parent.scrollTo(0, 0);
            })();
            </script>
            """,
            height=0,
            scrolling=False,
        )

    if selected_view == "Home":
        render_home_view()
    elif selected_view == "Info":
        pass
    elif selected_view == "Chatbot":
        render_chatbot_view(selected_mouse)
    elif selected_view == "Openfield":
        render_openfield_view(selected_mouse, prepared_openfield_inventory)
    elif selected_view == "Electrophysiology":
        render_ephys_view(selected_mouse, prepared_inventory)
    else:
        render_behavior_view(selected_view, selected_mouse)

except Exception as error:
    st.error("Application failed")
    st.exception(error)
