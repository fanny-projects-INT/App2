"""Shared empty and loading states for the dashboard."""

import streamlit as st


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


def loading_mouse_markup():
    return """
    <style>
    [data-testid="stAppViewContainer"] { visibility:hidden; }
    .page-loading-mask {
        visibility:visible; position:fixed; inset:0; z-index:999999;
        display:flex; align-items:center; justify-content:center; background:#F7F9FC;
    }
    .running-mouse { position:relative; width:156px; height:94px; color:#F05A3C; }
    .running-mouse svg {
        position:absolute; z-index:2; width:94px; height:61px; left:31px; top:3px;
        animation:mouse-bob .22s ease-in-out infinite alternate;
    }
    .running-mouse .body { fill:currentColor; stroke:currentColor; stroke-width:2; }
    .running-mouse .detail { fill:none; stroke:#F7F9FC; stroke-width:1.8; stroke-linecap:round; }
    .running-mouse .line { fill:none; stroke:currentColor; stroke-width:2.5; stroke-linecap:round; }
    .running-mouse .eye { fill:#F7F9FC; }
    .running-mouse .paw { fill:currentColor; animation:paw-step .28s ease-in-out infinite alternate; }
    .running-mouse .paw-2,.running-mouse .paw-4 { animation-direction:alternate-reverse; }
    @keyframes paw-step {
        from { transform:translate(-3px,-1px); }
        to { transform:translate(4px,2px); }
    }
    @keyframes mouse-bob { to { transform:translateY(-1.5px) rotate(-1deg); } }
    @media (prefers-reduced-motion:reduce) {
        .running-mouse svg,.running-mouse .paw { animation:none; }
    }
    </style>
    <div class="page-loading-mask" role="status" aria-label="Chargement de la page">
      <div class="running-mouse" aria-hidden="true">
        <svg viewBox="0 0 100 64">
          <path d="M19 37C3 37 3 21 15 17" class="line"/>
          <ellipse cx="48" cy="32" rx="29" ry="19" class="body"/>
          <path d="M63 22c5-12 21-11 26 1 4 10-3 20-14 19-10 0-16-10-12-20Z" class="body"/>
          <circle cx="70" cy="10" r="8" class="body"/>
          <path d="M68 8c3-2 6-1 7 2" class="detail"/>
          <circle cx="82" cy="23" r="2.1" class="eye"/>
          <path d="M88 31l7 1M87 34l7 3" class="line"/>
          <circle cx="34" cy="50" r="5" class="paw paw-1"/>
          <circle cx="44" cy="51" r="5" class="paw paw-2"/>
          <circle cx="56" cy="51" r="5" class="paw paw-3"/>
          <circle cx="66" cy="49" r="5" class="paw paw-4"/>
        </svg>
      </div>
    </div>
    """
