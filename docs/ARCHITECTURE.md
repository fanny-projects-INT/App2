# Repository structure

The dashboard keeps one Python entry point at the repository root and groups
all active feature modules in a single, shallow `dashboard` package.

```text
app.py                         Streamlit entry point and page orchestration
dashboard/                     All active Python feature modules

assets/components/             Custom Streamlit web components
scripts/windows/               Windows launch and installation helpers
tests/                         Main regression suite
exports/                       Generated analysis exports
alyx_dashboard_readonly/       Standalone read-only Alyx utility
```

The active feature modules are intentionally kept in one flat package:

| Module | Responsibility |
| --- | --- |
| `alyx_info.py` | Alyx-backed mouse and session information |
| `app_functions.py` | Shared behavior calculations and plots |
| `arena_metrics.py` | OFT/LDT metrics and their local cache file |
| `behavior_overview.py` | Behavior overview cards and figures |
| `histology_resampling.py` | Histology transforms and image sampling |
| `probe_profile.py` | Electrophysiology probe profile figures |
| `raw_qc.py` | Loading and rendering saved raw-signal QC |
| `ui_states.py` | Shared Streamlit tabs, empty states and prefetching |

## Dependency direction

`app.py` is the composition layer. Feature modules may use
`dashboard/app_functions.py`
and third-party libraries, but should not import `app.py`. This keeps feature
logic importable without starting Streamlit.

Custom component files are resolved relative to `app.py`; they do not depend on
the process working directory. Windows scripts similarly resolve the repository
root from their own location.

## Commands

Run the dashboard from the repository root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run the regression suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

`START_PUBLIC_APP.cmd` remains at the root as the stable double-click launcher.
The PowerShell implementation and setup helpers live in `scripts/windows/`.
