# INVIBE behavior & electrophysiology dashboard

The Streamlit app reads the behavioral Feather database directly. Its
electrophysiology view discovers final IBL GUI channel locations for each mouse
and displays all selected sessions in an interactive 3D atlas-space view.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the repository layout.

Default sources:

- behavior: `F:\full_db_all_rigs.feather`
- ephys/Brainreg root: `F:\coregistration_db`
- open field root: `F:\Behavior`

They can be overridden without changing the code:

```powershell
$env:APP_BEHAVIOR_DB = "D:\data\full_db_all_rigs.feather"
$env:APP_EPHYS_ROOT = "D:\ephys"
$env:APP_OPENFIELD_ROOT = "D:\behavior"
$env:APP_ATLAS_DIR = "C:\Users\me\.brainglobe\allen_mouse_25um_v1.2"
$env:OPENAI_API_KEY = "your-project-api-key"
# Optional server-wide override; otherwise the UI selects Sol or Terra
$env:OPENAI_CHAT_MODEL = "gpt-5.6-sol"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The Cazette Bot uses the OpenAI Responses API with validated read-only local
tools. Its general query engine can inspect the live schema, filter and
aggregate Behavior or Ephys data, compute descriptive statistics and
correlations, and render line, bar, scatter, histogram, density, box or violin
plots. `Advanced` uses GPT-5.6 Sol with high reasoning effort; `Fast` uses
GPT-5.6 Terra. It has no shell, filesystem-write, database-write, or source-code
editing tool. API responses are requested with `store=False`. If
`OPENAI_API_KEY` is not configured on the server, the key can be entered for
the current Streamlit session from the Cazette Bot tab.

For each mouse, fitted channel locations are discovered below:

`YYYY_MM_DD\alf\probeNN\channel_locations.json`

The IBL coordinates are converted from ML/AP/DV relative to Bregma into Allen
CCF space. Tracks from the same session use the same color, while the line style
distinguishes probes. Each track is rendered as a straight total-least-squares
3D fit inside the Allen mouse brain root mesh (`meshes\997.obj`).

The electrophysiology overview places a compact 3D atlas next to clickable
session cards. Each card summarizes insertion type, expected probe target,
good-unit count, and valid behavioral bouts. Clicking a card controls the
details shown below: `Overview`, `Activity`, or `Units & QC`.

## Responsiveness

The Info tab shows Alyx information for the mouse selected in the navigation.
Sign in there with your Alyx account to see its general profile and visual
session cards. Refresh reloads the selected mouse; Log out clears the connection
and fetched records from the browser session. Credentials
can be entered interactively and the authentication token is not cached on disk.
For automatic login, fill in the local `.streamlit/secrets.toml` file (ignored by Git):

```toml
[alyx]
username = "your-alyx-username"
password = "your-alyx-password"
```

Restart Streamlit after editing this file. The configured Alyx account is used
for all visitors to the app; Log out disconnects only the current browser session.
The connection uses `ALYX_URL` and `ALYX_LAB` from the environment or the supplied
handoff defaults. Install `requirements.txt` to include ONE-api and python-dotenv.

Ephys keeps Replay, Anatomy, Activity and Quality control on the same page.
Atlas and analysis figures are cached with their inputs. Spike counting uses
bounded chunks and vectorized bin accumulation to reduce memory and CPU cost.
Behavioral replay caches selected sessions and reads signal columns separately,
avoiding a second complete decoded database in memory.
Ephys and open-field inventories are cached for 60 seconds; new sessions become
visible on the next interaction after that interval.

## Loading and cache

Navigation, the mouse list and the Behavior overview read only four metadata
columns, without loading the full signal database. Detailed Behavior views pin the eight mice with the newest Behavior sessions
in a shared memory cache, preloaded in the background from Home. Only those
eight mice retain prepared data; older mice are loaded on demand. This is an
entry limit, not a fixed RAM budget. Renderers
copy the frame before adding display columns. Signals are decoded one column
at a time, keeping only the requested rows instead of caching the complete
decoded database. Replay likewise retains only selected sessions. Changing the
source file size or modification time invalidates the caches automatically.
The first detailed visit to an uncached mouse still reads the source file;
subsequent visits reuse the prepared data.

Ephys loads detailed probe analysis only when a detail tab is opened, and renders
the 3D selection plot only in Probe selection. Behavioral event counting uses
grouped sample bounds and binary searches while preserving the original results.

Run the regression checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Mouse selectors list the newest session first for the current view (Behavior,
Ephys or Arena tests), with alphabetical ties and undated mice last. A selected
mouse without data in the current view stays at the bottom of the list.

Measured on 2026-10-08: the eight latest visible Behavior mice contain about
448 MiB of raw Arrow data. Python objects, figures, replay, Ephys caches and
temporary decoding buffers consume additional memory. Both mouse caches are
shared across visitors and reset when the server restarts.

Info and Behavior cache the same eight visible mice with the newest Behavior
sessions, independently of consultation history. Info preloads the configured
server Alyx account from Home and shares its connection/data between visitors,
as authorized by the app owner. Info entries have no time-based expiration. Opening Home preloads the eight
recent mice. Info fetches the selected mouse first, then warms the group in the
background, without a Refresh button. Cached reads do not wait for background
network requests; uncached selected mice take priority in the preload queue. Older mice load on demand without
entering the reserved mouse cache. Manual Alyx logins remain session-isolated.
The group follows changes to the Behavior catalog. Server caches reset on
restart; credential changes select a different Info account cache.

Navigation uses native Streamlit tabs and fragments in Behavior, Info, Ephys and
Arena. Only the active panel is computed; tab selections survive temporarily
hidden nested groups. Streamlit owns page replacement and rendering: there are
no retained page copies, DOM observers, loading masks or artificial reveal
markers. New content uses Streamlit's normal rendering flow. The eight-mouse
caches and data-only background preloading remain unchanged.

Behavior Overview uses compact text cards, including session counts by protocol
across the mouse’s complete history, and a single timeline line with
colored dots for local Behavior sessions over the last 30 calendar days,
including today. Dates run from left to right, with gaps for inactive days. Day numbers appear below the line, with English weekday initials above it
and alternating week backgrounds separated by a vertical line on Mondays. Summary cards fit their contents and wrap on
narrow screens.
Hover over a dot for its date, protocol and version. There are no month controls.
The timeline itself does not connect to Alyx or load signal arrays.

Overview shows two larger Progression panels on the left (valid bouts and
rewarded licks/min), and four smaller panels on the right in one 500-pixel
figure. Global and per-session failure curves share a panel, with the existing
global smoothed curve drawn on top in black dashes; the failures axis is limited
to 0–30. The remaining panels show
rewards vs failures, linear slope and mean failures. It reuses cached data and
fits; hover details remain available. Behavior navigation now contains only Overview and Sessions.
Overview loads the cached prepared mouse data needed for these plots, but does
not request replay/video signals.

Empty Overview plot panels hide their titles and axes. Valid-bout and rewarded-lick
markers are larger for readability.

Sessions uses the same content-sized information cards as Overview.
The synchronized replay shows the mouse on the left and stacks behavioral
events and speed on the right. Below the replay, the behavioral timeline keeps its three stacked traces
on the left, with rewards vs failures and failure distribution beside it on
the right. All three plots share one 280-pixel row. Plot fonts are compact with automatic margins. Session
selection remains preserved when switching tabs; old Progression selections
open Overview.

Sessions aligns both rows into equal left/right halves: reconstruction and
behavioral timeline on the left, event/speed traces and the two analysis plots
on the right. Playback controls stay directly under the mouse reconstruction.


### Raw electrophysiology QC

In **Electrophysiology > Quality control**, the full-width RMS plot loads saved
`alf/<probe>/_iblqc_ephysTimeRmsAP.rms.npy` and its timestamps directly.
All saved time windows are displayed. There is no raw recording read or spectrum
calculation. Missing saved RMS produces an information message.
Code is in `dashboard/raw_qc.py`.


### Arena session metrics

Opening Arena tests checks the selected mouse's analyzed OFT/LDT sessions.
The app calculates metrics directly from `remapped_cropped_*_clean.csv`,
OFT `*_behavior.csv`, and the clean video's exact frame rate. Results are
saved atomically as `arena_metrics.json` in each session directory. Later visits
reuse this file; changes to source file paths, sizes, modification times, or the
algorithm version invalidate it. A missing clean video uses the reference 50 Hz
fallback, recorded as `fps_source` in the result.

`dashboard/arena_metrics.py` reuses the formulas from
`lp/analyse/visualisation_metriques.ipynb`: OFT calibration 16 pixels/cm,
center 300?700 px, likelihood threshold 0.5 and the same interpolation/smoothing;
LDT uses visibility of neck, torso and tail_base to identify the dark box.
OFT/LDT calculations were checked against the existing VF087 SQLite export.
Local results take precedence over legacy SQLite results for the same mouse,
assay and date, both in Overview and Sessions. SQLite remains a fallback for
sessions whose analyzed inputs are absent and is not modified. Failed updates
are reported in the app. SIT calculation is not defined in the reference pipeline;
existing SIT metrics can be shown, but raw videos alone do not generate metrics.
