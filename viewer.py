"""Dash viewer: MTG patch + EarthCARE curtain side by side."""

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import re
import io
import json
import base64
import warnings
warnings.filterwarnings("ignore", message="no explicit representation of timezones")
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
from functools import lru_cache
warnings.filterwarnings("ignore", module="earthcarekit")
import earthcarekit as eck
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, no_update

from matplotlib.colors import BoundaryNorm
from earthcarekit.plot.color.colormap.cmap import Cmap

from ec_track import load_ec_track as _load_ec_track_raw

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATCH_DIR = os.path.join(BASE_DIR, "DATA", "patches")
EC_DIR = os.path.join(BASE_DIR, "DATA", "EC")

# ── Pre-computed band percentiles (from compute_band_stats.py) ───────────────
_BAND_STATS_FILE = os.path.join(BASE_DIR, "band_stats.json")
_BAND_STATS = {}
if os.path.isfile(_BAND_STATS_FILE):
    with open(_BAND_STATS_FILE) as _f:
        _BAND_STATS = json.load(_f)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_patch(path):
    return xr.open_dataset(path, engine="h5netcdf")


def _fig_to_b64(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


# ── Scan patches ─────────────────────────────────────────────────────────────
_PATCH_RE = re.compile(r"EC_MTG_patch_(\w+)_(\d+)_(\d{8}T\d{6})\.nc")


def _scan_patches():
    patches = {}
    for f in sorted(os.listdir(PATCH_DIR)):
        m = _PATCH_RE.match(f)
        if not m:
            continue
        orbit_id, idx, time_str = m.groups()
        fpath = os.path.join(PATCH_DIR, f)
        ds = _read_patch(fpath)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        patches.setdefault(orbit_id, []).append({
            "idx": int(idx),
            "time_str": time_str,
            "filename": f,
            "path": fpath,
            "ec_time": str(ds.attrs["ec_time"]),
            "mtg_time": str(ds.attrs.get("mtg_time", "unknown")),
            "center_lat": float(ds.attrs["center_lat"]),
            "center_lon": float(ds.attrs["center_lon"]),
            # Footprint corners for the map
            "lat_min": float(np.nanmin(lat)),
            "lat_max": float(np.nanmax(lat)),
            "lon_min": float(np.nanmin(lon)),
            "lon_max": float(np.nanmax(lon)),
        })
        ds.close()
    for v in patches.values():
        v.sort(key=lambda x: x["idx"])
    return patches


PATCHES = _scan_patches()

ORBIT_EC_FILE = {}
for _oid, _plist in PATCHES.items():
    _ds = _read_patch(_plist[0]["path"])
    _ec_fname = _ds.attrs["ec_source_file"]
    _ds.close()
    _ec_path = os.path.join(EC_DIR, _ec_fname)
    if os.path.isfile(_ec_path):
        ORBIT_EC_FILE[_oid] = _ec_path

# ── RGB composite recipes (EUMETSAT standard) ────────────────────────────────
# Each band spec is either a channel name or a (ch1, ch2) tuple for ch1 − ch2.
_RGB_RECIPES = {
    "rgb_true": {
        "label": "True Colour",
        "bands": ["vis_06", "vis_05", "vis_04"],
        "gamma": [1.0, 1.0, 1.0],
    },
    "rgb_natural": {
        "label": "Natural Colour",
        "bands": ["nir_16", "vis_08", "vis_06"],
        "gamma": [1.0, 1.0, 1.0],
    },
    "rgb_natural_enh": {
        "label": "Natural Colour Enh.",
        "bands": ["nir_16", "vis_08", "vis_06"],
        "gamma": [1.3, 1.3, 1.3],
    },
    "rgb_cloud_phase": {
        "label": "Cloud Phase",
        "bands": ["nir_16", "nir_22", "vis_06"],
        "gamma": [1.0, 1.0, 1.0],
    },
    "rgb_cloud_type": {
        "label": "Cloud Type",
        "bands": ["vis_06", "nir_13", "nir_16"],
        "gamma": [1.0, 1.0, 1.0],
    },
    "rgb_day_micro": {
        "label": "Day Microphysics",
        "bands": ["vis_08", "ir_38", "ir_105"],
        "gamma": [1.0, 2.5, 1.0],
    },
    "rgb_fog": {
        "label": "Fog / Low Clouds",
        "bands": [("ir_123", "ir_105"), ("ir_105", "ir_38"), "ir_105"],
        "gamma": [1.0, 1.0, 1.0],
    },
}


def _recipe_channels(recipe):
    """Return set of raw channel names required by an RGB recipe."""
    channels = set()
    for b in recipe["bands"]:
        if isinstance(b, tuple):
            channels.update(b)
        else:
            channels.add(b)
    return channels


def _load_band(ds, band_spec):
    """Load a single band from ds.  band_spec: str or (ch1, ch2) for difference."""
    if isinstance(band_spec, tuple):
        ch1, ch2 = band_spec
        v1 = ds[f"effective_radiance_{ch1}"].values.astype(np.float32)
        v2 = ds[f"effective_radiance_{ch2}"].values.astype(np.float32)
        if v1.shape != v2.shape:
            v2 = np.repeat(np.repeat(v2,
                                      v1.shape[0] // v2.shape[0], axis=0),
                           v1.shape[1] // v2.shape[1], axis=1)
        return v1 - v2
    return ds[f"effective_radiance_{band_spec}"].values.astype(np.float32)


def _match_resolution(bands):
    """Resample bands to the largest (highest-res) shape via nearest-neighbor."""
    max_shape = max(b.shape for b in bands)
    out = []
    for b in bands:
        if b.shape == max_shape:
            out.append(b)
        else:
            zy = max_shape[0] // b.shape[0]
            zx = max_shape[1] // b.shape[1]
            out.append(np.repeat(np.repeat(b, zy, axis=0), zx, axis=1))
    return out


# ── Detect available channels from first patch ───────────────────────────────
_AVAILABLE_CHANNELS = set()
if PATCHES:
    _first_orbit = next(iter(PATCHES.values()))
    if _first_orbit:
        _ds = _read_patch(_first_orbit[0]["path"])
        for _vname in _ds.data_vars:
            if _vname.startswith("effective_radiance_"):
                _AVAILABLE_CHANNELS.add(_vname.replace("effective_radiance_", ""))
        _ds.close()

# Build channel dropdown options: available RGB recipes first, then individual channels
_CHANNEL_OPTIONS = []
for _key, _recipe in _RGB_RECIPES.items():
    if _recipe_channels(_recipe) <= _AVAILABLE_CHANNELS:
        _CHANNEL_OPTIONS.append({"label": _recipe["label"], "value": _key})
for _ch in sorted(_AVAILABLE_CHANNELS):
    _CHANNEL_OPTIONS.append({"label": _ch, "value": _ch})
_DEFAULT_CHANNEL = _CHANNEL_OPTIONS[0]["value"] if _CHANNEL_OPTIONS else None

EC_VARS = [
    "synergetic_target_classification",
    "ATLID_target_classification",
    "CPR_target_classification",
]

# ── Cloud & precipitation class definitions per variable ─────────────────────
# Each entry maps variable name -> list of (class_value, color, label) to keep.
# Sourced from earthcarekit colormap definitions.

_CLOUD_PRECIP_CLASSES = {
    "synergetic_target_classification": [
        [2, "#ff474c", "Rain in clutter"],
        [3, "#0504aa", "Snow in clutter"],
        [4, "#009337", "Cloud in clutter"],
        [5, "#840000", "Heavy rain"],
        [6, "#042e60", "Heavy mixed-phase precip."],
        [8, "#ffff84", "Liquid cloud"],
        [9, "#f5bf03", "Drizzling liquid cloud"],
        [10, "#f97306", "Warm rain"],
        [11, "#ff000d", "Cold rain"],
        [12, "#5539cc", "Melting snow"],
        [13, "#2976bb", "Snow (possible liquid)"],
        [14, "#0d75f8", "Snow"],
        [15, "#014182", "Rimed snow (possible liquid)"],
        [16, "#017b92", "Rimed snow + s'cooled liquid"],
        [17, "#06b48b", "Snow + liquid"],
        [18, "#aaff32", "S'cooled liquid cloud"],
        [19, "#6dedfd", "Ice cloud (possible liquid)"],
        [20, "#01f9c6", "Ice + liquid cloud"],
        [21, "#7bc8f6", "Ice cloud"],
    ],
    "ATLID_target_classification": [
        [1, "#1192E8", "(Warm) liquid cloud"],
        [2, "#004489", "S'cooled cloud"],
        [3, "#93FBFF", "Ice cloud"],
    ],
    "CPR_target_classification": [
        [2, "#ffff84", "Liquid cloud"],
        [3, "#f5bf03", "Drizzling liquid clouds"],
        [4, "#f97306", "Warm rain"],
        [5, "#ff000d", "Cold rain"],
        [6, "#c071fe", "Melting snow"],
        [7, "#004577", "Rimed snow"],
        [8, "#0165fc", "Snow"],
        [9, "#95d0fc", "Ice"],
        [12, "#840000", "Heavy rain likely"],
        [13, "#0504aa", "Mixed-phase precip. likely"],
        [14, "#840000", "Heavy rain"],
        [15, "#001146", "Heavy mixed-phase precip."],
        [16, "#bb3f3f", "Rain in clutter"],
        [17, "#5684ae", "Snow in clutter"],
        [18, "#eedc5b", "Cloud in clutter"],
    ],
}


def _make_filtered_cmap(ec_var):
    """Build a Cmap containing only cloud/precipitation classes for ec_var."""
    entries = _CLOUD_PRECIP_CLASSES.get(ec_var)
    if entries is None:
        return None, set()
    colors = [c for _, c, _ in entries]
    definitions = {k: l for k, _, l in entries}
    keep_values = set(definitions.keys())
    cmap = Cmap(colors=colors, name=f"{ec_var}_cloud_precip").to_categorical(definitions)
    return cmap, keep_values


# ── EC track cache ───────────────────────────────────────────────────────────
_ec_track_cache = {}


def _load_ec_track(ec_path):
    if ec_path in _ec_track_cache:
        return _ec_track_cache[ec_path]
    lats, lons, times = _load_ec_track_raw(ec_path)
    _ec_track_cache[ec_path] = (lats, lons, times)
    return lats, lons, times


# ── App ──────────────────────────────────────────────────────────────────────
app = Dash(__name__)

orbit_options = [{"label": oid, "value": oid} for oid in sorted(PATCHES.keys())]

app.layout = html.Div(
    style={"fontFamily": "sans-serif", "padding": "10px"},
    children=[
        html.H2("EarthCARE / MTG Collocation Viewer"),

        # Controls
        html.Div(
            style={"display": "flex", "gap": "20px", "alignItems": "flex-end",
                   "flexWrap": "wrap", "marginBottom": "10px"},
            children=[
                html.Div([
                    html.Label("Orbit"),
                    dcc.Dropdown(id="orbit-dd", options=orbit_options,
                                 value=orbit_options[0]["value"] if orbit_options else None,
                                 style={"width": "160px"}),
                ]),
                html.Div([
                    html.Label("Patch"),
                    dcc.Dropdown(id="patch-dd", style={"width": "200px"}),
                ]),
                html.Div([
                    html.Label("MTG channel"),
                    dcc.Dropdown(id="channel-dd",
                                 options=_CHANNEL_OPTIONS,
                                 value=_DEFAULT_CHANNEL,
                                 style={"width": "200px"}),
                ]),
                html.Div([
                    html.Label("EC variable"),
                    dcc.Dropdown(id="ecvar-dd",
                                 options=[{"label": v, "value": v} for v in EC_VARS],
                                 value=EC_VARS[0],
                                 style={"width": "320px"}),
                ]),
                html.Div([
                    html.Label("\u00a0"),
                    html.Button("Clear cache", id="clear-cache-btn",
                                style={"cursor": "pointer"}),
                ]),
                html.Div(id="cache-status",
                         style={"fontSize": "12px", "color": "#888",
                                "alignSelf": "center", "paddingTop": "18px"}),
            ],
        ),

        html.Div(id="patch-info",
                 style={"fontSize": "13px", "color": "#555", "marginBottom": "8px"}),

        # Row 1: World map + Context map + MTG patch
        html.Div(
            style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                   "marginBottom": "10px", "alignItems": "flex-start"},
            children=[
                html.Div([
                    html.H4("World Map", style={"marginBottom": "4px"}),
                    dcc.Loading(type="circle", children=[
                        html.Img(id="world-map",
                                 style={"height": "480px", "border": "1px solid #ccc"}),
                    ]),
                ]),
                html.Div([
                    html.H4("Context Map (click a patch)", style={"marginBottom": "4px"}),
                    dcc.Loading(type="circle", children=[
                        dcc.Graph(id="overview-map",
                                  style={"height": "520px", "width": "520px"},
                                  config={"scrollZoom": True}),
                    ]),
                ]),
                html.Div([
                    html.H4("MTG Patch", style={"marginBottom": "4px"}),
                    dcc.Loading(type="circle", children=[
                        html.Img(id="mtg-img",
                                 style={"height": "480px", "border": "1px solid #ccc"}),
                    ]),
                ]),
            ],
        ),

        # Row 2: Curtain
        html.Div([
            html.H4("EarthCARE Curtain", style={"marginBottom": "4px"}),
            dcc.Loading(type="circle", children=[
                html.Img(id="curtain-img",
                         style={"maxWidth": "100%", "border": "1px solid #ccc"}),
            ]),
        ]),
    ],
)


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("patch-dd", "options"),
    Output("patch-dd", "value"),
    Input("orbit-dd", "value"),
)
def update_patch_options(orbit_id):
    if not orbit_id or orbit_id not in PATCHES:
        return [], None
    opts = [{"label": f"{p['idx']:03d}  ({p['time_str']})", "value": p["idx"]}
            for p in PATCHES[orbit_id]]
    return opts, opts[0]["value"]


@app.callback(
    Output("cache-status", "children"),
    Input("clear-cache-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_cache(n_clicks):
    _render_world_map.cache_clear()
    _render_curtain.cache_clear()
    return "Cache cleared."


@lru_cache(maxsize=16)
def _render_world_map(ec_path):
    try:
        with eck.read_product(ec_path) as ds:
            mfig = eck.MapFigure(
                figsize=(5, 5), dpi=100,
                show_text_time=False,
                show_text_frame=False,
                show_text_overpass=False,
            )
            mfig.ecplot(ds, view="global", colorbar=False,
                        show_text_time=False, show_text_frame=False,
                        show_text_overpass=False)
            return _fig_to_b64(mfig.fig, dpi=100)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                fontsize=10, color="red", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_b64(fig)


@app.callback(
    Output("world-map", "src"),
    Input("orbit-dd", "value"),
)
def update_world_map(orbit_id):
    """Render earthcarekit MapFigure showing the EC track on a world map."""
    if orbit_id is None:
        return ""
    ec_path = ORBIT_EC_FILE.get(orbit_id)
    if ec_path is None:
        return ""
    return _render_world_map(ec_path)


def _footprint_rect(p):
    """Return (lats, lons) for a closed rectangle around the patch footprint."""
    la1, la2 = p["lat_min"], p["lat_max"]
    lo1, lo2 = p["lon_min"], p["lon_max"]
    return (
        [la1, la1, la2, la2, la1],
        [lo1, lo2, lo2, lo1, lo1],
    )


@app.callback(
    Output("overview-map", "figure"),
    Input("orbit-dd", "value"),
    Input("patch-dd", "value"),
)
def update_overview_map(orbit_id, selected_idx):
    empty = go.Figure()
    empty.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    if orbit_id is None:
        return empty

    ec_path = ORBIT_EC_FILE.get(orbit_id)
    if ec_path is None:
        return empty

    ec_lats, ec_lons, ec_times = _load_ec_track(ec_path)
    plist = PATCHES.get(orbit_id, [])

    # ── Compute mean EC time per patch and signed time difference ────────
    _EC_EPOCH = datetime(2000, 1, 1)
    time_diffs = []
    for p in plist:
        mask = (
            (ec_lats >= p["lat_min"]) & (ec_lats <= p["lat_max"]) &
            (ec_lons >= p["lon_min"]) & (ec_lons <= p["lon_max"])
        )
        td = 0.0
        if np.any(mask) and p["mtg_time"] != "unknown":
            secs = np.array([(t - _EC_EPOCH).total_seconds() for t in ec_times[mask]])
            mean_ec_time = _EC_EPOCH + timedelta(seconds=float(np.mean(secs)))
            try:
                mtg_dt = datetime.strptime(p["mtg_time"][:19], "%Y-%m-%d %H:%M:%S")
                td = (mean_ec_time - mtg_dt).total_seconds()
            except ValueError:
                td = 0.0
        time_diffs.append(td)

    # Symmetric RdBu normalization
    abs_max = max((abs(td) for td in time_diffs), default=1.0) or 1.0
    rdbu_cmap = matplotlib.colormaps["RdBu"]
    rdbu_norm = mpl_colors.Normalize(vmin=-abs_max, vmax=abs_max)

    # Build Plotly-compatible RdBu colorscale
    rdbu_plotly = []
    for v in np.linspace(0, 1, 11):
        rgba = rdbu_cmap(v)
        rdbu_plotly.append([float(v), f"rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})"])

    fig = go.Figure()

    # Trace 1: EC ground track
    fig.add_trace(go.Scattermap(
        lat=ec_lats.tolist(),
        lon=ec_lons.tolist(),
        mode="lines",
        line=dict(width=2, color="red"),
        name="EC track",
        hoverinfo="skip",
    ))

    # Patch footprints colored by time difference
    center_lats, center_lons, center_custom, center_text = [], [], [], []
    for i, p in enumerate(plist):
        is_sel = p["idx"] == selected_idx
        rect_lats, rect_lons = _footprint_rect(p)
        td = time_diffs[i]

        rgba = rdbu_cmap(rdbu_norm(td))
        fill_color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.4)"

        if is_sel:
            border_color = "black"
            width = 3
        else:
            border_color = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},0.8)"
            width = 1.5

        fig.add_trace(go.Scattermap(
            lat=rect_lats,
            lon=rect_lons,
            mode="lines",
            line=dict(width=width, color=border_color),
            fill="toself",
            fillcolor=fill_color,
            customdata=[p["idx"]] * 5,
            text=[f"Patch {p['idx']:03d} ({p['time_str']}) dt={td:.0f}s"] * 5,
            hovertemplate="%{text}<extra></extra>",
            name=f"Patch {p['idx']:03d}" if is_sel else "",
            showlegend=is_sel,
        ))
        center_lats.append(p["center_lat"])
        center_lons.append(p["center_lon"])
        center_custom.append(p["idx"])
        center_text.append(f"Patch {p['idx']:03d} ({p['time_str']}) dt={td:.0f}s")

    # Invisible center markers with colorbar for time difference
    if center_lats:
        fig.add_trace(go.Scattermap(
            lat=center_lats,
            lon=center_lons,
            mode="markers",
            marker=dict(
                size=20,
                color=time_diffs,
                colorscale=rdbu_plotly,
                cmin=-abs_max,
                cmax=abs_max,
                opacity=0,
                colorbar=dict(title="EC \u2212 MTG (s)", thickness=15, len=0.5),
            ),
            customdata=center_custom,
            text=center_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

    # Zoom to include all footprints (not just track)
    all_lats = [p["lat_min"] for p in plist] + [p["lat_max"] for p in plist]
    all_lons = [p["lon_min"] for p in plist] + [p["lon_max"] for p in plist]
    lat_min, lat_max = min(all_lats), max(all_lats)
    lon_min, lon_max = min(all_lons), max(all_lons)
    max_range = max(lat_max - lat_min, lon_max - lon_min)
    zoom = max(1, min(10, float(np.log2(360 / (max_range + 10)))))

    # Direction arrow on EC track: filter to visible region, draw V-shape arrowhead
    _margin = max(lat_max - lat_min, lon_max - lon_min) * 0.3
    _vis_mask = (
        (ec_lats >= lat_min - _margin) & (ec_lats <= lat_max + _margin) &
        (ec_lons >= lon_min - _margin) & (ec_lons <= lon_max + _margin)
    )
    _vis_lats = ec_lats[_vis_mask]
    _vis_lons = ec_lons[_vis_mask]
    if len(_vis_lats) >= 6:
        _mid = len(_vis_lats) // 2
        _off = max(2, len(_vis_lats) // 6)
        _dlat = float(_vis_lats[_mid + _off] - _vis_lats[_mid - _off])
        _dlon = float(_vis_lons[_mid + _off] - _vis_lons[_mid - _off])
        _norm = np.sqrt(_dlat**2 + _dlon**2) + 1e-10
        _dlat /= _norm
        _dlon /= _norm
        _plat, _plon = -_dlon, _dlat  # perpendicular unit vector
        _arrow_len = (lat_max - lat_min + lon_max - lon_min) * 0.04
        _wing = _arrow_len * 0.5
        _tip_lat, _tip_lon = float(_vis_lats[_mid]), float(_vis_lons[_mid])
        _base_lat = _tip_lat - _dlat * _arrow_len
        _base_lon = _tip_lon - _dlon * _arrow_len
        fig.add_trace(go.Scattermap(
            lat=[_base_lat + _plat * _wing, _tip_lat, _base_lat - _plat * _wing],
            lon=[_base_lon + _plon * _wing, _tip_lon, _base_lon - _plon * _wing],
            mode="lines",
            line=dict(width=3, color="red"),
            hoverinfo="skip",
            showlegend=False,
        ))

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(
                lat=float((lat_min + lat_max) / 2),
                lon=float((lon_min + lon_max) / 2),
            ),
            zoom=zoom,
        ),
        margin=dict(l=0, r=60, t=30, b=0),
        title=dict(text=f"Orbit {orbit_id} — {len(plist)} patches",
                   font=dict(size=13)),
        showlegend=False,
        uirevision=orbit_id,
    )
    return fig


@app.callback(
    Output("patch-dd", "value", allow_duplicate=True),
    Input("overview-map", "clickData"),
    prevent_initial_call=True,
)
def handle_map_click(click_data):
    if click_data is None:
        return no_update
    point = click_data["points"][0]
    custom = point.get("customdata")
    if custom is None:
        return no_update
    return int(custom)


@app.callback(
    Output("mtg-img", "src"),
    Output("patch-info", "children"),
    Input("orbit-dd", "value"),
    Input("patch-dd", "value"),
    Input("channel-dd", "value"),
)
def update_mtg_plot(orbit_id, selected_idx, channel):
    if orbit_id is None or selected_idx is None or channel is None:
        return "", ""

    plist = PATCHES.get(orbit_id, [])
    pinfo = next((p for p in plist if p["idx"] == selected_idx), None)
    if pinfo is None:
        return "", ""

    ds = _read_patch(pinfo["path"])

    # Read metadata common to all modes
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    center_lat = float(ds.attrs["center_lat"])
    center_lon = float(ds.attrs["center_lon"])
    ec_time = str(ds.attrs["ec_time"])
    mtg_time = str(ds.attrs["mtg_time"])
    time_diff = float(ds.attrs["time_diff_seconds"])

    recipe = _RGB_RECIPES.get(channel)
    if recipe is not None:
        # ── RGB composite ──
        bands = []
        for band_spec in recipe["bands"]:
            try:
                bands.append(_load_band(ds, band_spec))
            except KeyError:
                ds.close()
                missing = band_spec if isinstance(band_spec, str) else f"{band_spec[0]}-{band_spec[1]}"
                return "", f"Band {missing} not found — re-run pipeline with updated config"
        ds.close()

        bands = _match_resolution(bands)
        rgb = np.stack(bands, axis=-1)  # (H, W, 3)
        for i in range(3):
            band = rgb[:, :, i]
            valid = band[~np.isnan(band)]
            if len(valid) > 0:
                # Use global percentiles if available, else fall back to per-patch
                bspec = recipe["bands"][i]
                stat_key = f"{bspec[0]}-{bspec[1]}" if isinstance(bspec, tuple) else bspec
                if stat_key in _BAND_STATS:
                    lo, hi = _BAND_STATS[stat_key]["p2"], _BAND_STATS[stat_key]["p98"]
                else:
                    lo, hi = np.percentile(valid, [2, 98])
                rgb[:, :, i] = np.clip(band, lo, hi)
                rgb[:, :, i] = (rgb[:, :, i] - lo) / (hi - lo + 1e-10)
                g = recipe["gamma"][i]
                if g != 1.0:
                    rgb[:, :, i] = np.power(rgb[:, :, i], 1.0 / g)
            else:
                rgb[:, :, i] = 0
        rgb = np.nan_to_num(rgb, nan=0.0)
        # flip Y for correct orientation in imshow
        img = np.flip(rgb, axis=0)
        title_label = recipe["label"]
    else:
        # ── Single-channel grayscale ──
        var_name = f"effective_radiance_{channel}"
        if var_name not in ds.data_vars:
            ds.close()
            return "", "Channel not found"
        img = ds[var_name].values
        img = np.flip(img, axis=0)  # flip Y for correct orientation in imshow
        ds.close()
        title_label = channel.upper()

    info_text = (f"Patch {selected_idx:03d} | "
                 f"Center: ({center_lat:.2f}, {center_lon:.2f}) | "
                 f"EC: {ec_time[:19]} | MTG: {mtg_time[:19]} | "
                 f"dt: {time_diff:.0f}s")

    # ── Render with matplotlib ──
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    if recipe is not None:

        ax.imshow(img, origin="upper", aspect="equal")
    else:
        vmin, vmax = (np.nanpercentile(img[~np.isnan(img)], [2, 98])
                      if not np.all(np.isnan(img)) else (0, 1))
        ax.imshow(img, cmap="gray_r", vmin=vmin, vmax=vmax,
                  origin="upper", aspect="equal")

    # EC track overlay: lat/lon → pixel coords
    ny, nx = img.shape[:2]
    ec_path = ORBIT_EC_FILE.get(orbit_id)
    if ec_path is not None:
        ec_lats, ec_lons, _ = _load_ec_track(ec_path)
        lat_min = float(np.nanmin(lat))
        lat_max = float(np.nanmax(lat))
        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        mask = ((ec_lats >= lat_min) & (ec_lats <= lat_max) &
                (ec_lons >= lon_min) & (ec_lons <= lon_max))
        if np.any(mask):
            px = (ec_lons[mask] - lon_min) / (lon_max - lon_min) * (nx - 1)
            py = (lat_max - ec_lats[mask]) / (lat_max - lat_min) * (ny - 1)
            ax.plot(px, py, color="red", linewidth=2, label="EC track")
            # Direction arrow at track midpoint
            if len(px) >= 6:
                _mid = len(px) // 2
                _off = max(2, len(px) // 6)
                ax.annotate('', xy=(px[_mid + _off], py[_mid + _off]),
                            xytext=(px[_mid - _off], py[_mid - _off]),
                            arrowprops=dict(arrowstyle='->', color='red',
                                            lw=2.5, mutation_scale=20),
                            zorder=101)
            ax.legend(loc="upper right", fontsize=8)

    ax.set_title(f"{title_label}  |  Patch {selected_idx:03d}", fontsize=10)
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.tight_layout()
    return _fig_to_b64(fig, dpi=100), info_text


@lru_cache(maxsize=32)
def _render_curtain(ec_path, ec_var, selected_ec_time):
    """Render curtain figure. selected_ec_time is a string or None for highlight."""
    selection_kw = {}
    if selected_ec_time is not None:
        try:
            t = datetime.strptime(selected_ec_time, "%Y-%m-%d %H:%M:%S")
            selection_kw = dict(
                selection_time_range=(
                    (t - timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S"),
                    (t + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S"),
                ),
                selection_highlight=True,
                selection_highlight_alpha=0.4,
            )
        except Exception:
            pass

    try:
        with eck.read_product(ec_path) as ds:
            filtered_cmap, keep_values = _make_filtered_cmap(ec_var)
            mask_kw = {}
            if filtered_cmap is not None and keep_values:
                vals = ds[ec_var].values.copy().astype(np.float32)
                vals[~np.isin(ds[ec_var].values, list(keep_values))] = np.nan
                mask_kw = dict(values=vals, cmap=filtered_cmap)

            cf = eck.CurtainFigure(figsize=(14, 3.5), dpi=120)
            cf.ecplot(ds, var=ec_var,
                      height_range=(0, 20e3), colorbar=True,
                      show_info=False, **mask_kw, **selection_kw)
            cf.ecplot_elevation(ds)
            return _fig_to_b64(cf.fig, dpi=120)
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center",
                fontsize=10, color="red", transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_b64(fig)


@app.callback(
    Output("curtain-img", "src"),
    Input("orbit-dd", "value"),
    Input("ecvar-dd", "value"),
    Input("patch-dd", "value"),
)
def update_curtain(orbit_id, ec_var, selected_idx):
    if orbit_id is None or ec_var is None:
        return ""

    ec_path = ORBIT_EC_FILE.get(orbit_id)
    if ec_path is None:
        return ""

    selected_ec_time = None
    if selected_idx is not None:
        plist = PATCHES.get(orbit_id, [])
        pinfo = next((p for p in plist if p["idx"] == selected_idx), None)
        if pinfo is not None:
            selected_ec_time = pinfo["ec_time"][:19]

    return _render_curtain(ec_path, ec_var, selected_ec_time)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=8050)
