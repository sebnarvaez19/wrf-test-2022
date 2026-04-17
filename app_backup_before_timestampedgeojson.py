from __future__ import annotations

import time
from pathlib import Path

import branca.colormap as bcm
import folium
import numpy as np
import streamlit as st
import xarray as xr
from streamlit_folium import st_folium

DATA_DIR = Path(__file__).resolve().parent / "data"
DOMAIN_FILES = {
    "1st Domain (d01)": DATA_DIR / "wrfout_d01_merged.nc",
    "2nd Domain (d02)": DATA_DIR / "wrfout_d02_merged.nc",
}
VARIABLES = {
    "RAINNC": "Total rainfall (mm)",
    "T2": "2m temperature (deg C)",
}


def compute_bounds(ds: xr.Dataset, ny: int, nx: int) -> list[list[float]]:
    cen_lat = float(ds.attrs.get("CEN_LAT", 0.0))
    cen_lon = float(ds.attrs.get("CEN_LON", 0.0))
    dx = float(ds.attrs.get("DX", 9000.0))
    dy = float(ds.attrs.get("DY", 9000.0))

    lat_half = (ny * dy / 2.0) / 111_320.0
    lon_scale = max(np.cos(np.radians(cen_lat)), 1e-6)
    lon_half = (nx * dx / 2.0) / (111_320.0 * lon_scale)

    south = cen_lat - lat_half
    north = cen_lat + lat_half
    west = cen_lon - lon_half
    east = cen_lon + lon_half
    return [[south, west], [north, east]]


def normalize_grid(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmax, vmin):
        return np.zeros_like(arr, dtype=float)
    clipped = np.clip(arr, vmin, vmax)
    return (clipped - vmin) / (vmax - vmin)


def rgba_grid(norm: np.ndarray, variable: str) -> np.ndarray:
    cmap = bcm.linear.YlGnBu_09.scale(0, 1) if variable == "RAINNC" else bcm.linear.RdYlBu_11.scale(0, 1)
    rgba = np.empty((*norm.shape, 4), dtype=np.uint8)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            hex_color = cmap(float(norm[i, j])).lstrip("#")
            rgba[i, j] = [int(hex_color[k : k + 2], 16) for k in (0, 2, 4, 6)]
    return rgba


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


@st.cache_data(show_spinner=False)
def get_display_scale(path: str, variable: str) -> tuple[float, float]:
    ds = xr.open_dataset(path)
    arr = ds[variable].values.astype(float)
    if variable == "T2":
        arr = arr - 273.15
    vmin = float(np.nanpercentile(arr, 5))
    vmax = float(np.nanpercentile(arr, 95))
    ds.close()
    return vmin, vmax


def make_map(
    ds: xr.Dataset,
    da: xr.DataArray,
    variable: str,
    time_label: str,
    domain_label: str,
    vmin: float,
    vmax: float,
) -> folium.Map:
    grid = np.asarray(da.values, dtype=float)
    norm = normalize_grid(grid, vmin, vmax)
    rgba = rgba_grid(norm, variable)

    ny, nx = grid.shape
    bounds = compute_bounds(ds, ny, nx)
    center_lat = float(ds.attrs.get("CEN_LAT", 0.0))
    center_lon = float(ds.attrs.get("CEN_LON", 0.0))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")
    folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=0.78,
        interactive=False,
        cross_origin=False,
        zindex=3,
    ).add_to(m)

    cm = bcm.linear.YlGnBu_09.scale(vmin, vmax) if variable == "RAINNC" else bcm.linear.RdYlBu_11.scale(vmin, vmax)
    cm.caption = "RAINNC (mm)" if variable == "RAINNC" else "T2 (deg C)"
    cm.add_to(m)

    title_html = f"""
        <div style=\"position: fixed; top: 10px; left: 50px; z-index: 9999;
             background: white; padding: 8px 10px; border-radius: 6px;
             box-shadow: 0 0 6px rgba(0,0,0,0.2); font-size: 12px;\">
            <b>{domain_label}</b><br>{variable} at {time_label}
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def main() -> None:
    st.set_page_config(page_title="WRF NetCDF Dashboard", layout="wide")
    st.title("WRF Dashboard")
    st.caption("Interactive map of RAINNC and T2 from d01/d02 NetCDF outputs")

    domain_label = st.sidebar.selectbox("Domain", list(DOMAIN_FILES.keys()), index=0)
    variable = st.sidebar.selectbox(
        "Variable", list(VARIABLES.keys()), index=0, format_func=lambda v: f"{v} - {VARIABLES[v]}"
    )
    playback_speed = st.sidebar.slider("Animation speed (ms/frame)", 150, 2000, 450, 50)

    ds_path = str(DOMAIN_FILES[domain_label])
    ds = load_dataset(ds_path)
    if "XTIME" not in ds:
        st.error("XTIME was not found in dataset.")
        return
    if variable not in ds:
        st.error(f"Variable {variable} not found in dataset.")
        return

    times = ds["XTIME"].values
    nsteps = len(times)

    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0

    c1, c2, c3, c4 = st.columns([1, 1, 1, 5])
    with c1:
        if st.button("Play", use_container_width=True):
            st.session_state.playing = True
    with c2:
        if st.button("Pause", use_container_width=True):
            st.session_state.playing = False
    with c3:
        if st.button("Next", use_container_width=True):
            st.session_state.frame_idx = (int(st.session_state.frame_idx) + 1) % nsteps

    frame_idx = st.slider(
        "Time index",
        min_value=0,
        max_value=nsteps - 1,
        value=int(st.session_state.frame_idx),
        step=1,
        format="%d",
    )
    st.session_state.frame_idx = frame_idx

    current_time = np.datetime_as_string(times[frame_idx], unit="h")
    da = ds[variable].isel(time=frame_idx)
    if variable == "T2":
        da = da - 273.15

    vmin, vmax = get_display_scale(ds_path, variable)
    m = make_map(ds, da, variable, current_time, domain_label, vmin, vmax)
    st_folium(m, width=None, height=700, returned_objects=[])

    st.write(f"**Selected time:** {current_time}:00")
    st.write(f"**Selected variable:** {variable} ({VARIABLES[variable]})")

    if st.session_state.playing:
        time.sleep(playback_speed / 1000.0)
        st.session_state.frame_idx = (frame_idx + 1) % nsteps
        st.rerun()


if __name__ == "__main__":
    main()
