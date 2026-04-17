from __future__ import annotations

from pathlib import Path

import branca.colormap as bcm
import folium
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from folium.plugins import TimestampedGeoJson
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
RETURN_PERIOD_CSV = DATA_DIR / "return_period.csv"
RAINFALL_CSV = DATA_DIR / "rainfall.csv"
PIOJO_LAT = 10.746
PIOJO_LON = -75.108


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


def build_lat_lon_grid(bounds: list[list[float]], ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    (south, west), (north, east) = bounds
    lat_step = (north - south) / max(ny - 1, 1)
    lon_step = (east - west) / max(nx - 1, 1)
    lats = np.linspace(south, north, ny)
    lons = np.linspace(west, east, nx)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lat2d, lon2d, lat_step, lon_step


def normalize_value(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if np.isclose(vmin, vmax):
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> xr.Dataset:
    return xr.open_dataset(path)


@st.cache_data(show_spinner=False)
def load_return_period_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_rainfall_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return df


@st.cache_data(show_spinner=False)
def extract_point_series(path: str, variable: str, target_lat: float, target_lon: float) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    if variable not in ds or "XTIME" not in ds:
        ds.close()
        return pd.DataFrame(columns=["timestamp", "value"])

    cube = ds[variable].values.astype(float)
    ny, nx = cube.shape[1], cube.shape[2]
    bounds = compute_bounds(ds, ny, nx)
    lat2d, lon2d, _, _ = build_lat_lon_grid(bounds, ny, nx)

    # Nearest grid cell to requested lat/lon.
    dist2 = (lat2d - target_lat) ** 2 + (lon2d - target_lon) ** 2
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)

    series = cube[:, iy, ix]
    times = pd.to_datetime(ds["XTIME"].values, errors="coerce")
    if getattr(times, "tz", None) is not None:
        times = times.tz_convert(None)
    ds.close()
    return pd.DataFrame({"timestamp": times, "value": series})


@st.cache_data(show_spinner=False)
def build_timestamped_geojson(path: str, variable: str, sample_step: int, cell_scale: float) -> dict[str, object]:
    ds = xr.open_dataset(path)
    cube = ds[variable].values.astype(float)
    if variable == "T2":
        cube = cube - 273.15

    times = ds["XTIME"].values
    ny, nx = cube.shape[1], cube.shape[2]
    bounds = compute_bounds(ds, ny, nx)
    lat2d, lon2d, lat_step, lon_step = build_lat_lon_grid(bounds, ny, nx)

    vmin = float(np.nanpercentile(cube, 5))
    vmax = float(np.nanpercentile(cube, 95))
    cmap = bcm.linear.YlGnBu_09.scale(vmin, vmax) if variable == "RAINNC" else bcm.linear.RdYlBu_11_r.scale(vmin, vmax)

    half_lat = (lat_step * sample_step * cell_scale) / 2.0
    half_lon = (lon_step * sample_step * cell_scale) / 2.0

    features: list[dict] = []
    for t in range(cube.shape[0]):
        grid = cube[t]
        norm = normalize_value(grid, vmin, vmax)
        time_iso = np.datetime_as_string(times[t], unit="s") + "Z"

        for i in range(0, ny, sample_step):
            for j in range(0, nx, sample_step):
                val = float(grid[i, j])
                if not np.isfinite(val):
                    continue
                lat = float(lat2d[i, j])
                lon = float(lon2d[i, j])
                color = cmap(val)
                opacity = 0.45 + 0.50 * float(norm[i, j])

                polygon = [
                    [lon - half_lon, lat - half_lat],
                    [lon + half_lon, lat - half_lat],
                    [lon + half_lon, lat + half_lat],
                    [lon - half_lon, lat + half_lat],
                    [lon - half_lon, lat - half_lat],
                ]

                features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [polygon]},
                        "properties": {
                            "times": [time_iso],
                            "style": {
                                "color": color,
                                "fillColor": color,
                                "weight": 0.12,
                                "fillOpacity": opacity,
                            },
                            "popup": f"{variable}: {val:.2f}",
                        },
                    }
                )

    ds.close()
    return {
        "geojson": {"type": "FeatureCollection", "features": features},
        "vmin": vmin,
        "vmax": vmax,
    }


def make_map(ds: xr.Dataset, variable: str, domain_label: str, payload: dict[str, object], frame_ms: int) -> folium.Map:
    center_lat = float(ds.attrs.get("CEN_LAT", 0.0))
    center_lon = float(ds.attrs.get("CEN_LON", 0.0))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    TimestampedGeoJson(
        data=payload["geojson"],
        transition_time=frame_ms,
        period="PT1H",
        duration="PT1S",
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=6,
        loop_button=True,
        date_options="YYYY-MM-DD HH:mm",
        time_slider_drag_update=True,
    ).add_to(m)

    cm = (
        bcm.linear.YlGnBu_09.scale(payload["vmin"], payload["vmax"])
        if variable == "RAINNC"
        else bcm.linear.RdYlBu_11_r.scale(payload["vmin"], payload["vmax"])
    )
    cm.caption = "RAINNC (mm)" if variable == "RAINNC" else "T2 (deg C)"
    cm.add_to(m)

    title_html = f"""
        <div style=\"position: fixed; top: 10px; left: 50px; z-index: 9999;
             background: white; padding: 8px 10px; border-radius: 6px;
             box-shadow: 0 0 6px rgba(0,0,0,0.2); font-size: 12px;\">
            <b>{domain_label}</b><br>{variable} (default timestamp controls)
        </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    locations = [
        ("Barranquilla", 10.988, -74.792),
        ("Piojo", 10.746, -75.108),
        ("Hibacharo", 10.722, -75.140),
    ]
    for name, lat, lon in locations:
        folium.Marker(
            location=[lat, lon],
            tooltip=name,
            popup=f"{name} ({lat:.3f}, {lon:.3f})",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    return m


def main() -> None:
    st.set_page_config(page_title="WRF NetCDF Dashboard", layout="wide")
    st.title("Rainfall Piojó Nov 2022 (WRF Model)")

    domain_label = st.sidebar.selectbox("Domain", list(DOMAIN_FILES.keys()), index=1)
    variable = st.sidebar.selectbox("Variable", list(VARIABLES.keys()), index=0, format_func=lambda v: f"{v} - {VARIABLES[v]}")
    sample_step = st.sidebar.slider("Spatial sampling (higher = lighter)", 1, 8, 4, 1)
    cell_scale = st.sidebar.slider("Cell size factor", 0.6, 2.6, 1.0, 0.05)
    frame_ms = st.sidebar.slider("Transition time (ms)", 150, 1800, 250, 50)

    ds_path = str(DOMAIN_FILES[domain_label])
    ds = load_dataset(ds_path)
    if "XTIME" not in ds:
        st.error("XTIME was not found in dataset.")
        return
    if variable not in ds:
        st.error(f"Variable {variable} not found in dataset.")
        return

    payload = build_timestamped_geojson(ds_path, variable, sample_step, cell_scale)
    m = make_map(ds, variable, domain_label, payload, frame_ms)
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st_folium(m, width=None, height=720, returned_objects=[])
    with right_col:
        st.subheader("Rainfall registered [14010010]")
        if RAINFALL_CSV.exists():
            rain_df = load_rainfall_table(str(RAINFALL_CSV))
            if {"timestamp", "value"}.issubset(rain_df.columns):
                rain_sorted = rain_df.sort_values("timestamp")
                piojo_df = extract_point_series(ds_path, "RAINNC", PIOJO_LAT, PIOJO_LON)

                fig, ax = plt.subplots(figsize=(6, 3.1))
                ax.bar(rain_sorted["timestamp"], rain_sorted["value"], width=0.7, color="#1f77b4", alpha=0.75, label="Observed rainfall")

                if not piojo_df.empty:
                    # RAINNC is cumulative, convert to hourly increments: x_t - x_(t-1).
                    piojo_inc = piojo_df.copy()
                    piojo_inc["increment"] = piojo_inc["value"].diff().fillna(piojo_inc["value"])
                    piojo_inc["increment"] = piojo_inc["increment"].clip(lower=0.0)

                    # Aggregate model increments to daily totals for direct overlay with daily bars.
                    piojo_daily = (
                        piojo_inc.set_index("timestamp")[["increment"]]
                        .resample("D")
                        .sum()
                        .reset_index()
                        .rename(columns={"increment": "value"})
                    )
                    merged = pd.merge(rain_sorted[["timestamp"]], piojo_daily, on="timestamp", how="left")
                    ax.plot(
                        merged["timestamp"],
                        merged["value"],
                        color="#d62728",
                        linewidth=2.2,
                        marker="o",
                        markersize=4,
                        label="WRF RAINNC at Piojo",
                    )

                ax.set_xlabel("Time")
                ax.set_ylabel("Rainfall [mm]")
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
                ax.legend(loc="upper right", fontsize=8, frameon=True)
                fig.autofmt_xdate(rotation=0, ha="center")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            else:
                st.warning("`rainfall.csv` must include `timestamp` and `value` columns.")
        else:
            st.warning("`rainfall.csv` was not found in the data folder.")

        st.subheader("Return Period Table")
        if RETURN_PERIOD_CSV.exists():
            rp_df = load_return_period_table(str(RETURN_PERIOD_CSV))
            st.dataframe(rp_df, width="stretch", hide_index=True)
        else:
            st.warning("`return_period.csv` was not found in the data folder.")

    times = ds["XTIME"].values
    st.write(f"**Range:** {np.datetime_as_string(times[0], unit='h')}:00 to {np.datetime_as_string(times[-1], unit='h')}:00")
    st.write(f"**Variable:** {variable} ({VARIABLES[variable]})")


if __name__ == "__main__":
    main()
