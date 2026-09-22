"""
AI-Based Landslide Early Warning & Risk Monitoring Dashboard
Prototype for North Eastern Region (NER), India

Run with:
    streamlit run app.py
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium


# =========================================================
# OPTIONAL AUTO REFRESH
# =========================================================

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREFRESH = True
except ImportError:
    HAS_AUTOREFRESH = False


# =========================================================
# SECRETS / CONFIG
# =========================================================

def get_secret(name: str, default: str = "") -> str:
    """
    Read value from:
    1. Streamlit secrets
    2. Environment variable
    3. Default value
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.environ.get(name, default)


# =========================================================
# THINGSPEAK CONFIGURATION
# =========================================================

# Existing Channel ID from your previous dashboard
THINGSPEAK_CHANNEL_ID = get_secret(
    "THINGSPEAK_CHANNEL_ID",
    "3484539"
)

# NEW THINGSPEAK READ API KEY
THINGSPEAK_READ_API_KEY = get_secret(
    "THINGSPEAK_READ_API_KEY",
    "4BQ5MLQ7IU0MSDYM"
)

# ThingSpeak API URL
THINGSPEAK_URL = (
    f"https://api.thingspeak.com/channels/"
    f"{THINGSPEAK_CHANNEL_ID}/feeds.json"
)

# Cache live readings for 60 seconds
RISK_REFRESH_SECONDS = 60


# =========================================================
# MAPBOX CONFIGURATION
# =========================================================

MAPBOX_ACCESS_TOKEN = get_secret(
    "MAPBOX_ACCESS_TOKEN",
    ""
)

MAPBOX_STYLES = {
    "Outdoors (Terrain)": "outdoors-v12",
    "Satellite": "satellite-streets-v12",
    "Dark": "dark-v11",
}


def mapbox_tile_url(style_id: str) -> str:
    return (
        "https://api.mapbox.com/styles/v1/mapbox/"
        f"{style_id}/tiles/{{z}}/{{x}}/{{y}}"
        f"?access_token={MAPBOX_ACCESS_TOKEN}"
    )


MAPBOX_ATTR = (
    '© <a href="https://www.mapbox.com/about/maps/">Mapbox</a> '
    '© <a href="https://www.openstreetmap.org/copyright">'
    'OpenStreetMap</a> '
    '<strong><a href="https://www.mapbox.com/map-feedback/" '
    'target="_blank">Improve this map</a></strong>'
)


# =========================================================
# FREE BASEMAPS
# =========================================================

FREE_BASEMAPS = {

    "OpenStreetMap": {
        "tiles": "OpenStreetMap",
        "attr": None,
    },

    "Terrain (OpenTopoMap)": {
        "tiles": (
            "https://{s}.tile.opentopomap.org/"
            "{z}/{x}/{y}.png"
        ),
        "attr": (
            'Map data: © <a href="https://www.openstreetmap.org/copyright">'
            'OpenStreetMap</a> contributors, SRTM | '
            'Map style: © <a href="https://opentopomap.org">'
            'OpenTopoMap</a> (CC-BY-SA)'
        ),
    },

    "Satellite (Esri World Imagery)": {
        "tiles": (
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        "attr": (
            "Tiles © Esri — Source: Esri, Maxar, "
            "Earthstar Geographics, and the GIS User Community"
        ),
    },

    "Dark (CartoDB)": {
        "tiles": "CartoDB dark_matter",
        "attr": None,
    },
}


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="NER Landslide Early Warning System",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DARK THEME
# =========================================================

DARK_CSS = """
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            #10141c 0%,
            #0a0d12 60%,
            #05070a 100%
        );
    color: #e6edf3;
}

h1, h2, h3, h4 {
    color: #f0f4f8 !important;
    font-family: 'Segoe UI', sans-serif;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #161b22,
        #0d1117
    );

    border: 1px solid #2a3441;
    padding: 18px 12px;
    border-radius: 14px;

    box-shadow:
        0 0 18px rgba(0, 200, 255, 0.05);

    transition: all 0.3s ease-in-out;
}

[data-testid="stMetric"]:hover {
    box-shadow:
        0 0 22px rgba(0, 200, 255, 0.25);

    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: #8b98a5 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0d1117 0%,
            #05070a 100%
        );

    border-right: 1px solid #1f2630;
}

@keyframes pulseGlow {

    0% {
        box-shadow: 0 0 6px currentColor;
    }

    50% {
        box-shadow: 0 0 24px currentColor;
    }

    100% {
        box-shadow: 0 0 6px currentColor;
    }
}

.alert-box {
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 1px;
    animation: pulseGlow 2s infinite;
    margin-bottom: 18px;
}

.alert-green {
    background: #0d1f14;
    color: #3ddc84;
    border: 2px solid #3ddc84;
}

.alert-yellow {
    background: #241f0a;
    color: #ffd93d;
    border: 2px solid #ffd93d;
}

.alert-orange {
    background: #2b1608;
    color: #ff9f45;
    border: 2px solid #ff9f45;
}

.alert-red {
    background: #2a0a0a;
    color: #ff4b4b;
    border: 2px solid #ff4b4b;
}

.sub-caption {
    color: #7d8a97;
    font-size: 13px;
}

.data-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-left: 8px;
}

.tag-live {
    background: #0d1f14;
    color: #3ddc84;
    border: 1px solid #3ddc84;
}

.tag-sim {
    background: #241f0a;
    color: #ffd93d;
    border: 1px solid #ffd93d;
}

@keyframes fadeIn {

    from {
        opacity: 0;
        transform: translateY(8px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

.fade-block {
    animation: fadeIn 0.8s ease-in;
}

.live-dot {
    height: 10px;
    width: 10px;
    background-color: #ff4b4b;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
    animation: blink 1.2s infinite;
}

@keyframes blink {

    0%, 100% {
        opacity: 1;
    }

    50% {
        opacity: 0.2;
    }

}

</style>
"""

st.markdown(
    DARK_CSS,
    unsafe_allow_html=True
)


# =========================================================
# MONITORING STATIONS
# =========================================================

NER_STATIONS = {

    "Sohra (Cherrapunji), Meghalaya": {
        "lat": 25.2840,
        "lon": 91.7273,
        "slope_deg": 38,
        "base_risk": 0.55
    },

    "Aizawl, Mizoram": {
        "lat": 23.7271,
        "lon": 92.7176,
        "slope_deg": 42,
        "base_risk": 0.60
    },

    "Gangtok, Sikkim": {
        "lat": 27.3389,
        "lon": 88.6065,
        "slope_deg": 47,
        "base_risk": 0.65
    },

    "Kohima, Nagaland": {
        "lat": 25.6751,
        "lon": 94.1086,
        "slope_deg": 40,
        "base_risk": 0.50
    },

    "Itanagar, Arunachal Pradesh": {
        "lat": 27.0844,
        "lon": 93.6053,
        "slope_deg": 36,
        "base_risk": 0.48
    },
}


# Only Aizawl currently receives live ThingSpeak data
LIVE_DATA_STATION = "Aizawl, Mizoram"


# =========================================================
# RISK CALCULATION
# =========================================================

def compute_risk_score(
    rainfall_24h: float,
    soil_moisture: float,
    slope_deg: float
) -> float:

    score = (
        (rainfall_24h / 150) * 50
        + (soil_moisture / 100) * 30
        + (slope_deg / 60) * 20
    )

    return float(
        min(100, max(0, score))
    )


def risk_level(score: float):

    if score < 30:
        return "LOW", "green", "🟢"

    elif score < 55:
        return "MODERATE", "yellow", "🟡"

    elif score < 75:
        return "HIGH", "red", "🔴"

    else:
        return "SEVERE", "red", "🔴"


# =========================================================
# SIMULATED READING
# =========================================================

def _simulated_reading():

    base_risk = st.session_state.get(
        "base_risk",
        random.uniform(0.15, 0.5)
    )

    rainfall_24h = max(
        0.0,
        np.random.normal(
            40 + base_risk * 60,
            15
        )
    )

    soil_moisture = min(
        100.0,
        max(
            0.0,
            np.random.normal(
                50 + base_risk * 30,
                8
            )
        )
    )

    return rainfall_24h, soil_moisture


# =========================================================
# THINGSPEAK LIVE SENSOR DATA
# =========================================================

@st.cache_data(
    ttl=RISK_REFRESH_SECONDS,
    show_spinner=False
)
def get_sensor_data():

    """
    ThingSpeak configuration:

    Field 1 = Soil Moisture
    Field 2 = Rainfall
    """

    if not THINGSPEAK_READ_API_KEY:
        return _simulated_reading() + (False,)

    try:

        params = {
            "api_key": THINGSPEAK_READ_API_KEY,
            "results": 1
        }

        response = requests.get(
            THINGSPEAK_URL,
            params=params,
            timeout=8
        )

        response.raise_for_status()

        data = response.json()

        feeds = data.get("feeds") or []

        if not feeds:
            return _simulated_reading() + (False,)

        feed = feeds[0]

        # -------------------------------------------------
        # THINGSPEAK FIELD MAPPING
        # Field 1 = Moisture
        # Field 2 = Rainfall
        # -------------------------------------------------

        field1 = feed.get("field1")
        field2 = feed.get("field2")

        if field1 is None or field2 is None:
            return _simulated_reading() + (False,)

        soil_moisture = float(field1)
        rainfall = float(field2)

        return (
            rainfall,
            soil_moisture,
            True
        )

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError
    ) as e:

        st.session_state.setdefault(
            "last_error",
            str(e)
        )

        return _simulated_reading() + (False,)


# =========================================================
# STATION READING
# =========================================================

@st.cache_data(
    ttl=RISK_REFRESH_SECONDS,
    show_spinner=False
)
def get_station_reading(
    station_name: str,
    is_live_capable: bool
):

    info = NER_STATIONS[station_name]

    if is_live_capable:

        rainfall, moisture, is_live = (
            get_sensor_data()
        )

    else:

        rng = np.random.default_rng(
            abs(hash(station_name)) % (2**32)
        )

        base_risk = info["base_risk"]

        rainfall = max(
            0.0,
            rng.normal(
                40 + base_risk * 60,
                15
            )
        )

        moisture = min(
            100.0,
            max(
                0.0,
                rng.normal(
                    50 + base_risk * 30,
                    8
                )
            )
        )

        is_live = False

    score = compute_risk_score(
        rainfall,
        moisture,
        info["slope_deg"]
    )

    return (
        rainfall,
        moisture,
        score,
        is_live
    )


# =========================================================
# HISTORICAL DATA
# =========================================================

def generate_history(
    hours: int,
    base_risk: float,
    slope_deg: float
) -> pd.DataFrame:

    now = datetime.now()

    timestamps = [
        now - timedelta(hours=h)
        for h in range(hours)
    ][::-1]

    rainfall_list = []
    moisture_list = []
    risk_list = []

    rng = np.random.default_rng(42)

    for i in range(hours):

        local_risk = (
            base_risk
            + 0.15 * np.sin(i / 6)
        )

        rainfall = max(
            0.0,
            rng.normal(
                40 + local_risk * 60,
                15
            )
        )

        moisture = min(
            100.0,
            max(
                0.0,
                rng.normal(
                    50 + local_risk * 30,
                    8
                )
            )
        )

        score = compute_risk_score(
            rainfall,
            moisture,
            slope_deg
        )

        rainfall_list.append(rainfall)
        moisture_list.append(moisture)
        risk_list.append(score)

    return pd.DataFrame({

        "timestamp": timestamps,

        "rainfall": rainfall_list,

        "soil_moisture": moisture_list,

        "risk_score": risk_list

    })


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## ⛰️ Control Panel"
    )

    station = st.selectbox(
        "📍 Select Monitoring Station",
        list(NER_STATIONS.keys())
    )

    st.markdown("---")

    simulate_storm = st.toggle(
        "⛈️ Simulate Heavy Rainfall Event",
        value=False
    )

    auto_refresh = st.toggle(
        f"🔄 Live Auto-Refresh "
        f"({RISK_REFRESH_SECONDS}s)",
        value=False
    )

    st.markdown("---")

    st.markdown(
        "### 🗺️ Basemap"
    )

    basemap_options = list(
        FREE_BASEMAPS.keys()
    )

    if MAPBOX_ACCESS_TOKEN:

        basemap_options = (
            [
                f"Mapbox {k}"
                for k in MAPBOX_STYLES.keys()
            ]
            + basemap_options
        )

    map_style_label = st.selectbox(
        "Map style",
        basemap_options,
        index=0
    )

    if not MAPBOX_ACCESS_TOKEN:

        with st.expander(
            "Want Mapbox terrain/satellite styling too? (optional)"
        ):

            st.markdown(
                """
The map already works with free map tiles.

If you want Mapbox styling:

1. Create a Mapbox account.
2. Generate an access token.
3. Add it as `MAPBOX_ACCESS_TOKEN`.
4. Reload the dashboard.
"""
            )

    st.markdown("---")

    st.caption(
        "Data sources: LIVE reading from "
        "ThingSpeak for "
        f"**{LIVE_DATA_STATION}**. "
        "Other stations and the 48h trend "
        "use simulated data."
    )

    if not THINGSPEAK_READ_API_KEY:

        st.warning(
            "No ThingSpeak API key set — "
            "all readings are simulated.",
            icon="⚠️"
        )

    else:

        st.success(
            "ThingSpeak Read API connected",
            icon="✅"
        )

    st.markdown("---")

    st.markdown(
        '<span class="live-dot"></span> '
        '**System Status: ONLINE**',
        unsafe_allow_html=True
    )


# =========================================================
# SESSION STATE
# =========================================================

st.session_state["base_risk"] = (
    0.85
    if simulate_storm
    else random.uniform(0.15, 0.5)
)


# =========================================================
# AUTO REFRESH
# =========================================================

if auto_refresh:

    if HAS_AUTOREFRESH:

        st_autorefresh(
            interval=RISK_REFRESH_SECONDS * 1000,
            key="auto_refresh_tick"
        )

    else:

        st.sidebar.info(
            "Install streamlit-autorefresh for "
            "automatic refresh."
        )

        st.sidebar.button(
            "🔄 Refresh now"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    "<div class='fade-block'>",
    unsafe_allow_html=True
)

col_title, col_time = st.columns(
    [3, 1]
)

with col_title:

    st.title(
        "🛰️ AI-Based Landslide Early Warning System"
    )

    st.markdown(
        "<span class='sub-caption'>"
        "North Eastern Region (NER), India — "
        "Prototype Dashboard"
        "</span>",
        unsafe_allow_html=True
    )


with col_time:

    st.markdown(
        f"""
        <div style='text-align:right; padding-top:20px;'>
        <span class='live-dot'></span>
        {datetime.now().strftime(
            '%d %b %Y, %H:%M:%S'
        )}
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    "</div>",
    unsafe_allow_html=True
)

st.markdown("---")


# =========================================================
# CURRENT READING
# =========================================================

station_info = NER_STATIONS[station]

is_live_capable = (
    station == LIVE_DATA_STATION
)

(
    rainfall_24h,
    soil_moisture,
    risk_score,
    is_live
) = get_station_reading(
    station,
    is_live_capable
)

slope_angle = station_info["slope_deg"]


level, color, emoji = risk_level(
    risk_score
)


if is_live:

    data_tag = (
        '<span class="data-tag tag-live">'
        'LIVE'
        '</span>'
    )

else:

    data_tag = (
        '<span class="data-tag tag-sim">'
        'SIMULATED'
        '</span>'
    )


st.markdown(
    f"""
    <div class='alert-box alert-{color}'>
        {emoji}
        CURRENT ALERT LEVEL: {level}
        &nbsp;|&nbsp;
        Risk Score: {risk_score:.1f}/100
    </div>

    <div style='text-align:center;
                margin-top:-10px;
                margin-bottom:14px;'>
        {data_tag}
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ERROR MESSAGE
# =========================================================

if (
    "last_error" in st.session_state
    and not is_live
):

    st.caption(
        "⚠️ Live fetch issue "
        "(using simulated fallback): "
        + st.session_state["last_error"]
    )


# =========================================================
# METRIC CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "🌧️ Rainfall (24h)",
    f"{rainfall_24h:.1f} mm",
    delta=f"{rainfall_24h - 40:.1f} mm vs avg"
)


c2.metric(
    "💧 Soil Moisture",
    f"{soil_moisture:.1f} %",
    delta=f"{soil_moisture - 50:.1f}%"
)


c3.metric(
    "⛰️ Slope Angle",
    f"{slope_angle:.1f}°"
)


c4.metric(
    "📍 Station",
    station.split(",")[0]
)


st.markdown("---")


# =========================================================
# MAP + GAUGE
# =========================================================

map_col, gauge_col = st.columns(
    [2, 1]
)


# =========================================================
# REGIONAL MAP
# =========================================================

with map_col:

    st.subheader(
        "🗺️ Regional Risk Map"
    )

    if (
        map_style_label.startswith("Mapbox ")
        and MAPBOX_ACCESS_TOKEN
    ):

        style_key = (
            map_style_label.replace(
                "Mapbox ",
                ""
            )
        )

        style_id = MAPBOX_STYLES[
            style_key
        ]

        m = folium.Map(

            location=[
                26.0,
                92.5
            ],

            zoom_start=6,

            tiles=mapbox_tile_url(
                style_id
            ),

            attr=MAPBOX_ATTR
        )

        map_source_caption = (
            map_style_label
        )

    else:

        basemap = FREE_BASEMAPS[
            map_style_label
        ]

        if basemap["attr"]:

            m = folium.Map(

                location=[
                    26.0,
                    92.5
                ],

                zoom_start=6,

                tiles=basemap["tiles"],

                attr=basemap["attr"]
            )

        else:

            m = folium.Map(

                location=[
                    26.0,
                    92.5
                ],

                zoom_start=6,

                tiles=basemap["tiles"]
            )

        map_source_caption = (
            f"{map_style_label} "
            "(free, no API key)"
        )


    color_map = {

        "green": "#3ddc84",

        "yellow": "#ffd93d",

        "red": "#ff4b4b"
    }


    RISK_AREA_MIN_RADIUS_M = 4000

    RISK_AREA_MAX_RADIUS_M = 18000


    for name, info in NER_STATIONS.items():

        s_is_live_capable = (
            name == LIVE_DATA_STATION
        )

        (
            _,
            _,
            s_risk,
            s_is_live
        ) = get_station_reading(
            name,
            s_is_live_capable
        )

        lvl, clr, _ = risk_level(
            s_risk
        )

        tag = (
            "LIVE"
            if s_is_live
            else "sim"
        )


        area_radius = (
            RISK_AREA_MIN_RADIUS_M
            + (s_risk / 100)
            * (
                RISK_AREA_MAX_RADIUS_M
                - RISK_AREA_MIN_RADIUS_M
            )
        )


        # Risk area

        folium.Circle(

            location=[
                info["lat"],
                info["lon"]
            ],

            radius=area_radius,

            color=color_map[clr],

            weight=1,

            fill=True,

            fill_color=color_map[clr],

            fill_opacity=(
                0.30
                if clr == "red"
                else 0.15
            ),

            popup=(
                f"{name} — Risk Area<br>"
                f"Level: {lvl} "
                f"({s_risk:.0f}) "
                f"[{tag}]"
            )

        ).add_to(m)


        # Station marker

        folium.CircleMarker(

            location=[
                info["lat"],
                info["lon"]
            ],

            radius=(
                14
                if name == station
                else 9
            ),

            popup=(
                f"{name}<br>"
                f"Risk: {lvl} "
                f"({s_risk:.0f}) "
                f"[{tag}]"
            ),

            color=color_map[clr],

            fill=True,

            fill_color=color_map[clr],

            fill_opacity=0.9,

            weight=(
                2
                if name == station
                else 1
            )

        ).add_to(m)


    st_folium(
        m,
        height=420,
        width=None
    )


    st.caption(
        f"Basemap: {map_source_caption}"
    )


    st.caption(
        "🔴 Red = HIGH/SEVERE risk "
        "&nbsp;|&nbsp; "
        "🟡 Yellow = MODERATE risk "
        "&nbsp;|&nbsp; "
        "🟢 Green = LOW/safe risk."
    )


# =========================================================
# RISK GAUGE
# =========================================================

with gauge_col:

    st.subheader(
        "📊 Risk Gauge"
    )


    fig_gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=risk_score,

            number={
                "suffix": "",
                "font": {
                    "color": "white",
                    "size": 40
                }
            },

            gauge={

                "axis": {
                    "range": [0, 100],
                    "tickcolor": "white"
                },

                "bar": {
                    "color": "#00d4ff"
                },

                "bgcolor": "rgba(0,0,0,0)",

                "steps": [

                    {
                        "range": [0, 30],
                        "color": "#0d1f14"
                    },

                    {
                        "range": [30, 55],
                        "color": "#241f0a"
                    },

                    {
                        "range": [55, 75],
                        "color": "#2b1608"
                    },

                    {
                        "range": [75, 100],
                        "color": "#2a0a0a"
                    }

                ],

                "threshold": {

                    "line": {
                        "color": "red",
                        "width": 4
                    },

                    "thickness": 0.8,

                    "value": 75
                }

            }

        )
    )


    fig_gauge.update_layout(

        paper_bgcolor="rgba(0,0,0,0)",

        font={
            "color": "white"
        },

        height=380,

        margin=dict(
            t=30,
            b=10
        )
    )


    st.plotly_chart(
        fig_gauge,
        use_container_width=True
    )


st.markdown("---")


# =========================================================
# HISTORICAL TRENDS
# =========================================================

st.subheader(
    "📈 48-Hour Trend — Rainfall, Soil Moisture & Risk Score"
)

st.caption(
    "Simulated trend (placeholder for "
    "historical IMD rainfall + sensor records)."
)


hist_df = generate_history(

    48,

    st.session_state["base_risk"],

    slope_angle
)


fig = go.Figure()


# Rainfall

fig.add_trace(

    go.Scatter(

        x=hist_df["timestamp"],

        y=hist_df["rainfall"],

        name="Rainfall (mm)",

        line=dict(
            color="#00d4ff",
            width=2
        ),

        fill="tozeroy",

        fillcolor=(
            "rgba(0,212,255,0.08)"
        )
    )
)


# Soil moisture

fig.add_trace(

    go.Scatter(

        x=hist_df["timestamp"],

        y=hist_df["soil_moisture"],

        name="Soil Moisture (%)",

        line=dict(
            color="#ffd93d",
            width=2
        )
    )
)


# Risk score

fig.add_trace(

    go.Scatter(

        x=hist_df["timestamp"],

        y=hist_df["risk_score"],

        name="Risk Score",

        line=dict(
            color="#ff4b4b",
            width=3,
            dash="dot"
        )
    )
)


fig.update_layout(

    paper_bgcolor="rgba(0,0,0,0)",

    plot_bgcolor="rgba(0,0,0,0)",

    font=dict(
        color="white"
    ),

    legend=dict(

        orientation="h",

        yanchor="bottom",

        y=1.02,

        xanchor="right",

        x=1
    ),

    xaxis=dict(
        gridcolor="#1f2630"
    ),

    yaxis=dict(
        gridcolor="#1f2630"
    ),

    height=420,

    margin=dict(
        t=30,
        b=10
    )
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =========================================================
# ALERT LOG
# =========================================================

st.subheader(
    "🔔 Recent Alert Log"
)


log_df = hist_df.tail(8).copy()


log_df["level"] = (
    log_df["risk_score"]
    .apply(
        lambda x: risk_level(x)[0]
    )
)


log_df = log_df[
    [
        "timestamp",
        "rainfall",
        "soil_moisture",
        "risk_score",
        "level"
    ]
]


log_df.columns = [

    "Time",

    "Rainfall (mm)",

    "Soil Moisture (%)",

    "Risk Score",

    "Alert Level"

]


st.dataframe(

    log_df.iloc[::-1],

    use_container_width=True,

    hide_index=True
)
