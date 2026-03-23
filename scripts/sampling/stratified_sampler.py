import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import folium
import os
import json
from scripts.utils.logger import get_logger
from scripts.utils.config_loader import load_config

logger = get_logger("stratified_sampler")
config = load_config()

# ── Stratum definitions ───────────────────────────────────────────────────────
# Each stratum has:
#   - count     : number of coordinates to generate
#   - lat/lon ranges : bounding boxes that best represent that land type globally
#   - description : what energy story this stratum tells

STRATA = {
    "dense_urban": {
        "count": 750,
        "regions": [
            # Western Europe
            {"lat": (48.0, 53.0), "lon": (2.0, 15.0)},
            # Eastern Europe
            {"lat": (44.0, 52.0), "lon": (15.0, 30.0)},
            # North America East Coast
            {"lat": (35.0, 45.0), "lon": (-80.0, -70.0)},
            # North America West Coast
            {"lat": (32.0, 48.0), "lon": (-125.0, -115.0)},
            # East Asia — Japan, South Korea
            {"lat": (34.0, 38.0), "lon": (129.0, 141.0)},
            # South Asia — India
            {"lat": (12.0, 28.0), "lon": (72.0, 88.0)},
            # Southeast Asia
            {"lat": (1.0, 15.0), "lon": (100.0, 120.0)},
            # Middle East
            {"lat": (24.0, 32.0), "lon": (44.0, 56.0)},
            # South America
            {"lat": (-25.0, -15.0), "lon": (-50.0, -40.0)},
            # Africa
            {"lat": (-5.0, 10.0), "lon": (5.0, 40.0)},
        ]
    },
    "industrial": {
        "count": 750,
        "regions": [
            # Germany Ruhr
            {"lat": (51.0, 52.0), "lon": (6.5, 8.0)},
            # Poland industrial
            {"lat": (49.0, 51.0), "lon": (18.0, 22.0)},
            # US Midwest industrial
            {"lat": (41.0, 43.0), "lon": (-88.0, -80.0)},
            # China industrial belt
            {"lat": (30.0, 40.0), "lon": (110.0, 122.0)},
            # Russia industrial
            {"lat": (54.0, 60.0), "lon": (56.0, 65.0)},
            # Gulf oil infrastructure
            {"lat": (24.0, 28.0), "lon": (48.0, 56.0)},
            # Australia mining
            {"lat": (-25.0, -20.0), "lon": (116.0, 122.0)},
            # South Africa mining
            {"lat": (-27.0, -25.0), "lon": (26.0, 30.0)},
            # Brazil industrial
            {"lat": (-24.0, -20.0), "lon": (-48.0, -43.0)},
            # India industrial
            {"lat": (20.0, 25.0), "lon": (72.0, 80.0)},
        ]
    },
    "suburban": {
        "count": 750,
        "regions": [
            # US suburbs
            {"lat": (33.0, 42.0), "lon": (-95.0, -75.0)},
            # Western Europe suburbs
            {"lat": (47.0, 52.0), "lon": (-2.0, 12.0)},
            # Australia suburbs
            {"lat": (-38.0, -30.0), "lon": (144.0, 153.0)},
            # Japan suburbs
            {"lat": (34.0, 36.0), "lon": (135.0, 140.0)},
            # South America suburbs
            {"lat": (-35.0, -20.0), "lon": (-60.0, -45.0)},
            # South Africa suburbs
            {"lat": (-30.0, -25.0), "lon": (27.0, 32.0)},
        ]
    },
    "agricultural": {
        "count": 500,
        "regions": [
            # US Midwest farmland
            {"lat": (38.0, 48.0), "lon": (-100.0, -85.0)},
            # European farmland
            {"lat": (46.0, 52.0), "lon": (0.0, 20.0)},
            # India agricultural
            {"lat": (20.0, 30.0), "lon": (74.0, 85.0)},
            # China agricultural
            {"lat": (28.0, 38.0), "lon": (108.0, 120.0)},
            # Argentina Pampas
            {"lat": (-38.0, -28.0), "lon": (-65.0, -55.0)},
            # Sub-Saharan Africa
            {"lat": (5.0, 15.0), "lon": (10.0, 35.0)},
            # Ukraine/Russia breadbasket
            {"lat": (46.0, 52.0), "lon": (28.0, 40.0)},
            # Kazakhstan steppes
            {"lat": (45.0, 55.0), "lon": (55.0, 80.0)},  
            # Mexico
            {"lat": (18.0, 28.0), "lon": (-110.0, -95.0)},  
        ]
    },
    "forest": {
        "count": 500,
        "regions": [
            # Amazon rainforest
            {"lat": (-8.0, 2.0), "lon": (-70.0, -50.0)},
            # Congo basin
            {"lat": (-5.0, 5.0), "lon": (15.0, 28.0)},
            # Boreal forest Canada
            {"lat": (50.0, 60.0), "lon": (-110.0, -80.0)},
            # Boreal forest Russia
            {"lat": (55.0, 65.0), "lon": (60.0, 100.0)},
            # Southeast Asia forest
            {"lat": (-5.0, 5.0), "lon": (100.0, 118.0)},
            # Scandinavia forest
            {"lat": (60.0, 68.0), "lon": (15.0, 28.0)},
            # Siberian forest
            {"lat": (45.0, 55.0), "lon": (80.0, 110.0)},  
            # Great Lakes forest Canada
            {"lat": (45.0, 50.0), "lon": (-90.0, -75.0)},  
        ]
    },
    "coastal": {
        "count": 500,
        "regions": [
            # North Sea wind zones
            {"lat": (53.0, 58.0), "lon": (3.0, 10.0)},
            # Mediterranean coast
            {"lat": (36.0, 44.0), "lon": (0.0, 20.0)},
            # US Atlantic coast
            {"lat": (35.0, 42.0), "lon": (-76.0, -68.0)},
            # Southeast Asia coast
            {"lat": (0.0, 15.0), "lon": (98.0, 115.0)},
            # West Africa coast
            {"lat": (4.0, 14.0), "lon": (-18.0, 2.0)},
            # Australia coast
            {"lat": (-38.0, -25.0), "lon": (113.0, 154.0)},
            # Chile coast
            {"lat": (-45.0, -30.0), "lon": (-76.0, -70.0)},
        ]
    },
    "arid": {
        "count": 500,
        "regions": [
            # Sahara desert
            {"lat": (18.0, 30.0), "lon": (-10.0, 25.0)},
            # Arabian peninsula
            {"lat": (20.0, 28.0), "lon": (44.0, 58.0)},
            # Southwest USA
            {"lat": (32.0, 38.0), "lon": (-118.0, -108.0)},
            # Atacama desert
            {"lat": (-28.0, -18.0), "lon": (-72.0, -65.0)},
            # Central Australia outback
            {"lat": (-28.0, -20.0), "lon": (128.0, 138.0)},
            # Gobi desert
            {"lat": (38.0, 46.0), "lon": (98.0, 112.0)},
            # Iran/Pakistan arid
            {"lat": (26.0, 34.0), "lon": (56.0, 68.0)},
            # Horn of Africa/Sudan
            {"lat": (15.0, 25.0), "lon": (35.0, 45.0)},
            # Central Asia/Mongolia
            {"lat": (38.0, 48.0), "lon": (80.0, 100.0)},
        ]
    },
    "alpine": {
        "count": 375,
        "regions": [
            # Alps
            {"lat": (45.0, 47.5), "lon": (6.0, 14.0)},
            # Himalayas
            {"lat": (27.0, 35.0), "lon": (75.0, 90.0)},
            # Andes
            {"lat": (-22.0, -10.0), "lon": (-70.0, -64.0)},
            # Rocky Mountains
            {"lat": (40.0, 48.0), "lon": (-115.0, -105.0)},
            # Caucasus
            {"lat": (41.0, 44.0), "lon": (40.0, 48.0)},
            # East Africa highlands
            {"lat": (-3.0, 5.0), "lon": (35.0, 40.0)},
        ]
    },
    "informal_settlements": {
        "count": 375,
        "regions": [
            # Sub-Saharan Africa
            {"lat": (-5.0, 10.0), "lon": (15.0, 40.0)},
            # South Asia slums
            {"lat": (10.0, 25.0), "lon": (72.0, 90.0)},
            # Southeast Asia
            {"lat": (10.0, 20.0), "lon": (100.0, 108.0)},
            # West Africa
            {"lat": (5.0, 15.0), "lon": (-5.0, 10.0)},
            # Latin America
            {"lat": (-25.0, -5.0), "lon": (-70.0, -40.0)},
        ]
    },
    "water_wetland": {
        "count": 250,
        "regions": [
            # Mekong delta
            {"lat": (9.0, 12.0), "lon": (104.0, 107.0)},
            # Nile delta
            {"lat": (30.0, 32.0), "lon": (30.0, 32.0)},
            # Mississippi delta
            {"lat": (28.0, 31.0), "lon": (-92.0, -88.0)},
            # Ganges delta
            {"lat": (21.0, 24.0), "lon": (88.0, 92.0)},
            # Okavango delta
            {"lat": (-20.0, -18.0), "lon": (22.0, 24.0)},
            # Amazon river basin
            {"lat": (-5.0, 0.0), "lon": (-62.0, -50.0)},
            # Scandinavia lakes
            {"lat": (60.0, 65.0), "lon": (25.0, 30.0)},
        ]
    }
}


def generate_bbox(lat, lon, size_m=256):
    """Generate a 256x256m bounding box around a coordinate."""
    delta_lat = (size_m / 2) / 111320
    delta_lon = (size_m / 2) / (111320 * np.cos(np.radians(lat)))
    return {
        "min_lat": lat - delta_lat,
        "max_lat": lat + delta_lat,
        "min_lon": lon - delta_lon,
        "max_lon": lon + delta_lon
    }


def is_too_close(lat, lon, existing_points, min_spacing_km=1.0):
    """Check if a point is too close to any existing point."""
    if len(existing_points) == 0:
        return False
    for existing_lat, existing_lon in existing_points:
        dlat = abs(lat - existing_lat) * 111.32
        dlon = abs(lon - existing_lon) * 111.32 * np.cos(np.radians(lat))
        distance = np.sqrt(dlat**2 + dlon**2)
        if distance < min_spacing_km:
            return True
    return False


def sample_stratum(stratum_name, stratum_config, existing_points, min_spacing_km=1.0):
    """Sample coordinates for a single stratum."""
    count = stratum_config["count"]
    regions = stratum_config["regions"]
    points = []
    max_attempts = count * 50
    attempts = 0

    while len(points) < count and attempts < max_attempts:
        # Pick a random region from this stratum
        region = regions[np.random.randint(len(regions))]
        lat = np.random.uniform(region["lat"][0], region["lat"][1])
        lon = np.random.uniform(region["lon"][0], region["lon"][1])

        # Check minimum spacing
        all_points = existing_points + [(p["lat"], p["lon"]) for p in points]
        if not is_too_close(lat, lon, all_points, min_spacing_km):
            bbox = generate_bbox(lat, lon)
            points.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "stratum": stratum_name,
                "bbox_min_lat": round(bbox["min_lat"], 6),
                "bbox_max_lat": round(bbox["max_lat"], 6),
                "bbox_min_lon": round(bbox["min_lon"], 6),
                "bbox_max_lon": round(bbox["max_lon"], 6),
            })
        attempts += 1

    logger.info(f"{stratum_name}: {len(points)}/{count} points generated")
    return points


def generate_all_coordinates():
    """Generate all 5000 stratified coordinates."""
    all_points = []
    existing_points = []

    for stratum_name, stratum_config in STRATA.items():
        logger.info(f"Sampling stratum: {stratum_name}")
        points = sample_stratum(
            stratum_name,
            stratum_config,
            existing_points
        )
        all_points.extend(points)
        existing_points.extend([(p["lat"], p["lon"]) for p in points])

    return all_points


def save_csv(points, output_path):
    """Save coordinates to CSV."""
    df = pd.DataFrame(points)
    df.insert(0, "id", range(1, len(df) + 1))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} coordinates to {output_path}")
    return df


def plot_world_map(df, output_path):
    """Plot all coordinates on a world map using Folium."""
    m = folium.Map(location=[20, 0], zoom_start=2)

    # Color per stratum
    colors = {
        "dense_urban": "red",
        "industrial": "darkred",
        "suburban": "orange",
        "agricultural": "green",
        "forest": "darkgreen",
        "coastal": "blue",
        "arid": "beige",
        "alpine": "purple",
        "informal_settlements": "pink",
        "water_wetland": "darkblue"
    }

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2,
            color=colors.get(row["stratum"], "gray"),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['stratum']} | {row['lat']}, {row['lon']}"
        ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logger.info(f"Map saved to {output_path}")


if __name__ == "__main__":
    logger.info("Starting stratified sampling...")

    points = generate_all_coordinates()
    
    df = save_csv(
        points,
        "/srv/THESIS/energy_profiling_thesis/data/coordinates/coordinates_5000.csv"
    )

    plot_world_map(
        df,
        "/srv/THESIS/energy_profiling_thesis/outputs/maps/world_map_5000.html"
    )

    # Print summary
    print("\n=== Sampling Summary ===")
    print(f"Total coordinates: {len(df)}")
    print("\nPer stratum:")
    print(df["stratum"].value_counts())
    print("\nSample rows:")
    print(df.head(10))