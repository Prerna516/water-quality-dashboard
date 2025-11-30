import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# CONFIGURATION
CSV_PATH = "river_data/netravati.csv" # Your restored real file
SHP_PATH = "shapefiles/netra_river_shape.shp"

print(f"🕵️ Checking overlap between {CSV_PATH} and {SHP_PATH}...")

try:
    # 1. Load Data
    df = pd.read_csv(CSV_PATH)
    # Clean column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Find lat/lon
    lat_col = next((c for c in df.columns if 'lat' in c), None)
    lon_col = next((c for c in df.columns if 'lon' in c or 'lng' in c), None)
    
    if not lat_col or not lon_col:
        print("❌ Error: Could not find Latitude/Longitude columns in CSV.")
        print(f"   Columns found: {list(df.columns)}")
        exit()

    # 2. Load Shapefile
    gdf_shape = gpd.read_file(SHP_PATH)
    poly = gdf_shape.geometry.unary_union
    
    # 3. Check Bounds
    minx, miny, maxx, maxy = poly.bounds
    print(f"\n🗺️  Shapefile Bounds: Lat {miny:.4f}-{maxy:.4f}, Lon {minx:.4f}-{maxx:.4f}")
    
    data_min_lat = df[lat_col].min()
    data_max_lat = df[lat_col].max()
    data_min_lon = df[lon_col].min()
    data_max_lon = df[lon_col].max()
    
    print(f"📍 CSV Data Bounds: Lat {data_min_lat:.4f}-{data_max_lat:.4f}, Lon {data_min_lon:.4f}-{data_max_lon:.4f}")

    # 4. Check Overlap
    points_inside = 0
    for _, row in df.iterrows():
        pnt = Point(row[lon_col], row[lat_col])
        if poly.contains(pnt):
            points_inside += 1
            
    print(f"\n📉 Result: {points_inside} out of {len(df)} points are inside the shapefile.")
    
    if points_inside == 0:
        print("❌ CRITICAL FAILURE: Your CSV data is geographically OUTSIDE the shapefile.")
        print("   The heatmap is generating, but it's empty because no data falls in the river.")
    else:
        print("✅ Data looks good! The issue might be in the code logic.")

except Exception as e:
    print(f"❌ Script Error: {e}")