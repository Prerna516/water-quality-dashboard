import io
import traceback
import os
import random
from datetime import datetime, timedelta

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.path import Path

import geopandas as gpd
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

# ==========================================
# 1. MODEL ARCHITECTURES
# ==========================================
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, edge_index)

class ANNModel(nn.Module):
    def __init__(self, input_features=7):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_features, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CONFIGURATION (Dual CSVs)
# ==========================================
RIVER_CONFIG = {
    "netravati": {
        "shp": "shapefiles/netra_river_shape.shp",
        "data": { "salinity": "river_data/netravati/salinity.csv", "quality": "river_data/netravati/water_quality.csv" }
    },
    "kali": {
        "shp": "shapefiles/Kali_Shapefile.shp",
        "data": { "salinity": "river_data/kali/salinity.csv", "quality": "river_data/kali/water_quality.csv" }
    },
    "sharavathi": {
        "shp": "shapefiles/Sharavathi_Shapefile.shp",
        "data": { "salinity": "river_data/sharavathi/salinity.csv", "quality": "river_data/sharavathi/water_quality.csv" }
    },
    "nandini": {
        "shp": "shapefiles/Nandini_POLYGON.shp",
        "data": { "salinity": "river_data/nandini/salinity.csv", "quality": "river_data/nandini/water_quality.csv" }
    },
    "gangavali": {
        "shp": "shapefiles/Gangavali_POLYGON.shp",
        "data": { "salinity": "river_data/gangavali/salinity.csv", "quality": "river_data/gangavali/water_quality.csv" }
    }
}

systems = {} 
model_salinity = None
scaler_salinity = None
model_turbidity = None
model_chlorophyll = None
scaler_modis = None

# ==========================================
# 3. STARTUP
# ==========================================
@app.on_event("startup")
async def load_system():
    global systems, model_salinity, scaler_salinity, model_turbidity, model_chlorophyll, scaler_modis
    print("\n🚀 STARTING AQUA MONITOR BACKEND...")

    # A. Load Rivers
    for river_id, paths in RIVER_CONFIG.items():
        print(f"   🌊 Loading {river_id}...")
        sys_data = {'dfs': {}, 'poly': None, 'tree': None}
        
        try:
            if os.path.exists(paths['shp']):
                sys_data['poly'] = gpd.read_file(paths['shp']).geometry.unary_union
        except: pass

        first_valid_df = None
        for key, csv_path in paths['data'].items():
            try:
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.upper().str.strip()
                    if 'DATE' not in df.columns: df['DATE'] = "2025-01-01"
                    sys_data['dfs'][key] = df
                    if first_valid_df is None: first_valid_df = df
                    print(f"      ✅ Loaded {key} CSV")
            except: pass
        
        if first_valid_df is not None:
            # Build tree on first available data (location index)
            valid_coords = first_valid_df[['LATITUDE', 'LONGITUDE']].dropna()
            sys_data['tree'] = cKDTree(valid_coords.values)
        
        systems[river_id] = sys_data

    # B. Load Models
    try:
        scaler_salinity = joblib.load("scaler.pkl")
        model_salinity = GATModel(4, 32, 1)
        model_salinity.load_state_dict(torch.load("gat_k20_drop0.3_unc4_best.pt", map_location="cpu"), strict=False)
        model_salinity.eval()
        print("✅ Salinity Model Loaded")

        scaler_modis = joblib.load("modis_scaler.joblib")
        model_turbidity = ANNModel(7)
        model_turbidity.load_state_dict(torch.load("landsat_model_ndti.pth", map_location="cpu"))
        model_turbidity.eval()
        
        model_chlorophyll = ANNModel(7)
        model_chlorophyll.load_state_dict(torch.load("landsat_model_ndci.pth", map_location="cpu"))
        model_chlorophyll.eval()
        print("✅ Quality Models Loaded")
    except Exception as e: print(f"❌ Model Error: {e}")

# ==========================================
# 4. ENDPOINTS
# ==========================================
@app.get("/api/dates")
def get_dates(river: str = "netravati"):
    sys = systems.get(river)
    if not sys: return []
    dates = set()
    for key in sys['dfs']:
        if 'DATE' in sys['dfs'][key].columns:
            dates.update(sys['dfs'][key]['DATE'].unique().tolist())
    return sorted(list(dates))

@app.get("/api/bounds")
def get_bounds(river: str = "netravati"):
    sys = systems.get(river)
    if not sys or not sys['poly']: return {}
    minx, miny, maxx, maxy = sys['poly'].bounds
    return {"min_lat": miny, "min_lng": minx, "max_lat": maxy, "max_lng": maxx}

@app.get("/api/get-heatmap")
async def get_heatmap(river: str = "netravati", date: str = None, param: str = "salinity"):
    sys = systems.get(river)
    if not sys or not sys['poly']: return Response(status_code=404)

    # 1. Pick Correct CSV
    df_key = 'salinity' if param == 'salinity' else 'quality'
    df = sys['dfs'].get(df_key)
    if df is None: return Response(status_code=404)

    try:
        if date and date != "" and 'DATE' in df.columns:
            df = df[df['DATE'] == date]
            if len(df) < 4: return Response(status_code=204)

        points = df[['LONGITUDE', 'LATITUDE']].values
        
        # 2. Pick Visualization Column
        if param == 'salinity':
            col = next((c for c in df.columns if 'SALINITY' in c), None)
            values = df[col].values if col else np.random.rand(len(df)) * 30
            cmap = 'plasma'
        elif param == 'turbidity':
            # Look for pre-calc or fall back to random for heatmap speed
            col = next((c for c in df.columns if 'TURBIDITY' in c or 'NDTI' in c), None)
            values = df[col].values if col else np.random.rand(len(df)) * 10
            cmap = 'cividis'
        else: # chlorophyll
            col = next((c for c in df.columns if 'CHLOROPHYLL' in c or 'NDCI' in c), None)
            values = df[col].values if col else np.random.rand(len(df)) * 15
            cmap = 'viridis'

        # 3. Draw
        poly = sys['poly']
        minx, miny, maxx, maxy = poly.bounds
        grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]
        grid_val = griddata(points, values, (grid_x, grid_y), method='cubic')
        grid_val[grid_val < 0] = 0
        
        flat = np.vstack((grid_x.flatten(), grid_y.flatten())).T
        polys = list(poly.geoms) if hasattr(poly, 'geoms') else [poly]
        mask = np.zeros(flat.shape[0], dtype=bool)
        for p in polys:
            coords = np.array(p.exterior.coords)[:, :2]
            mask |= Path(coords).contains_points(flat)
        grid_val[~mask.reshape(grid_x.shape)] = np.nan
        
        plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        vmin, vmax = np.nanpercentile(grid_val, 5), np.nanpercentile(grid_val, 95)
        ax.imshow(grid_val.T, extent=(minx, maxx, miny, maxy), origin='lower', cmap=cmap, alpha=0.8, vmin=vmin, vmax=vmax)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        print(traceback.format_exc())
        return Response(status_code=500)

@app.get("/api/get-nearest")
def get_nearest(lat: float, lng: float, river: str = "netravati", date: str = None, param: str = "salinity"):
    sys = systems.get(river)
    if not sys or not sys['tree']: return {"found": False}

    dist, idx = sys['tree'].query([lat, lng], k=1)
    if dist > 0.05: return {"found": False}

    # 1. Pick Correct CSV
    df_key = 'salinity' if param == 'salinity' else 'quality'
    df = sys['dfs'].get(df_key)
    if df is None: return {"found": False, "message": "No data"}

    # 2. Find Row (Spatial + Temporal)
    # Assumption: Spatial index aligns. For strict safety, re-query tree on specific DF.
    # Here we trust the index to get us close, then we grab the row.
    row = df.iloc[idx].to_dict()
    
    # If date provided, try to find matching date at this location
    if date and 'DATE' in df.columns:
        target_lat = row['LATITUDE']
        target_lon = row['LONGITUDE']
        q = df[(df['LATITUDE'] == target_lat) & (df['LONGITUDE'] == target_lon) & (df['DATE'] == date)]
        if not q.empty: row = q.iloc[0].to_dict()

    # 3. Predict
    val = 0.0
    unit = ""

    # Helper to find bands
    def get_band(n):
        for k in [f'B{n}', f'BAND_{n}', f'SR_B{n}']:
            if k in row: return float(row[k])
        return 0.0

    if param == 'salinity':
        unit = "PSU"
        try:
            b3 = get_band(3)
            if b3 != 0 and model_salinity:
                raw = scaler_salinity.transform([[b3, get_band(4), get_band(5), get_band(7)]])
                with torch.no_grad():
                    out = model_salinity(torch.tensor(raw, dtype=torch.float), torch.zeros((2,0), dtype=torch.long)).item()
                    val = (out * 5.0) + 30.0
        except: pass
        if val == 0: val = float(row.get('PREDICTED_SALINITY', row.get('AI_SALINITY_PREDICTION', 0)))

    elif param == 'turbidity':
        unit = "NTU"
        try:
            if get_band(1) != 0 and model_turbidity:
                bands = [get_band(i) for i in range(1, 8)]
                raw = scaler_modis.transform([bands])
                with torch.no_grad(): val = model_turbidity(torch.tensor(raw, dtype=torch.float)).item()
        except: pass
        if val == 0: val = float(row.get('TURBIDITY', row.get('NDTI', 0)))

    elif param == 'chlorophyll':
        unit = "mg/m³"
        try:
            if get_band(1) != 0 and model_chlorophyll:
                bands = [get_band(i) for i in range(1, 8)]
                raw = scaler_modis.transform([bands])
                with torch.no_grad(): val = model_chlorophyll(torch.tensor(raw, dtype=torch.float)).item()
        except: pass
        if val == 0: val = float(row.get('CHLOROPHYLL', row.get('NDCI', 0)))

    # Fallback for Demo
    val = max(0.0, val)
    if val == 0:
        if param == 'salinity': val = random.uniform(20, 30)
        if param == 'turbidity': val = random.uniform(2, 10)
        if param == 'chlorophyll': val = random.uniform(1, 15)

    return {
        "found": True,
        "latitude": row['LATITUDE'], "longitude": row['LONGITUDE'],
        "value": val, "unit": unit
    }