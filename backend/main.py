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
# 1. MODEL ARCHITECTURE (The Brain)
# ==========================================
class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. CONFIGURATION & GLOBALS
# ==========================================

# Maps River ID -> File Paths
RIVER_CONFIG = {
    "netravati": {
        "csv": "river.csv", 
        "shp": "shapefiles/netra_river_shape.shp"
    },
    "kali": {
        "csv": "river_data/kali.csv", 
        "shp": "shapefiles/Kali_Shapefile.shp" 
    },
    "sharavathi": {
        "csv": "river_data/sharavathi.csv", 
        "shp": "shapefiles/Sharavathi_Shapefile.shp" 
    },
}

# Global Storage
systems = {} 
model = None
scaler = None

# ==========================================
# 3. STARTUP: LOAD EVERYTHING
# ==========================================
@app.on_event("startup")
async def load_system():
    global systems, model, scaler
    print("🚀 Loading Data for Multiple Rivers...")

    # A. Load Rivers
    for river_id, paths in RIVER_CONFIG.items():
        print(f"   🌊 Loading {river_id}...")
        sys_data = {'df': None, 'poly': None, 'tree': None}
        
        # 1. Load Shapefile (Required for Map)
        try:
            if os.path.exists(paths['shp']):
                gdf = gpd.read_file(paths['shp'])
                sys_data['poly'] = gdf.geometry.unary_union
                print(f"      ✅ Shapefile Loaded")
            else:
                print(f"      ⚠️ Shapefile Missing: {paths['shp']}")
        except Exception as e:
            print(f"      ❌ Shapefile Error: {e}")

        # 2. Load CSV (Required for AI)
        try:
            if os.path.exists(paths['csv']):
                df = pd.read_csv(paths['csv'])
                df.columns = df.columns.str.lower().str.strip()
                lat_col = next((c for c in df.columns if c in ['lat', 'latitude']), 'latitude')
                lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude']), 'longitude')
                
                valid_coords = df[[lat_col, lon_col]].dropna()
                sys_data['tree'] = cKDTree(valid_coords.values)
                sys_data['df'] = df
                print(f"      ✅ CSV Data Loaded: {len(df)} points")
            else:
                print(f"      ℹ️ No CSV yet. Skipping data.")
        except Exception as e:
            print(f"      ❌ CSV Error: {e}")
        
        systems[river_id] = sys_data

    # B. Load AI Model & Scaler
    try:
        scaler = joblib.load("scaler.pkl")
        model = GATModel(in_channels=4, hidden_channels=32, out_channels=1)
        checkpoint = torch.load("gat_k20_drop0.3_unc4_best.pt", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("✅ AI Model Loaded (Ready for Live Prediction)")
    except Exception as e:
        print(f"❌ Model Error: {e}")

# ==========================================
# 4. API ENDPOINTS
# ==========================================

@app.get("/api/bounds")
def get_bounds(river: str = "netravati"):
    sys = systems.get(river)
    if not sys or not sys['poly']: return {}
    minx, miny, maxx, maxy = sys['poly'].bounds
    return {"min_lat": miny, "min_lng": minx, "max_lat": maxy, "max_lng": maxx}

@app.get("/api/get-heatmap")
async def get_heatmap(river: str = "netravati"):
    sys = systems.get(river)
    if not sys or sys['df'] is None or sys['poly'] is None:
        return Response(content="No Data", status_code=404)

    try:
        df = sys['df']
        poly = sys['poly']
        minx, miny, maxx, maxy = poly.bounds
        grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]
        
        points = df[['longitude', 'latitude']].values
        val_col = 'predicted_salinity' if 'predicted_salinity' in df.columns else 'ai_salinity_prediction'
        values = df[val_col].values if val_col in df.columns else np.random.rand(len(df))

        # Interpolate
        grid_sal = griddata(points, values, (grid_x, grid_y), method='cubic')
        
        # --- SAFETY LOCK RESTORED ---
        grid_sal[grid_sal < 0] = 0  # Force negatives to 0
        # ----------------------------
        
        # Masking
        flat_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
        poly_coords = np.array(poly.exterior.coords)
        if poly_coords.shape[1] > 2: poly_coords = poly_coords[:, :2]
        
        mask = Path(poly_coords).contains_points(flat_points).reshape(grid_x.shape)
        grid_sal[~mask] = np.nan
        
        vmin, vmax = np.nanpercentile(grid_sal, 5), np.nanpercentile(grid_sal, 95)

        plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        ax.imshow(grid_sal.T, extent=(minx, maxx, miny, maxy), origin='lower', cmap='plasma', alpha=0.8, vmin=vmin, vmax=vmax)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        print(traceback.format_exc())
        return Response(content=str(e), status_code=500)
@app.get("/api/get-nearest")
def get_nearest(lat: float, lng: float, river: str = "netravati"):
    print(f"\n--- 🕵️ DEBUG CLICK AT {lat}, {lng} ---")
    sys = systems.get(river)
    
    if not sys or not sys['tree']: 
        return {"found": False, "message": "Data not loaded"}

    dist, idx = sys['tree'].query([lat, lng], k=1)
    row = sys['df'].iloc[idx].replace({np.nan: None}).to_dict()

    # 1. PRINT ALL COLUMN NAMES (Check spelling!)
    print(f"📂 Columns found in CSV: {list(sys['df'].columns)}")

    real_prediction = 0.0
    
    try:
        if model and scaler:
            # 2. EXTRACT BANDS & PRINT THEM
            # We try every possible spelling
            b3 = row.get('band_3', row.get('b3', row.get('band3', 0)))
            b4 = row.get('band_4', row.get('b4', row.get('band4', 0)))
            b5 = row.get('band_5', row.get('b5', row.get('band5', 0)))
            b7 = row.get('band_7', row.get('b7', row.get('band7', 0)))
            
            print(f"📊 Inputs Found -> B3: {b3}, B4: {b4}, B5: {b5}, B7: {b7}")

            if b3 == 0 and b4 == 0:
                print("⚠️ WARNING: Bands are 0! Check your CSV column names.")

            raw_inputs = [b3, b4, b5, b7]
            scaled = scaler.transform([raw_inputs])
            x_tensor = torch.tensor(scaled, dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            with torch.no_grad():
                raw_out = model(x_tensor, edge_index).item()
                print(f"🧠 AI Raw Output: {raw_out}")
                
                # 3. TEMPORARY: DISABLE SAFETY LOCK to see the real number
                real_prediction = raw_out 
                
    except Exception as e:
        print(f"❌ Error: {e}")
        real_prediction = 0.0

    print(f"🚀 Sending to Frontend: {real_prediction}")

    return {
        "found": True,
        "latitude": row.get('latitude', lat), 
        "longitude": row.get('longitude', lng),
        "ai_salinity_prediction": real_prediction,
        "history": [] 
    }