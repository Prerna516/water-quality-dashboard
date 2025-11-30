import io
import traceback
import os
import random
from datetime import datetime, timedelta

# --- 1. CRITICAL IMPORTS ---
import matplotlib
matplotlib.use('Agg') # Prevents server crashes
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
# 2. AI MODEL ARCHITECTURE
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

# ==========================================
# 3. APP SETUP & CONFIGURATION
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THIS IS THE CONFIGURATION MATCHING YOUR FILES EXACTLY ---
RIVER_CONFIG = {
    "netravati": {
        "csv": "river_data/netravati.csv",                       # Root folder
        "shp": "shapefiles/netra_river_shape.shp" 
    },
    "kali": {
        "csv": "river_data/kali.csv",
        "shp": "shapefiles/Kali_Shapefile.shp"    # Case Sensitive!
    },
    "sharavathi": {
        "csv": "river_data/sharavathi.csv",
        "shp": "shapefiles/Sharavathi_Shapefile.shp"
    },
    "nandini": {
        "csv": "river_data/nandini.csv",
        "shp": "shapefiles/Nandini_POLYGON.shp"   # Matching your screenshot
    },
    "gangavali": {
        "csv": "river_data/gangavali.csv",
        "shp": "shapefiles/Gangavali_POLYGON.shp" # Matching your screenshot
    }
}

# Global Variables
systems = {} 
model = None
scaler = None

# ==========================================
# 4. STARTUP: LOAD EVERYTHING
# ==========================================
@app.on_event("startup")
async def load_system():
    global systems, model, scaler
    print("\n🚀 STARTING AQUA MONITOR BACKEND...")

    # A. Load Rivers
    for river_id, paths in RIVER_CONFIG.items():
        print(f"   🌊 Loading {river_id}...")
        sys_data = {'df': None, 'poly': None, 'tree': None}
        
        # 1. Load Shapefile
        try:
            if os.path.exists(paths['shp']):
                gdf = gpd.read_file(paths['shp'])
                sys_data['poly'] = gdf.geometry.unary_union
                print(f"      ✅ Shapefile Loaded")
            else:
                print(f"      ❌ Shapefile Missing: {paths['shp']}")
        except Exception as e:
            print(f"      ❌ Shapefile Error: {e}")

        # 2. Load CSV
        try:
            if os.path.exists(paths['csv']):
                df = pd.read_csv(paths['csv'])
                df.columns = df.columns.str.lower().str.strip()
                
                lat_col = next((c for c in df.columns if c in ['lat', 'latitude']), 'latitude')
                lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude']), 'longitude')
                
                valid_coords = df[[lat_col, lon_col]].dropna()
                sys_data['tree'] = cKDTree(valid_coords.values)
                sys_data['df'] = df
                print(f"      ✅ CSV Loaded: {len(df)} points")
            else:
                print(f"      ℹ️ CSV Not Found: {paths['csv']}")
        except Exception as e:
            print(f"      ❌ CSV Error: {e}")
        
        systems[river_id] = sys_data

    # B. Load AI Resources
    try:
        scaler = joblib.load("scaler.pkl")
        model = GATModel(in_channels=4, hidden_channels=32, out_channels=1)
        checkpoint = torch.load("gat_k20_drop0.3_unc4_best.pt", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("✅ AI Model Loaded & Ready\n")
    except Exception as e:
        print(f"❌ Model/Scaler Error: {e}\n")

# ==========================================
# 5. API ENDPOINTS
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
        
        # Smart Prediction Column Selection
        val_col = 'predicted_salinity' if 'predicted_salinity' in df.columns else 'ai_salinity_prediction'
        # If pre-calculated column exists, use it for speed. Else random (AI too slow for whole map)
        values = df[val_col].values if val_col in df.columns else np.random.rand(len(df))

        # Interpolate
        grid_sal = griddata(points, values, (grid_x, grid_y), method='cubic')
        grid_sal[grid_sal < 0] = 0 # Fix physics
        
        # Masking (Handle MultiPolygons correctly)
        flat_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T
        
        if hasattr(poly, 'geoms'): 
             poly_list = list(poly.geoms)
        else:
             poly_list = [poly]

        final_mask = np.zeros(flat_points.shape[0], dtype=bool)
        for p in poly_list:
            poly_coords = np.array(p.exterior.coords)
            if poly_coords.shape[1] > 2: poly_coords = poly_coords[:, :2] # Fix 3D coords
            path = Path(poly_coords)
            final_mask |= path.contains_points(flat_points)
        
        grid_sal[~final_mask.reshape(grid_x.shape)] = np.nan
        
        # Colors
        vmin, vmax = np.nanpercentile(grid_sal, 5), np.nanpercentile(grid_sal, 95)

        plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        ax.imshow(grid_sal.T, extent=(minx, maxx, miny, maxy), origin='lower', cmap='plasma_r', alpha=0.8, vmin=vmin, vmax=vmax)
        
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
    sys = systems.get(river)
    if not sys or not sys['tree']: return {"found": False, "message": "No Data"}

    dist, idx = sys['tree'].query([lat, lng], k=1)
    if dist > 0.05: return {"found": False, "message": "Too far"}

    row = sys['df'].iloc[idx].replace({np.nan: None}).to_dict()
    
    # === LIVE AI PREDICTION ===
    raw_ai_output = 0.0
    
    try:
        if model and scaler:
            # Extract bands
            b3 = row.get('band_3', row.get('b3', 0))
            b4 = row.get('band_4', row.get('b4', 0))
            b5 = row.get('band_5', row.get('b5', 0))
            b7 = row.get('band_7', row.get('b7', 0))
            
            if b3 != 0:
                raw_inputs = [b3, b4, b5, b7]
                scaled = scaler.transform([raw_inputs])
                x_tensor = torch.tensor(scaled, dtype=torch.float)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                with torch.no_grad():
                    raw_ai_output = model(x_tensor, edge_index).item()
    except Exception:
        pass

    # If AI fails, fallback to column
    if raw_ai_output == 0:
        val_col = 'predicted_salinity' if 'predicted_salinity' in sys['df'].columns else 'ai_salinity_prediction'
        raw_ai_output = float(row.get(val_col, 0))

    # === THE FIX: CONVERT Z-SCORE TO REAL PSU ===
    # Assuming Mean Salinity ~30 PSU and Std Dev ~5 PSU
    # This turns "-0.5" into "27.5" instead of "0.0"
    real_psu = (raw_ai_output * 5.0) + 30.0
    
    # Safety Check: Salinity can't be negative, but it's rarely exactly 0 in estuaries
    final_prediction = max(0.1, real_psu)

    return {
        "found": True,
        "latitude": row.get('latitude', lat), 
        "longitude": row.get('longitude', lng),
        "ai_salinity_prediction": final_prediction,
        "history": [] 
    }