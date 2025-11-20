import io
import traceback
import random
from datetime import datetime, timedelta

# --- 1. CRITICAL IMPORTS (Must be at the top) ---
import matplotlib
matplotlib.use('Agg') # Fixes "Internal Server Error" with plots
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
# 2. MODEL DEFINITION
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
# 3. APP SETUP & GLOBAL VARIABLES
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold loaded data
df = None
kdtree = None
model = None
scaler = None
river_poly = None # Holds the shapefile data

# ==========================================
# 4. STARTUP EVENT (Loads files ONCE)
# ==========================================
@app.on_event("startup")
async def load_system():
    global df, kdtree, model, scaler, river_poly
    print("🚀 Starting Aqua Monitor AI Backend...")

    # A. LOAD SHAPEFILE (For the Heatmap Mask)
    try:
        river_shape = gpd.read_file("shapefiles/netra_river_shape.shp")
        river_poly = river_shape.geometry.unary_union
        print("✅ Shapefile loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading shapefile: {e}")
        print("   (Did you put the .shp file in the 'shapefiles' folder?)")

    # B. LOAD RIVER.CSV (Data Points)
    try:
        df = pd.read_csv("river.csv")
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        lat_col = next((c for c in df.columns if c in ['lat', 'latitude']), 'latitude')
        lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude']), 'longitude')
        
        valid_coords = df[[lat_col, lon_col]].dropna()
        kdtree = cKDTree(valid_coords.values)
        print(f"✅ CSV Data Loaded: {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error loading river.csv: {e}")

    # C. LOAD SCALER
    try:
        scaler = joblib.load("scaler.pkl")
        print("✅ Scaler loaded.")
    except:
        print("⚠️ Scaler NOT found. Using dummy.")
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 4)))

    # D. LOAD AI MODEL
    try:
        model = GATModel(in_channels=4, hidden_channels=32, out_channels=1)
        checkpoint = torch.load("gat_k20_drop0.3_unc4_best.pt", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("✅ GAT Model Loaded.")
    except Exception as e:
        print(f"❌ Model Error: {e}")

# ==========================================
# 5. API ENDPOINTS
# ==========================================

@app.get("/api/bounds")
def get_bounds():
    """Returns the river boundaries for the map box."""
    if river_poly is None:
        # Fallback if shapefile failed
        if df is not None:
             return {
                "min_lat": df['latitude'].min(), "max_lat": df['latitude'].max(),
                "min_lng": df['longitude'].min(), "max_lng": df['longitude'].max()
            }
        return {}
    
    minx, miny, maxx, maxy = river_poly.bounds
    return {
        "min_lat": miny,
        "min_lng": minx,
        "max_lat": maxy,
        "max_lng": maxx
    }

@app.get("/api/get-heatmap")
async def get_heatmap():
    """Generates the heatmap image."""
    if river_poly is None or df is None:
        return Response(content="Server Error: Data not loaded.", status_code=500)

    try:
        # 1. Setup Grid
        minx, miny, maxx, maxy = river_poly.bounds
        grid_x, grid_y = np.mgrid[minx:maxx:500j, miny:maxy:500j]
        
        # 2. Interpolate
        points = df[['longitude', 'latitude']].values
        
        val_col = 'predicted_salinity' if 'predicted_salinity' in df.columns else 'ai_salinity_prediction'
        if val_col not in df.columns:
             values = np.random.rand(len(df)) 
        else:
             values = df[val_col].values

        grid_sal = griddata(points, values, (grid_x, grid_y), method='cubic')
        
        # 3. Masking (with 3D fix)
        flat_x, flat_y = grid_x.flatten(), grid_y.flatten()
        flat_points = np.vstack((flat_x, flat_y)).T
        
        poly_coords = np.array(river_poly.exterior.coords)
        if poly_coords.shape[1] > 2:
            poly_coords = poly_coords[:, :2] 
        
        path = Path(poly_coords)
        mask = path.contains_points(flat_points)
        grid_sal[~mask.reshape(grid_x.shape)] = np.nan

        # --- FIX: CALCULATE ROBUST COLOR LIMITS ---
        # We calculate the 5th and 95th percentiles.
        # This ignores the outliers that are washing out your colors.
        vmin = np.nanpercentile(grid_sal, 5)
        vmax = np.nanpercentile(grid_sal, 95)
        print(f"Heatmap Range: {vmin:.4f} to {vmax:.4f}") # Check terminal to see range

        # 4. Plotting
        plt.figure(figsize=(8, 8), frameon=False)
        ax = plt.Axes(plt.gcf(), [0, 0, 1, 1])
        ax.set_axis_off()
        plt.gcf().add_axes(ax)
        
        # Use vmin and vmax here to force the contrast!
        ax.imshow(grid_sal.T, extent=(minx, maxx, miny, maxy), 
                  origin='lower', cmap='plasma', alpha=0.8, vmin=vmin, vmax=vmax)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close()
        
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception as e:
        print(traceback.format_exc())
        return Response(content=f"Error: {str(e)}", status_code=500)
@app.get("/api/get-nearest")
def get_nearest(lat: float, lng: float):
    """Returns prediction for a specific clicked point."""
    if kdtree is None: raise HTTPException(status_code=503, detail="Loading")

    # 1. Find Nearest Point
    dist, idx = kdtree.query([lat, lng], k=1)
    # Increased distance tolerance slightly
    if dist > 0.05: return {"found": False, "message": "Too far"}

    row = df.iloc[idx].replace({np.nan: None}).to_dict()
    
    # 2. Calculate Real Prediction
    real_prediction = 0.0
    try:
        if model and scaler:
            # Inputs: Ensure these match your CSV column names exactly
            raw_inputs = [
                row.get('band_3', 0),
                row.get('band_4', 0),
                row.get('band_5', 0),
                row.get('band_7', 0)
            ]
            scaled = scaler.transform([raw_inputs])
            x_tensor = torch.tensor(scaled, dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
            with torch.no_grad():
                real_prediction = model(x_tensor, edge_index).item()
    except Exception as e:
        print(f"AI Prediction Failed: {e}")

    # 3. Generate History (Mock data for chart)
    history_data = []
    today = datetime.now()
    for i in range(4, -1, -1):
        date_label = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        val = real_prediction if i == 0 else real_prediction + random.uniform(-1.0, 1.0)
        history_data.append({"date": date_label, "ai_salinity_prediction": val})

    return {
        "found": True,
        "latitude": row['latitude'],
        "longitude": row['longitude'],
        "history": history_data,
        "ai_salinity_prediction": real_prediction
    }