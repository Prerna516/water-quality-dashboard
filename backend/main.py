from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from scipy.spatial import cKDTree
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# 1. DEFINE THE AI MODEL ARCHITECTURE
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

# Global Variables
df = None
kdtree = None
model = None
scaler = None

# ==========================================
# 2. STARTUP: LOAD DATA, MODEL, & SCALER
# ==========================================
@app.on_event("startup")
async def load_system():
    global df, kdtree, model, scaler
    print("🚀 Starting Aqua Monitor AI Backend...")

    # A. LOAD RIVER.CSV (Band Data)
    try:
        # Ensure river.csv is in the backend folder!
        df = pd.read_csv("river.csv")
        df.columns = df.columns.str.lower().str.strip()
        
        lat_col = next((c for c in df.columns if c in ['lat', 'latitude']), 'latitude')
        lon_col = next((c for c in df.columns if c in ['lon', 'lng', 'longitude']), 'longitude')
        
        valid_coords = df[[lat_col, lon_col]].dropna()
        kdtree = cKDTree(valid_coords.values)
        print(f"✅ Band Data Loaded: {len(df)} locations.")
    except Exception as e:
        print(f"❌ Error loading river.csv: {e}")

    # B. LOAD SCALER (The 'Currency Converter' for the AI)
    try:
        scaler = joblib.load("scaler.pkl")
        print("✅ Scaler loaded.")
    except:
        print("⚠️ Scaler NOT found. Creating dummy (Predictions will be inaccurate).")
        scaler = StandardScaler()
        scaler.fit(np.zeros((1, 4)))

    # C. LOAD GAT MODEL (The Brain)
    try:
        model = GATModel(in_channels=4, hidden_channels=32, out_channels=1)
        checkpoint = torch.load("gat_k20_drop0.3_unc4_best.pt", map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        print("✅ GAT Model Loaded.")
    except Exception as e:
        print(f"❌ Model Error: {e}")

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.get("/api/bounds")
def get_bounds():
    if df is None: return {}
    return {
        "min_lat": df['latitude'].min(), "max_lat": df['latitude'].max(),
        "min_lng": df['longitude'].min(), "max_lng": df['longitude'].max()
    }

@app.get("/api/get-nearest")
def get_nearest(lat: float, lng: float):
    if kdtree is None: raise HTTPException(status_code=503, detail="Loading")

    # 1. Find Nearest Point
    dist, idx = kdtree.query([lat, lng], k=1)
    if dist > 0.05: return {"found": False, "message": "Too far"}

    row = df.iloc[idx].replace({np.nan: None}).to_dict()
    
    # 2. Calculate Real Prediction for TODAY
    real_prediction = 0.0
    try:
        if model and scaler:
            # We need Band 3, 4, 5, 7
            raw_inputs = [
                row.get('band_3', 0),
                row.get('band_4', 0),
                row.get('band_5', 0),
                row.get('band_7', 0)
            ]
            # Scale inputs
            scaled = scaler.transform([raw_inputs])
            
            # Run Model
            x_tensor = torch.tensor(scaled, dtype=torch.float)
            edge_index = torch.zeros((2, 0), dtype=torch.long) # Dummy edges
            
            with torch.no_grad():
                # Get the number out of the tensor
                real_prediction = model(x_tensor, edge_index).item()
    except Exception as e:
        print(f"AI Prediction Failed: {e}")

    # 3. Generate DEMO HISTORY (Simulated Trend for Chart)
    # We use the real prediction as the baseline and add small noise
    history_data = []
    today = datetime.now()
    
    for i in range(4, -1, -1): # 5 Days
        date_label = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        
        if i == 0:
            val = real_prediction # Today is real
        else:
            noise = random.uniform(-1.0, 1.0) # Small variation
            val = real_prediction + noise
            
        history_data.append({
            "date": date_label,
            "ai_salinity_prediction": val
        })

    return {
        "found": True,
        "latitude": row['latitude'],
        "longitude": row['longitude'],
        "history": history_data, # For the Chart
        "ai_salinity_prediction": real_prediction # For the Big Number
    }