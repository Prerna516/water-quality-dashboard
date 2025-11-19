import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load your data (This acts like 'modis_df' in your snippet)
try:
    # We use river.csv because it contains the Band columns
    df = pd.read_csv("river.csv")
    print("✅ Loaded river.csv")

    # 2. Select the EXACT same columns as your training code
    # Your code used: [['Band_3', 'Band_4', 'Band_5', 'Band_7']]
    feature_cols = ['Band_3', 'Band_4', 'Band_5', 'Band_7']
    
    # Handle lowercase/uppercase differences just in case
    df.columns = df.columns.str.strip() 
    # Ensure we have the columns
    X = df[feature_cols].values

    print("⏳ Fitting Scaler (calculating Mean and Std Dev)...")
    
    # 3. Create and Fit the Scaler (Same as your code)
    scaler = StandardScaler()
    scaler.fit(X)

    # 4. SAVE IT (This is the missing step!)
    joblib.dump(scaler, "scaler.pkl")
    
    print("🎉 Success! 'scaler.pkl' created.")
    print(f"   Mean: {scaler.mean_}")
    print("   You can now run the backend.")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Ensure 'river.csv' is in this folder and has columns: Band_3, Band_4, Band_5, Band_7")