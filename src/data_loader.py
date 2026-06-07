import pandas as pd
import numpy as np

def load_and_preprocess(filepath, is_test=False):
    """Loads the dataset and extracts cyclical time features."""
    df = pd.read_csv(filepath)
    
    # Parse timestamp
    time_parts = df['timestamp'].str.split(':', expand=True).astype(int)
    df['hour'] = time_parts[0]
    df['minute'] = time_parts[1]
    
    # Cyclical temporal profiles
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
    
    # Create time slot proxy (0 to 95)
    df['time_slot'] = (df['hour'] * 60 + df['minute']) // 15
    
    # Create some composite spatial features
    df['geohash_p4'] = df['geohash'].str[:4]
    
    if 'RoadType' in df.columns and 'NumberofLanes' in df.columns:
        df['RoadType_NumberofLanes'] = df['RoadType'].astype(str) + "_" + df['NumberofLanes'].astype(str)
        
    return df
