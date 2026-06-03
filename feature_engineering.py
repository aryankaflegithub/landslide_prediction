import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
gdf = gpd.read_file("inventory/inventory/landslide_nepal_1987-2020.shp")

print("="*70)
print("FEATURE ENGINEERING FOR ML")
print("="*70)

# Convert date format YYYYMMDD to datetime
gdf['occur_date_int'] = gdf['occur_date'].astype(int)
# Handle invalid dates (0, malformed dates)
gdf['date'] = pd.NaT
valid_mask = (gdf['occur_date_int'] > 19870101) & (gdf['occur_date_int'] < 20201231)
gdf.loc[valid_mask, 'date'] = pd.to_datetime(
    gdf.loc[valid_mask, 'occur_date_int'].astype(str),
    format='%Y%m%d',
    errors='coerce'
)

# For invalid dates, use just the year
gdf.loc[~valid_mask & gdf['date'].isna(), 'date'] = pd.to_datetime(
    gdf.loc[~valid_mask & gdf['date'].isna(), 'year'].astype(str) + '0101',
    format='%Y%m%d'
)

gdf['month'] = gdf['date'].dt.month
gdf['day_of_year'] = gdf['date'].dt.dayofyear

# Extract spatial features
gdf['lon'] = gdf.geometry.centroid.x
gdf['lat'] = gdf.geometry.centroid.y

# Feature 1: Spatial clustering (number of events in nearby cells)
print("\n[1/5] Computing spatial features...")
# Simple grid-based approach: count events in 0.1 degree cells
gdf['grid_lon'] = (gdf['lon'] // 0.1 * 0.1).astype(float)
gdf['grid_lat'] = (gdf['lat'] // 0.1 * 0.1).astype(float)

spatial_density = gdf.groupby(['grid_lon', 'grid_lat']).size().reset_index(name='spatial_density')
gdf = gdf.merge(spatial_density, on=['grid_lon', 'grid_lat'], how='left')

# Feature 2: Temporal features
print("[2/5] Computing temporal features...")
gdf['is_monsoon'] = gdf['month'].isin([6, 7, 8, 9]).astype(int)  # June-Sept monsoon
gdf['sin_month'] = np.sin(2 * np.pi * gdf['month'] / 12)  # Cyclic encoding
gdf['cos_month'] = np.cos(2 * np.pi * gdf['month'] / 12)

# Feature 3: Historical event counts (cumulative up to that point)
print("[3/5] Computing cumulative historical features...")
gdf_sorted = gdf.sort_values('date').reset_index(drop=True)

# Cumulative events per year
yearly_cumsum = gdf_sorted.groupby('year').cumcount()
gdf_sorted['events_ytd'] = yearly_cumsum + 1  # Events so far this year

# Events in past 90 days (rolling window)
gdf_sorted['days_since_epoch'] = (gdf_sorted['date'] - gdf_sorted['date'].min()).dt.days
gdf_sorted['events_past_90d'] = gdf_sorted['year'].rolling(window=30, min_periods=1).count()

gdf = gdf_sorted.copy()

# Feature 4: Event size (log scale)
print("[4/5] Computing size features...")
gdf['log_area'] = np.log10(gdf['area_geome'] + 1)  # +1 to avoid log(0)
gdf['size_category'] = pd.cut(gdf['area_geome'],
                               bins=[0, 1000, 10000, 100000, 2000000],
                               labels=['tiny', 'small', 'medium', 'large'])

# Feature 5: Distance to previous event
print("[5/5] Computing distance features...")
gdf['lon_diff'] = gdf['lon'].diff().fillna(0)
gdf['lat_diff'] = gdf['lat'].diff().fillna(0)
gdf['euclidean_dist'] = np.sqrt(gdf['lon_diff']**2 + gdf['lat_diff']**2)
gdf['days_since_prev'] = (gdf['date'] - gdf['date'].shift()).dt.days.fillna(0)

print("\n" + "="*70)
print("FEATURE SUMMARY")
print("="*70)

features_df = gdf[[
    'date', 'year', 'month', 'day_of_year',
    'lon', 'lat', 'grid_lon', 'grid_lat',
    'is_monsoon', 'sin_month', 'cos_month',
    'spatial_density', 'events_ytd', 'events_past_90d',
    'area_geome', 'log_area', 'size_category',
    'euclidean_dist', 'days_since_prev'
]].copy()

print(f"\nTotal features engineered: {len(features_df.columns)}")
print(f"\nFeature list:")
for i, col in enumerate(features_df.columns, 1):
    print(f"  {i:2d}. {col:25s} {features_df[col].dtype}")

print(f"\nData shape: {features_df.shape}")
print(f"\nSample data:")
print(features_df.head(10))

print(f"\nFeature statistics:")
print(features_df.describe())

# Save engineered features
features_df.to_csv('landslide_features.csv', index=False)
gdf.to_file('landslide_with_features.geojson', driver='GeoJSON')

print(f"\n[OK] Features saved to: landslide_features.csv")
print(f"[OK] Spatial data saved to: landslide_with_features.geojson")

# Prepare train/test split
print("\n" + "="*70)
print("TRAIN/TEST SPLIT")
print("="*70)

# Temporal split: train on 1987-2014, test on 2015-2020
train_data = features_df[features_df['year'] < 2015]
test_data = features_df[features_df['year'] >= 2015]

print(f"\nTrain set: {len(train_data)} events ({train_data['year'].min()}-{train_data['year'].max()})")
print(f"Test set:  {len(test_data)} events ({test_data['year'].min()}-{test_data['year'].max()})")

# Numerical features for ML
ml_features = [
    'lon', 'lat', 'month', 'day_of_year',
    'is_monsoon', 'sin_month', 'cos_month',
    'spatial_density', 'events_ytd', 'events_past_90d',
    'log_area', 'days_since_prev', 'euclidean_dist'
]

X_train = train_data[ml_features].fillna(0)
X_test = test_data[ml_features].fillna(0)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nML-ready features: {len(ml_features)}")
print(f"  X_train shape: {X_train_scaled.shape}")
print(f"  X_test shape: {X_test_scaled.shape}")

# Save for ML pipeline
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
train_data.to_csv('train_set.csv', index=False)
test_data.to_csv('test_set.csv', index=False)

print(f"\n[OK] ML arrays saved (X_train.npy, X_test.npy)")
print(f"[OK] Train/test CSV files saved")

print("\n" + "="*70)
print("NEXT: Build ML models using X_train.npy and X_test.npy")
print("="*70)
