import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
gdf = gpd.read_file("inventory/inventory/landslide_nepal_1987-2020.shp")

print("="*70)
print("LANDSLIDE EARLY DETECTION SYSTEM - DATA EXPLORATION")
print("="*70)

# Investigate occur_date format
print(f"\n1. DATE FORMAT ANALYSIS")
print(f"   Sample occur_date values: {gdf['occur_date'].head(10).values}")
print(f"   Data type: {gdf['occur_date'].dtype}")
print(f"   Unique dates: {gdf['occur_date'].nunique()}")

print(f"\n2. DATASET OVERVIEW")
print(f"   Total records: {len(gdf)}")
print(f"   Year range: {gdf['year'].min()}-{gdf['year'].max()}")

print(f"\n3. TEMPORAL DISTRIBUTION (Top 15 years)")
top_years = gdf['year'].value_counts().sort_values(ascending=False).head(15)
for year, count in top_years.items():
    print(f"   {year}: {count} landslides")

print(f"\n4. GEOGRAPHIC COVERAGE")
bounds = gdf.total_bounds
print(f"   Longitude: {bounds[0]:.2f} to {bounds[2]:.2f}")
print(f"   Latitude: {bounds[1]:.2f} to {bounds[3]:.2f}")
print(f"   Valid geometries: {gdf.geometry.is_valid.sum()}/{len(gdf)}")

print(f"\n5. LANDSLIDE SCALE")
print(f"   Area (m^2) - Min: {gdf['area_geome'].min():.0f}")
print(f"   Area (m^2) - Mean: {gdf['area_geome'].mean():.0f}")
print(f"   Area (m^2) - Median: {gdf['area_geome'].median():.0f}")
print(f"   Area (m^2) - Max: {gdf['area_geome'].max():.0f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temporal trend
yearly = gdf['year'].value_counts().sort_index()
axes[0, 0].plot(yearly.index, yearly.values, marker='o', linewidth=2)
axes[0, 0].set_title('Landslides per Year', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Year')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(True, alpha=0.3)

# Size distribution
axes[0, 1].hist(np.log10(gdf['area_geome']), bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Landslide Area Distribution (log scale)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('log10(Area in m^2)')
axes[0, 1].set_ylabel('Frequency')

# Spatial distribution
gdf.plot(ax=axes[1, 0], alpha=0.3, edgecolor='k', linewidth=0.1)
axes[1, 0].set_title('Spatial Distribution of Landslides', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Longitude')
axes[1, 0].set_ylabel('Latitude')

# Size vs Year
axes[1, 1].scatter(gdf['year'], np.log10(gdf['area_geome']), alpha=0.4, s=20)
axes[1, 1].set_title('Landslide Size Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('log10(Area in m^2)')

plt.tight_layout()
plt.savefig('landslide_exploration.png', dpi=100, bbox_inches='tight')
print(f"\nVisualization saved: landslide_exploration.png")

# Export processed data
gdf.to_file('landslide_processed.geojson', driver='GeoJSON')
print(f"Processed data saved: landslide_processed.geojson")

print("\n" + "="*70)
print("DATA READY FOR EARLY DETECTION SYSTEM DEVELOPMENT")
print("="*70)
