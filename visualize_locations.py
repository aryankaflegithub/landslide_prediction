import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import numpy as np

# Load data
gdf = gpd.read_file("inventory/inventory/landslide_nepal_1987-2020.shp")

print("MAP BUILDING...")

# Extract coords
lons = gdf.geometry.centroid.x.values
lats = gdf.geometry.centroid.y.values

# Get bounds
min_lat, min_lon = lats.min(), lons.min()
max_lat, max_lon = lats.max(), lons.max()
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

print(f"\nLOCATION BOUNDS:")
print(f"  North: {max_lat:.2f}")
print(f"  South: {min_lat:.2f}")
print(f"  East: {max_lon:.2f}")
print(f"  West: {min_lon:.2f}")
print(f"  Center: ({center_lat:.2f}, {center_lon:.2f})")

# MAP 1: Interactive heatmap (folium)
print("\nBUILDING INTERACTIVE MAP...")
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=8,
    tiles='OpenStreetMap'
)

# Add all points
for lon, lat in zip(lons, lats):
    folium.CircleMarker(
        location=[lat, lon],
        radius=2,
        color='red',
        opacity=0.3,
        weight=1
    ).add_to(m)

# Add heatmap layer
heat_data = [[lat, lon] for lat, lon in zip(lats, lons)]
HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

# Add legend
title_html = '''
             <div style="position: fixed;
                     bottom: 50px; right: 50px; width: 250px; height: 100px;
                     background-color: white; border:2px solid grey; z-index:9999;
                     font-size:14px; padding: 10px">
             <b>Nepal Landslides 1987-2020</b><br>
             Total: 5,102 events<br>
             Red points = landslide locations<br>
             Heatmap = concentration zones
             </div>
             '''
m.get_root().html.add_child(folium.Element(title_html))

m.save('landslide_map_interactive.html')
print("  DONE. File: landslide_map_interactive.html")

# MAP 2: Static scatter plot
print("\nBUILDING STATIC MAP...")
fig, ax = plt.subplots(figsize=(14, 10))

scatter = ax.scatter(lons, lats, c=gdf['year'], cmap='viridis',
                     s=20, alpha=0.6, edgecolors='black', linewidth=0.3)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Landslide Locations - Nepal (1987-2020)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax, label='Year')

plt.tight_layout()
plt.savefig('landslide_locations_static.png', dpi=150, bbox_inches='tight')
print("  DONE. File: landslide_locations_static.png")

# MAP 3: By region (grid analysis)
print("\nBUILDING REGIONAL CLUSTERS...")
fig, ax = plt.subplots(figsize=(14, 10))

# Create grid and count events
grid_lon = (lons // 0.5) * 0.5
grid_lat = (lats // 0.5) * 0.5

regions = pd.DataFrame({'lon': lons, 'lat': lats, 'grid_lon': grid_lon, 'grid_lat': grid_lat})
region_counts = regions.groupby(['grid_lon', 'grid_lat']).size().reset_index(name='count')

# Plot regions
scatter2 = ax.scatter(region_counts['grid_lon'], region_counts['grid_lat'],
                     c=region_counts['count'], s=region_counts['count']*5,
                     cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=1)

ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title('Landslide Hotspots - Regional Analysis', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

cbar2 = plt.colorbar(scatter2, ax=ax, label='Events per 0.5° grid')

plt.tight_layout()
plt.savefig('landslide_hotspots.png', dpi=150, bbox_inches='tight')
print("  DONE. File: landslide_hotspots.png")

# MAP 4: By year (temporal)
print("\nBUILDING TEMPORAL MAP...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

years_to_show = [1995, 2005, 2015, 2020]
for idx, year in enumerate(years_to_show):
    ax = axes[idx // 2, idx % 2]

    mask = gdf['year'] == year
    year_lons = lons[mask]
    year_lats = lats[mask]

    ax.scatter(year_lons, year_lats, c='red', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'{year} ({len(year_lons)} events)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min_lon - 0.5, max_lon + 0.5)
    ax.set_ylim(min_lat - 0.5, max_lat + 0.5)

plt.suptitle('Landslide Locations by Year', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('landslide_by_year.png', dpi=150, bbox_inches='tight')
print("  DONE. File: landslide_by_year.png")

print("\n" + "="*60)
print("MAPS DONE!")
print("="*60)
print("\nFILES MADE:")
print("  1. landslide_map_interactive.html - CLICK and zoom (BEST)")
print("  2. landslide_locations_static.png - Simple scatter")
print("  3. landslide_hotspots.png - Hotspot zones")
print("  4. landslide_by_year.png - Timeline")
print("\nOPEN: landslide_map_interactive.html in web browser")
print("="*60)
