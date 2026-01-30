"""
Zaawansowane analizy i wizualizacje - Sklepy Żabka w Polsce

Skupiamy się na prostych, zrozumiałych metrykach:
- dystans do najbliższej Żabki (w metrach i kategoriach)
- obszary niedostępne (>= 500 mieszkańców i > 1.5 km do sklepu)
- heatmapa gęstości sklepów
- krzywa pokrycia populacji dystansem
"""

import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')

sns.set_theme(
    style="whitegrid",
    palette="viridis",
    rc={
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "grid.color": "#dddddd",
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
    },
)

os.makedirs("output", exist_ok=True)

# ============================================================================
# WCZYTYWANIE DANYCH
# ============================================================================

print("=== WCZYTYWANIE DANYCH ===")

df_zabka_shop = pd.read_csv("data/zabka_shops.csv", sep=",")
df_clean_zabka_shop = df_zabka_shop[
    (df_zabka_shop['lat'] > 49) & (df_zabka_shop['lat'] < 55) &
    (df_zabka_shop['lng'] > 14) & (df_zabka_shop['lng'] < 25)
].copy()
df_clean_zabka_shop['services'] = df_clean_zabka_shop['services'].fillna('').astype(str)

gdf_zabka_shops_4326 = gpd.GeoDataFrame(
    df_clean_zabka_shop,
    geometry=gpd.points_from_xy(df_clean_zabka_shop.lng, df_clean_zabka_shop.lat),
    crs="EPSG:4326"
)

try:
    gdf_population = gpd.read_file("data/GRID_NSP2021_RES/GRID_NSP2021_RES.shp")
    gdf_population.set_crs(epsg=2180, allow_override=True, inplace=True)
    gdf_zabka_shops_2180 = gdf_zabka_shops_4326.to_crs(epsg=2180)
    print(f"Dane wczytane: {len(df_clean_zabka_shop)} sklepów, siatka GUS OK")
except Exception as e:
    print(f"Błąd: {e}")
    raise SystemExit(1)

# ============================================================================
# PODSTAWOWE PRZYGOTOWANIE: LICZBA SKLEPÓW W KOMÓRCE + DYSTANS
# ============================================================================

print("\n=== OBLICZENIA: SKLEPY W SIATCE + DYSTANS DO NAJBLIŻSZEJ ŻABKI ===")

joined_data = gpd.sjoin(gdf_zabka_shops_2180, gdf_population, how="inner", predicate="within")
shops_in_grid = joined_data['index_right'].value_counts()
gdf_population['shop_count'] = 0
gdf_population.loc[shops_in_grid.index, 'shop_count'] = shops_in_grid

shop_coords = np.column_stack([
    gdf_zabka_shops_2180.geometry.x,
    gdf_zabka_shops_2180.geometry.y
])
tree = cKDTree(shop_coords)

grid_centroids = gdf_population.geometry.centroid
grid_coords = np.column_stack([grid_centroids.x, grid_centroids.y])
distances, _ = tree.query(grid_coords)

gdf_population['distance_to_nearest'] = distances
gdf_population['distance_km'] = gdf_population['distance_to_nearest'] / 1000.0

distance_bins = [0, 1000, 2000, 5000, np.inf]
distance_labels = ['< 1km', '1-2km', '2-5km', '> 5km']
gdf_population['distance_category'] = pd.cut(
    gdf_population['distance_to_nearest'],
    bins=distance_bins,
    labels=distance_labels
)

# ============================================================================
# OBSZARY NIEDOSTĘPNE (PROSTE, ZROZUMIAŁE KRYTERIA)
# ============================================================================

print("\n=== OBSZARY NIEDOSTĘPNE ===")

underserved_areas = gdf_population[
    (gdf_population['RES'] >= 500) &
    (gdf_population['distance_to_nearest'] >= 1500)
].copy()

underserved_areas['priority_index'] = underserved_areas['RES'] * underserved_areas['distance_km']
underserved_areas = underserved_areas.sort_values('priority_index', ascending=False)

print(f"Obszary niedostępne: {len(underserved_areas)}")
print("Definicja: >= 500 mieszkańców i > 1.5 km do najbliższej Żabki")

# ============================================================================
# PODSTAWOWE STATYSTYKI DOSTĘPNOŚCI
# ============================================================================

total_pop = gdf_population['RES'].sum()
pop_within_1km = gdf_population[gdf_population['distance_to_nearest'] <= 1000]['RES'].sum()
pop_within_2km = gdf_population[gdf_population['distance_to_nearest'] <= 2000]['RES'].sum()
pop_within_5km = gdf_population[gdf_population['distance_to_nearest'] <= 5000]['RES'].sum()

print("\n=== POKRYCIE POPULACJI ===")
print(f"Populacja w promieniu 1 km: {pop_within_1km/total_pop*100:.2f}%")
print(f"Populacja w promieniu 2 km: {pop_within_2km/total_pop*100:.2f}%")
print(f"Populacja w promieniu 5 km: {pop_within_5km/total_pop*100:.2f}%")

# ============================================================================
# MAPA ZAAWANSOWANA
# ============================================================================

print("\n=== GENEROWANIE MAPY ZAAWANSOWANEJ ===")

advanced_map = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles=None)

folium.TileLayer(
    tiles="CartoDB positron",
    name="Tło mapy (Jasne)",
    control=True,
    show=True
).add_to(advanced_map)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="Tło mapy (OSM)",
    control=True,
    show=False
).add_to(advanced_map)

folium.TileLayer(
    tiles="CartoDB dark_matter",
    name="Tło mapy (Ciemne)",
    control=True,
    show=False
).add_to(advanced_map)

# Warstwa 1: Heatmapa gęstości sklepów
heat_data = [[row.lat, row.lng] for row in df_clean_zabka_shop.itertuples()]
heatmap_layer = folium.FeatureGroup(
    name="Gęstość sklepów (heatmapa)",
    show=False
)
HeatMap(
    heat_data,
    radius=14,
    blur=22,
    max_zoom=13,
    gradient={0.2: '#4c78a8', 0.4: '#72b7b2', 0.6: '#f58518', 0.8: '#e45756', 1: '#b279a2'}
).add_to(heatmap_layer)
heatmap_layer.add_to(advanced_map)

# Warstwa 2: Obszary niedostępne
color_map = {
    '< 1km': '#2ecc71',
    '1-2km': '#f1c40f',
    '2-5km': '#e67e22',
    '> 5km': '#e74c3c'
}

underserved_layer = folium.FeatureGroup(
    name="Obszary niedostępne (>= 500 osób, > 1.5 km)",
    show=True
)

underserved_4326 = underserved_areas.head(250).to_crs(epsg=4326)

for _, row in underserved_4326.iterrows():
    centroid = row.geometry.centroid
    radius = min(12, max(4, row.RES / 300))
    color = color_map.get(row.distance_category, '#999999')

    folium.CircleMarker(
        location=[centroid.y, centroid.x],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=1,
        popup=folium.Popup(
            f"""
            <b>Obszar niedostępny</b><br>
            Ludność: {int(row['RES'])} osób<br>
            Dystans do najbliższej Żabki: {int(row['distance_to_nearest'])} m<br>
            Priorytet (ludność * dystans_km): {row['priority_index']:.1f}
            """,
            max_width=260
        ),
        tooltip=f"Ludność: {int(row['RES'])} | Dystans: {int(row['distance_to_nearest'])} m"
    ).add_to(underserved_layer)

underserved_layer.add_to(advanced_map)

# Warstwa 3: Strefy dostępności (próbka dla czytelności)
accessibility_layer = folium.FeatureGroup(
    name="Strefy dostępności (kategorie dystansu)",
    show=False
)

sample_source = gdf_population[gdf_population['RES'] > 200]
sample_grid = sample_source.sample(
    min(2500, len(sample_source)),
    random_state=42
)
sample_grid_4326 = sample_grid.to_crs(epsg=4326)

for _, row in sample_grid_4326.iterrows():
    centroid = row.geometry.centroid
    color = color_map.get(row.distance_category, '#999999')

    folium.CircleMarker(
        location=[centroid.y, centroid.x],
        radius=3,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=0,
        tooltip=f"Dystans: {row.distance_category} | Ludność: {int(row.RES)}"
    ).add_to(accessibility_layer)

accessibility_layer.add_to(advanced_map)

# Legenda dla stref dostępności
legend_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background-color: rgba(255, 255, 255, 0.95);
    padding: 10px 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
    font-size: 12px;
    color: #222;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
">
  <div style="font-weight: 600; margin-bottom: 6px;">Dystans do najbliższej Żabki</div>
  <div><span style="background:#2ecc71;width:10px;height:10px;display:inline-block;margin-right:6px;"></span>&lt; 1 km</div>
  <div><span style="background:#f1c40f;width:10px;height:10px;display:inline-block;margin-right:6px;"></span>1-2 km</div>
  <div><span style="background:#e67e22;width:10px;height:10px;display:inline-block;margin-right:6px;"></span>2-5 km</div>
  <div><span style="background:#e74c3c;width:10px;height:10px;display:inline-block;margin-right:6px;"></span>&gt; 5 km</div>
</div>
"""

advanced_map.get_root().html.add_child(folium.Element(legend_html))

custom_css = """
<style>
    .leaflet-control-layers {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
        padding: 10px !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid #ddd !important;
    }
    .leaflet-control-layers label {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        margin-bottom: 5px !important;
        cursor: pointer !important;
    }
</style>
"""

advanced_map.get_root().html.add_child(folium.Element(custom_css))
folium.LayerControl(collapsed=False).add_to(advanced_map)

output_file = 'output/mapa_zaawansowana_analiza.html'
advanced_map.save(output_file)
print(f"Mapa zapisana: {output_file}")

# ============================================================================
# WYKRESY ANALITYCZNE (JASNY MOTYW)
# ============================================================================

print("\n=== GENEROWANIE WYKRESÓW ANALITYCZNYCH ===")

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="white")
for ax in axes.ravel():
    ax.set_facecolor("white")

# Wykres 1: Populacja wg dystansu
ax1 = axes[0, 0]
pop_by_category = gdf_population.groupby('distance_category')['RES'].sum().reindex(distance_labels)
pop_by_category = pop_by_category.fillna(0)
bar_colors = [color_map[label] for label in distance_labels]
pop_by_category.plot(kind='bar', ax=ax1, color=bar_colors)
ax1.set_title('Populacja według dystansu do najbliższej Żabki', fontsize=14, fontweight='bold')
ax1.set_xlabel('Dystans', fontsize=12)
ax1.set_ylabel('Liczba mieszkańców', fontsize=12)
ax1.tick_params(axis='x', rotation=0)

# Wykres 2: Krzywa pokrycia populacji
ax2 = axes[0, 1]
pop_data = gdf_population[gdf_population['RES'] > 0].copy()
pop_data = pop_data.sort_values('distance_km')
pop_data['cum_pop'] = pop_data['RES'].cumsum()
pop_data['cum_share'] = pop_data['cum_pop'] / pop_data['RES'].sum()
ax2.plot(pop_data['distance_km'], pop_data['cum_share'], color='#4c78a8', linewidth=2)
ax2.axvline(1, color='#999999', linestyle='--', linewidth=1)
ax2.axvline(2, color='#999999', linestyle='--', linewidth=1)
ax2.axvline(5, color='#999999', linestyle='--', linewidth=1)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)
ax2.set_title('Krzywa pokrycia populacji dystansem', fontsize=14, fontweight='bold')
ax2.set_xlabel('Dystans do najbliższej Żabki (km)', fontsize=12)
ax2.set_ylabel('Udział populacji (0-1)', fontsize=12)

# Wykres 3: Ludność vs dystans do najbliższej Żabki
ax3 = axes[1, 0]
scatter_source = gdf_population[gdf_population['RES'] > 0]
sample_scatter = scatter_source.sample(
    min(8000, len(scatter_source)),
    random_state=42
)
ax3.scatter(
    sample_scatter['distance_km'],
    sample_scatter['RES'],
    alpha=0.3,
    s=10,
    color='#72b7b2'
)
ax3.axvline(1.5, color='#e45756', linestyle='--', linewidth=1, label='Próg 1.5 km')
ax3.axhline(500, color='#f58518', linestyle='--', linewidth=1, label='Próg 500 osób')
ax3.set_xlim(0, 10)
ax3.set_title('Ludność vs dystans do najbliższej Żabki', fontsize=14, fontweight='bold')
ax3.set_xlabel('Dystans (km)', fontsize=12)
ax3.set_ylabel('Liczba mieszkańców w komórce', fontsize=12)
ax3.legend(frameon=False)

# Wykres 4: Top 15 obszarów niedostępnych
ax4 = axes[1, 1]
top_15 = underserved_areas.head(15)
ax4.barh(top_15['CODE'].astype(str), top_15['priority_index'], color='#b279a2')
ax4.invert_yaxis()
ax4.set_title('Top 15 obszarów niedostępnych (priorytet)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Priorytet = ludność * dystans_km', fontsize=12)
ax4.set_ylabel('Kod komórki (GUS)', fontsize=12)

plt.tight_layout()
plt.savefig('output/analiza_zaawansowana.png', dpi=300, bbox_inches='tight', facecolor="white")
print("Wykresy zapisane: output/analiza_zaawansowana.png")
plt.show()

# ============================================================================
# RAPORT KOŃCOWY
# ============================================================================

print("\n" + "=" * 70)
print("RAPORT KOŃCOWY - ZAAWANSOWANA ANALIZA")
print("=" * 70)

print("\nStatystyki:")
print(f"  • Liczba sklepów: {len(df_clean_zabka_shop)}")
print(f"  • Liczba komórek siatki: {len(gdf_population)}")
print(f"  • Obszary niedostępne: {len(underserved_areas)}")

print("\nPokrycie populacji:")
print(f"  • W promieniu 1 km: {pop_within_1km/total_pop*100:.2f}%")
print(f"  • W promieniu 2 km: {pop_within_2km/total_pop*100:.2f}%")
print(f"  • W promieniu 5 km: {pop_within_5km/total_pop*100:.2f}%")

far_population = gdf_population[gdf_population['distance_to_nearest'] > 5000]['RES'].sum()
print(f"\nPopulacja > 5 km od sklepu: {far_population:,.0f}")

print("\n" + "=" * 70)
print("Analiza zakończona pomyślnie!")
print("=" * 70)
