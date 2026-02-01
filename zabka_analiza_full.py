"""
Kompleksowa analiza sieci sklepów Żabka w Polsce - Rozszerzona wersja

Autorzy:
- Jakub Rosiak 251620
- Mateusz Kosowski 251558
- Nikodem Nowak 251598

Analiza obejmuje:
- Statystyki sieci (usługi, lokalizacje)
- Rozmieszczenie przestrzenne i dostępność
- Identyfikacja obszarów niedostępnych
- Interaktywne mapy z warstwami analitycznymi
- Eksport danych do CSV
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
from scipy.spatial import cKDTree
from scipy import stats
from branca.element import MacroElement
from jinja2 import Template

warnings.filterwarnings('ignore')

# Konfiguracja wizualizacji
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333',
    'grid.color': '#ddd',
    'text.color': '#222',
    'figure.figsize': (14, 8)
})

os.makedirs("output", exist_ok=True)


# ============================================================================
# FUNKCJE POMOCNICZE
# ============================================================================

def add_bar_labels(ax, fontsize=9, offset=0):
    """Dodaje etykiety wartości do słupków na wykresie."""
    for p in ax.patches:
        w = p.get_width()
        if w > 0:
            ax.text(w + offset, p.get_y() + p.get_height()/2, f'{int(w)}', 
                    ha='left', va='center', fontsize=fontsize)


# ============================================================================
# WCZYTYWANIE DANYCH (RAZ, EFEKTYWNIE)
# ============================================================================

print("=" * 70)
print("WCZYTYWANIE DANYCH")
print("=" * 70)

# Sklepy Żabka - filtracja Polski
df_shops = pd.read_csv("data/zabka_shops.csv")
df_shops = df_shops[
    (df_shops['lat'].between(49, 55)) &
    (df_shops['lng'].between(14, 25))
].copy()
df_shops['services'] = df_shops['services'].fillna('').astype(str)

print(f"✓ Sklepy Żabka: {len(df_shops)}")

# GeoDataFrame - EPSG:4326 (mapy) i EPSG:2180 (obliczenia)
gdf_shops_4326 = gpd.GeoDataFrame(
    df_shops,
    geometry=gpd.points_from_xy(df_shops.lng, df_shops.lat),
    crs="EPSG:4326"
)

# Siatka GUS z populacją
try:
    gdf_population = gpd.read_file("data/GRID_NSP2021_RES/GRID_NSP2021_RES.shp")
    gdf_population.set_crs(epsg=2180, allow_override=True, inplace=True)
    gdf_shops_2180 = gdf_shops_4326.to_crs(epsg=2180)
    print(f"✓ Siatka GUS: {len(gdf_population)} komórek")
except Exception as e:
    print(f"✗ Błąd wczytywania siatki GUS: {e}")
    raise SystemExit(1)

# ============================================================================
# OBLICZENIA PRZESTRZENNE (ZOPTYMALIZOWANE)
# ============================================================================

print("\n" + "=" * 70)
print("OBLICZENIA PRZESTRZENNE")
print("=" * 70)

# 1. Sklepy w komórkach siatki (spatial join raz)
joined = gpd.sjoin(gdf_shops_2180, gdf_population, how="inner", predicate="within")
shop_counts = joined['index_right'].value_counts()
gdf_population['shop_count'] = 0
gdf_population.loc[shop_counts.index, 'shop_count'] = shop_counts
print(f"✓ Spatial join: sklepy w komórkach")

# 2. Dystans do najbliższej Żabki (KDTree - szybki)
shop_coords = np.column_stack([
    gdf_shops_2180.geometry.x,
    gdf_shops_2180.geometry.y
])
tree = cKDTree(shop_coords)

centroids = gdf_population.geometry.centroid
grid_coords = np.column_stack([centroids.x, centroids.y])
distances, _ = tree.query(grid_coords)

gdf_population['distance_m'] = distances
gdf_population['distance_km'] = distances / 1000
print(f"✓ KDTree: dystanse obliczone")

# 3. Kategorie dystansu
bins = [0, 1000, 2000, 5000, np.inf]
labels = ['< 1km', '1-2km', '2-5km', '> 5km']
gdf_population['distance_cat'] = pd.cut(gdf_population['distance_m'], bins=bins, labels=labels)

# 4. Ludzie na sklep (tylko tam gdzie są sklepy)
mask_shops = gdf_population['shop_count'] > 0
gdf_population['people_per_shop'] = np.nan
gdf_population.loc[mask_shops, 'people_per_shop'] = (
    gdf_population.loc[mask_shops, 'RES'] / gdf_population.loc[mask_shops, 'shop_count']
)

# 5. Obszary niedostępne (>= 500 osób i > 1.5 km)
underserved = gdf_population[
    (gdf_population['RES'] >= 500) &
    (gdf_population['distance_m'] > 1500)
].copy()
underserved['priority'] = underserved['RES'] * underserved['distance_km']
underserved = underserved.sort_values('priority', ascending=False)
print(f"✓ Obszary niedostępne: {len(underserved)}")

# ============================================================================
# STATYSTYKI
# ============================================================================

print("\n" + "=" * 70)
print("STATYSTYKI DOSTĘPNOŚCI")
print("=" * 70)

total_pop = gdf_population['RES'].sum()
pop_within_1km = gdf_population[gdf_population['distance_m'] <= 1000]['RES'].sum()
pop_within_2km = gdf_population[gdf_population['distance_m'] <= 2000]['RES'].sum()
pop_within_5km = gdf_population[gdf_population['distance_m'] <= 5000]['RES'].sum()
far_pop = gdf_population[gdf_population['distance_m'] > 5000]['RES'].sum()

print(f"  • W promieniu 1 km: {pop_within_1km/total_pop*100:.1f}%")
print(f"  • W promieniu 2 km: {pop_within_2km/total_pop*100:.1f}%")
print(f"  • W promieniu 5 km: {pop_within_5km/total_pop*100:.1f}%")
print(f"  • Dalej niż 5 km: {far_pop:,.0f} osób ({far_pop/total_pop*100:.1f}%)")

# Dodatkowe statystyki
avg_distance = gdf_population['distance_m'].mean()
median_distance = gdf_population['distance_m'].median()
print(f"  • Średni dystans: {avg_distance:.0f} m")
print(f"  • Mediana dystansu: {median_distance:.0f} m")

# ============================================================================
# WYKRESY STATYSTYCZNE - ROZSZERZONE (10 WYKRESÓW)
# ============================================================================

print("\n" + "=" * 70)
print("GENEROWANIE WYKRESÓW")
print("=" * 70)

# Przygotowanie danych do wykresów
services_exploded = df_shops['services'].str.split(',').explode().str.strip()
services_counts = services_exploded.value_counts()

legend_map = {
    'ZBC': 'Żabka Café', 'ODP': 'Odpiek Pieczywa', 'PAC': 'Paczki',
    'TER': 'Płatność Kartą', 'GSM': 'Doładowania', 'KPO': 'Karty Podarunkowe',
    'RAC': 'Rachunki', 'REJ': 'Rejestracja SIM', 'DEN': 'Usługi Energetyczne',
    'LOT': 'Lotto', 'BIH': 'Cashback', 'DKM': 'Karta Miejska'
}
services_names = [legend_map.get(code, code) for code in services_counts.index]

top_cities = df_shops['city'].value_counts().head(10)
voivodeships = df_shops['voivodeship'].value_counts()
pop_by_dist = gdf_population.groupby('distance_cat')['RES'].sum().reindex(labels, fill_value=0)

# Kolory dla dystansów
color_map = {'< 1km': '#2ecc71', '1-2km': '#f1c40f', '2-5km': '#e67e22', '> 5km': '#e74c3c'}
dist_colors = [color_map[label] for label in labels]

# ==================== PLANSZA 1: 6 WYKRESÓW PODSTAWOWYCH ====================
fig = plt.figure(figsize=(18, 12), facecolor='white')
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Usługi dodatkowe
ax1 = fig.add_subplot(gs[0, 0])
sns.barplot(x=services_counts.values, y=services_names, palette="viridis", ax=ax1)
ax1.set_title("Usługi dodatkowe w sklepach", fontweight='bold', fontsize=13)
ax1.set_xlabel("Liczba placówek")
add_bar_labels(ax1)

# 2. Województwa
ax2 = fig.add_subplot(gs[0, 1])
sns.barplot(x=voivodeships.values, y=voivodeships.index, palette="mako", ax=ax2)
ax2.set_title("Liczba sklepów wg województw", fontweight='bold', fontsize=13)
ax2.set_xlabel("Liczba sklepów")
add_bar_labels(ax2)

# 3. Top miasta
ax3 = fig.add_subplot(gs[1, 0])
sns.barplot(x=top_cities.values, y=top_cities.index, palette="rocket", ax=ax3)
ax3.set_title("Top 10 miast", fontweight='bold', fontsize=13)
ax3.set_xlabel("Liczba sklepów")
add_bar_labels(ax3)

# 4. Populacja wg dystansu
ax4 = fig.add_subplot(gs[1, 1])
pop_by_dist.plot(kind='bar', ax=ax4, color=dist_colors)
ax4.set_title("Populacja wg dystansu do Żabki", fontweight='bold', fontsize=13)
ax4.set_xlabel("Dystans")
ax4.set_ylabel("Liczba mieszkańców")
ax4.set_xticklabels(labels, rotation=0)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))

# 5. Krzywa pokrycia
ax5 = fig.add_subplot(gs[2, 0])
pop_data = gdf_population[gdf_population['RES'] > 0].copy()
pop_data = pop_data.sort_values('distance_km')
pop_data['cum_pop'] = pop_data['RES'].cumsum()
pop_data['cum_share'] = pop_data['cum_pop'] / pop_data['RES'].sum()
ax5.plot(pop_data['distance_km'], pop_data['cum_share'], color='#4c78a8', linewidth=2.5)
for d in [1, 2, 5]:
    ax5.axvline(d, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax5.set_xlim(0, 10)
ax5.set_ylim(0, 1)
ax5.set_title("Krzywa pokrycia populacji", fontweight='bold', fontsize=13)
ax5.set_xlabel("Dystans do Żabki (km)")
ax5.set_ylabel("Udział populacji")
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x*100)}%'))

# 6. Scatter - ludność vs dystans
ax6 = fig.add_subplot(gs[2, 1])
sample = gdf_population[gdf_population['RES'] > 0].sample(
    min(5000, len(gdf_population)), random_state=42
)
ax6.scatter(sample['distance_km'], sample['RES'], alpha=0.3, s=8, color='#72b7b2')
ax6.axvline(1.5, color='#e45756', linestyle='--', linewidth=1.5, label='Próg 1.5 km')
ax6.axhline(500, color='#f58518', linestyle='--', linewidth=1.5, label='Próg 500 osób')
ax6.set_xlim(0, 10)
ax6.set_title("Ludność vs dystans (próbka 5000)", fontweight='bold', fontsize=13)
ax6.set_xlabel("Dystans (km)")
ax6.set_ylabel("Ludność w komórce")
ax6.legend(frameon=True, loc='upper right')

plt.savefig('output/analiza_kompletna.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Zapisano: output/analiza_kompletna.png")
plt.close()

# ==================== PLANSZA 2: 2 WYKRESY UZUPEŁNIAJĄCE ====================
fig2 = plt.figure(figsize=(18, 6), facecolor='white')
gs2 = fig2.add_gridspec(1, 2, wspace=0.2)

# 7. HISTOGRAM DYSTANSÓW
ax7 = fig2.add_subplot(gs2[0, 0])
pop_with_res = gdf_population[gdf_population['RES'] > 0]
ax7.hist(pop_with_res['distance_km'], bins=50, weights=pop_with_res['RES'], 
         color='#4c78a8', edgecolor='white', alpha=0.8)
ax7.axvline(1, color='#2ecc71', linestyle='--', linewidth=2, label='1 km')
ax7.axvline(2, color='#f1c40f', linestyle='--', linewidth=2, label='2 km')
ax7.axvline(5, color='#e74c3c', linestyle='--', linewidth=2, label='5 km')
ax7.set_title("Rozkład dystansów do najbliższej Żabki", fontweight='bold', fontsize=13)
ax7.set_xlabel("Dystans (km)")
ax7.set_ylabel("Liczba mieszkańców")
ax7.set_xlim(0, 15)
ax7.legend(loc='upper right', frameon=True)
ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))

# 8. KORELACJA: GĘSTOŚĆ VS SKLEPY
ax8 = fig2.add_subplot(gs2[0, 1])
# Agregacja per komórka - ile mieszkańców na km² vs ile sklepów w promieniu 1km
grid_with_pop = gdf_population[gdf_population['RES'] > 100].copy()
sample_corr = grid_with_pop.sample(min(3000, len(grid_with_pop)), random_state=42)

ax8.scatter(sample_corr['RES'], sample_corr['shop_count'], alpha=0.4, s=15, color='#72b7b2')

# Linia trendu
mask_valid = (sample_corr['shop_count'] > 0) & (sample_corr['RES'] > 0)
if mask_valid.sum() > 10:
    slope, intercept, r_value, _, _ = stats.linregress(
        sample_corr.loc[mask_valid, 'RES'], 
        sample_corr.loc[mask_valid, 'shop_count']
    )
    x_line = np.linspace(sample_corr['RES'].min(), sample_corr['RES'].max(), 100)
    y_line = slope * x_line + intercept
    ax8.plot(x_line, y_line, color='#e45756', linewidth=2, label=f'R² = {r_value**2:.2f}')
    ax8.legend(loc='upper right', frameon=True)

ax8.set_title("Korelacja: liczba mieszkańców vs sklepy", fontweight='bold', fontsize=13)
ax8.set_xlabel("Liczba mieszkańców w komórce")
ax8.set_ylabel("Liczba sklepów")

plt.tight_layout()
plt.savefig('output/analiza_rozszerzona.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Zapisano: output/analiza_rozszerzona.png")
plt.close()

# ============================================================================
# EKSPORT DO CSV (ZAMIAST EXCEL - BEZ DODATKOWYCH ZALEŻNOŚCI)
# ============================================================================

print("\n" + "=" * 70)
print("EKSPORT DO CSV")
print("=" * 70)

# Folder na eksport
os.makedirs("output/csv", exist_ok=True)

# 1. Podsumowanie
summary_data = {
    'Metryka': [
        'Liczba sklepów Żabka',
        'Liczba komórek siatki GUS',
        'Całkowita populacja',
        'Populacja w promieniu 1 km',
        'Populacja w promieniu 2 km',
        'Populacja w promieniu 5 km',
        'Populacja dalej niż 5 km',
        'Średni dystans do Żabki (m)',
        'Mediana dystansu do Żabki (m)',
        'Liczba obszarów niedostępnych',
        '% populacji w 1 km',
        '% populacji w 2 km',
        '% populacji w 5 km'
    ],
    'Wartość': [
        len(df_shops),
        len(gdf_population),
        int(total_pop),
        int(pop_within_1km),
        int(pop_within_2km),
        int(pop_within_5km),
        int(far_pop),
        round(avg_distance, 0),
        round(median_distance, 0),
        len(underserved),
        round(pop_within_1km/total_pop*100, 1),
        round(pop_within_2km/total_pop*100, 1),
        round(pop_within_5km/total_pop*100, 1)
    ]
}
df_summary = pd.DataFrame(summary_data)
df_summary.to_csv('output/csv/01_podsumowanie.csv', index=False, encoding='utf-8-sig')
print("  ✓ Zapisano: output/csv/01_podsumowanie.csv")

# 2. Ranking miast - usunięto (zbyt duże uproszczenie)

# 2. Top 50 obszarów niedostępnych (dawniej nr 3)
underserved_export = underserved.head(50).copy()
underserved_export = underserved_export[['RES', 'distance_m', 'distance_km', 'priority']].reset_index(drop=True)
underserved_export.columns = ['Ludność', 'Dystans (m)', 'Dystans (km)', 'Priorytet ekspansji']
underserved_export['Ranking'] = range(1, len(underserved_export) + 1)
underserved_export = underserved_export[['Ranking', 'Ludność', 'Dystans (m)', 'Dystans (km)', 'Priorytet ekspansji']]
underserved_export.to_csv('output/csv/02_obszary_niedostepne.csv', index=False, encoding='utf-8-sig')
print("  ✓ Zapisano: output/csv/02_obszary_niedostepne.csv")

# 3. Top miasta (dawniej nr 4)
df_top_cities = df_shops['city'].value_counts().head(30).reset_index()
df_top_cities.columns = ['Miasto', 'Liczba sklepów']
df_top_cities.to_csv('output/csv/03_top_miasta_sklepy.csv', index=False, encoding='utf-8-sig')
print("  ✓ Zapisano: output/csv/03_top_miasta_sklepy.csv")

print("✓ Eksport CSV zakończony: output/csv/")

# ============================================================================
# MAPA INTERAKTYWNA - KOMPLEKSOWA (WSZYSTKIE WARSTWY)
# ============================================================================

print("\n" + "=" * 70)
print("MAPA INTERAKTYWNA")
print("=" * 70)

main_map = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles=None)

# Tła mapy
for name, tiles, show in [
    ("Jasne", "CartoDB positron", True),
    ("OSM", "OpenStreetMap", False),
    ("Ciemne", "CartoDB dark_matter", False)
]:
    folium.TileLayer(tiles=tiles, name=f"Tło: {name}", control=True, show=show).add_to(main_map)

# WARSTWA 1: Sklepy Żabka (klastry) - DOMYŚLNIE ON
shops_layer = folium.FeatureGroup(name="🐸 Sklepy Żabka (wszystkie)", show=True)
cluster = MarkerCluster().add_to(shops_layer)
for row in df_shops.itertuples():
    folium.Marker(
        [row.lat, row.lng],
        tooltip=f"<b>{row.city}</b><br>{row.address}",
        icon=folium.Icon(color="green", icon="frog", prefix="fa")
    ).add_to(cluster)
shops_layer.add_to(main_map)

# WARSTWA 2: Obszary niedostępne - OFF
underserved_layer = folium.FeatureGroup(name="⚠️ Obszary niedostępne (>500 osób, >1.5km)", show=False)
underserved_4326 = underserved.head(250).to_crs(epsg=4326)

for _, row in underserved_4326.iterrows():
    c = row.geometry.centroid
    color = color_map.get(row.distance_cat, '#999')
    radius = min(18, max(6, row.RES / 200))
    
    folium.CircleMarker(
        [c.y, c.x],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        weight=1.5,
        popup=f"<b>Niedostępny obszar</b><br>Ludność: {int(row.RES)}<br>Dystans: {int(row.distance_m)}m ({row.distance_km:.1f} km)<br>Priorytet: {row.priority:.0f}",
        tooltip=f"💡 {int(row.RES)} osób | {int(row.distance_m)}m"
    ).add_to(underserved_layer)

underserved_layer.add_to(main_map)

# WARSTWA 3: Heatmapa gęstości - OFF
heat_data = [[row.lat, row.lng] for row in df_shops.itertuples()]
heatmap_layer = folium.FeatureGroup(name="🔥 Heatmapa gęstości sklepów", show=False)
HeatMap(
    heat_data,
    radius=15,
    blur=25,
    max_zoom=13,
    gradient={0.2: '#4c78a8', 0.5: '#72b7b2', 0.7: '#f58518', 0.9: '#e45756'}
).add_to(heatmap_layer)
heatmap_layer.add_to(main_map)

# WARSTWA 4: Choropleth obciążenie - OFF
gdf_with_shops = gdf_population[gdf_population['shop_count'] > 0].copy()
gdf_with_shops_4326 = gdf_with_shops.to_crs(epsg=4326)

# Bins - proste przedziały do 95 percentyla, potem jeden duży przedział do max
percentile_95 = gdf_with_shops['people_per_shop'].quantile(0.95)
max_val = gdf_with_shops['people_per_shop'].max()

# Rozsądne przedziały + jeden duży na końcu dla outlierów
bins_choro = [0, 1000, 2500, 5000, 7500, percentile_95, max_val + 1]
bins_choro = sorted(list(set([float(b) for b in bins_choro])))

print(f"  Choropleth bins: {[int(b) for b in bins_choro]}")

choropleth = folium.Choropleth(
    geo_data=gdf_with_shops_4326,
    data=gdf_with_shops_4326,
    columns=['CODE', 'people_per_shop'],
    key_on='feature.properties.CODE',
    fill_color='YlOrRd',
    fill_opacity=0.6,
    line_opacity=0.1,
    legend_name='Mieszkańców na sklep',
    name="📊 Obciążenie sklepów (ludzie/sklep)",
    show=False,
    bins=bins_choro,
    nan_fill_color='white',
    nan_fill_opacity=0
)

tooltip_choropleth = folium.GeoJsonTooltip(
    fields=['RES', 'shop_count', 'people_per_shop'],
    aliases=['Ludność:', 'Sklepy:', 'Ludzi/sklep:'],
    localize=True
)
choropleth.geojson.add_child(tooltip_choropleth)
choropleth.add_to(main_map)

# Kontrolka legendy choropleth
class BindLegendToLayer(MacroElement):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self._template = Template("""
        {% macro script(this, kwargs) %}
            var map = {{this.layer._parent.get_name()}};
            var layer = {{this.layer.get_name()}};
            function getLegend() {
                return document.querySelector('.legend');
            }
            function setLegendVisibility() {
                var legend = getLegend();
                if (!legend) return;
                legend.style.display = map.hasLayer(layer) ? 'block' : 'none';
            }
            map.on('overlayadd', function(e) {
                if (e.layer === layer) setLegendVisibility();
            });
            map.on('overlayremove', function(e) {
                if (e.layer === layer) setLegendVisibility();
            });
            setTimeout(setLegendVisibility, 200);
        {% endmacro %}
        """)

main_map.add_child(BindLegendToLayer(choropleth))

# WARSTWA 5: Strefy dostępności (bąbelki) - OFF
zones_layer = folium.FeatureGroup(name="🎯 Strefy dostępności (kolorowe bąbelki)", show=False)
sample_zones = gdf_population[gdf_population['RES'] > 200].sample(
    min(2500, len(gdf_population)), random_state=42
).to_crs(epsg=4326)

for _, row in sample_zones.iterrows():
    c = row.geometry.centroid
    color = color_map.get(row.distance_cat, '#999')
    radius = min(10, max(3, row.RES / 300))
    
    folium.CircleMarker(
        [c.y, c.x],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.6,
        weight=0,
        tooltip=f"{row.distance_cat} | {int(row.RES)} osób"
    ).add_to(zones_layer)

zones_layer.add_to(main_map)

# WARSTWA 6: TOP 10 LOKALIZACJI NA NOWE SKLEPY - OFF
top_locations_layer = folium.FeatureGroup(name="🏆 Top 10 lokalizacji na nowy sklep", show=False)
top_10_locations = underserved.head(10).to_crs(epsg=4326)

for rank, (_, row) in enumerate(top_10_locations.iterrows(), 1):
    c = row.geometry.centroid
    
    folium.Marker(
        [c.y, c.x],
        popup=f"""
        <div style='width:200px'>
            <h4 style='color:#e74c3c; margin:5px 0'>🏆 #{rank} Priorytet ekspansji</h4>
            <b>Populacja:</b> {int(row.RES):,} osób<br>
            <b>Dystans do Żabki:</b> {row.distance_km:.1f} km<br>
            <b>Wskaźnik priorytetu:</b> {row.priority:,.0f}
        </div>
        """,
        tooltip=f"🏆 #{rank} | {int(row.RES):,} osób | {row.distance_km:.1f} km",
        icon=folium.Icon(color="red", icon="star", prefix="fa")
    ).add_to(top_locations_layer)

top_locations_layer.add_to(main_map)

# Stylizacja kontrolki warstw
css = """
<style>
.leaflet-control-layers {
    border-radius: 12px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25) !important;
    padding: 12px !important;
    background: rgba(255,255,255,0.97) !important;
    font-family: 'Segoe UI', sans-serif !important;
}
.leaflet-control-layers-toggle {
    width: 44px !important;
    height: 44px !important;
}
.leaflet-control-layers label {
    margin: 6px 0 !important;
    font-size: 13px !important;
}
</style>
"""
main_map.get_root().html.add_child(folium.Element(css))

folium.LayerControl(collapsed=False).add_to(main_map)
main_map.save('output/mapa_interaktywna.html')
print("✓ Zapisano: output/mapa_interaktywna.html")

# ============================================================================
# RAPORT KOŃCOWY
# ============================================================================

print("\n" + "=" * 70)
print("RAPORT KOŃCOWY")
print("=" * 70)
print(f"\n📊 Sklepy: {len(df_shops)}")
print(f"📍 Komórki siatki: {len(gdf_population)}")
print(f"⚠️  Obszary niedostępne: {len(underserved)}")
print(f"\n✅ Populacja w 1 km: {pop_within_1km/total_pop*100:.1f}%")
print(f"✅ Populacja w 2 km: {pop_within_2km/total_pop*100:.1f}%")
print(f"✅ Populacja w 5 km: {pop_within_5km/total_pop*100:.1f}%")
print(f"\n📁 Wygenerowane pliki:")
print(f"   • output/analiza_kompletna.png (6 wykresów podstawowych)")
print(f"   • output/analiza_rozszerzona.png (2 wykresy dodatkowe)")
print(f"   • output/csv/ (3 pliki CSV z danymi)")
print(f"   • output/mapa_interaktywna.html (6 warstw)")
print("\n" + "=" * 70)
print("✨ ANALIZA ZAKOŃCZONA POMYŚLNIE")
print("=" * 70)
