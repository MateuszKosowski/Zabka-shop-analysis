"""
Analiza dostępności i usług sklepów Żabka w Polsce

Autorzy:
- Jakub Rosiak 251620,
- Mateusz Kosowski 251558,
- Nikodem Nowak 251598

Nasz projekt koncentruje się na analizie rozmieszczenia sieci sklepów Żabka w Polsce,
wykorzystując zbiór danych o lokalizacji prawie 10 tysięcy placówek (stan na rok 2024)
oraz dane demograficzne z Narodowego Spisu Powszechnego 2021 (siatka kilometrowa GUS).
"""

# ============================================================================
# IMPORTY I KONFIGURACJA
# ============================================================================

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import warnings
from branca.element import MacroElement
from jinja2 import Template

# Konfiguracja
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
warnings.filterwarnings('ignore')

# ============================================================================
# WCZYTYWANIE DANYCH
# ============================================================================

print("=== WCZYTYWANIE DANYCH ===")

# Dane sklepów Żabka
df_zabka_shop = pd.read_csv("data/zabka_shops.csv", sep=",")
df_clean_zabka_shop = df_zabka_shop[
    (df_zabka_shop['lat'] > 49) & (df_zabka_shop['lat'] < 55) &
    (df_zabka_shop['lng'] > 14) & (df_zabka_shop['lng'] < 25)
].copy()
df_clean_zabka_shop['services'] = df_clean_zabka_shop['services'].fillna('').astype(str)
print(f"Liczba sklepów Żabka w Polsce (2024r.): {len(df_clean_zabka_shop)}")

# Siatka GUS z danymi o populacji
try:
    gdf_population = gpd.read_file("data/GRID_NSP2021_RES/GRID_NSP2021_RES.shp")
    gdf_population.set_crs(epsg=2180, allow_override=True, inplace=True)
    print(f"Siatka wczytana. Układ: {gdf_population.crs}")
except Exception as e:
    print(f"Błąd wczytywania siatki: {e}")
    gdf_population = None

# Przygotowanie GeoDataFrame w układzie 4326 dla Folium
gdf_zabka_shops_4326 = gpd.GeoDataFrame(
    df_clean_zabka_shop,
    geometry=gpd.points_from_xy(df_clean_zabka_shop.lng, df_clean_zabka_shop.lat),
    crs="EPSG:4326"
)

# Przygotowanie GeoDataFrame w układzie 2180 do łączenia z siatką GUS
if gdf_population is not None:
    gdf_zabka_shops_2180 = gdf_zabka_shops_4326.to_crs(epsg=2180)

# ============================================================================
# PREZENTACJA DANYCH STATYSTYCZNYCH SIECI
# ============================================================================

print("\n=== GENEROWANIE WYKRESÓW STATYSTYCZNYCH ===")

# Przygotowanie danych do wykresów
services_exploded = df_clean_zabka_shop['services'].str.split(',').explode().str.strip()
services_counts = services_exploded.value_counts()

legend_map = {
    'ZBC': 'Żabka Café (Kawa/HotDog)', 'ODP': 'Odpiek Pieczywa',
    'PAC': 'Paczki (Odbiór/Nadanie)', 'TER': 'Płatność Kartą',
    'GSM': 'Doładowania Telefonu', 'KPO': 'Karty Podarunkowe',
    'RAC': 'Opłacanie Rachunków', 'REJ': 'Rejestracja SIM',
    'DEN': 'Usługi Energetyczne', 'LOT': 'Lotto',
    'BIH': 'Cashback (Wypłata)', 'DKM': 'Karta Miejska'
}
services_names = [legend_map.get(code, code) for code in services_counts.index]

# Funkcja pomocnicza do dodawania etykiet na wykresy
def add_label(ax):
    for p in ax.patches:
        width = p.get_width()
        if width > 0:
            ax.annotate(f'{int(width)}',
                        (width, p.get_y() + p.get_height() / 2),
                        ha='left', va='center',
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=10, color='black')

# Przygotowanie danych do wykresów
top_cities = df_clean_zabka_shop['city'].value_counts().head(10)
voivodeships = df_clean_zabka_shop['voivodeship'].value_counts()

# Generowanie wykresów
fig, axes = plt.subplots(3, 1, figsize=(12, 18))

# Wykres 1: Struktura usług dodatkowych
sns.barplot(x=services_counts.values, y=services_names, palette="viridis", ax=axes[0])
axes[0].set_title("Struktura usług dodatkowych", fontsize=16, fontweight='bold', color='black')
axes[0].set_xlabel("Liczba placówek", fontsize=12, color='black')
axes[0].set_ylabel("Rodzaj usługi", fontsize=12, color='black')
add_label(axes[0])

# Wykres 2: Liczba sklepów wg województw
sns.barplot(x=voivodeships.values, y=voivodeships.index, palette="mako", ax=axes[1])
axes[1].set_title("Liczba sklepów wg województw", fontsize=16, fontweight='bold', color='black')
axes[1].set_xlabel("Liczba sklepów", fontsize=12, color='black')
axes[1].set_ylabel("Województwo", fontsize=12, color='black')
add_label(axes[1])

# Wykres 3: Top 10 miast
sns.barplot(x=top_cities.values, y=top_cities.index, palette="rocket", ax=axes[2])
axes[2].set_title("Top 10 miast z największą liczbą Żabek", fontsize=16, fontweight='bold', color='black')
axes[2].set_xlabel("Liczba sklepów", fontsize=12, color='black')
axes[2].set_ylabel("Miasto", fontsize=12, color='black')
add_label(axes[2])

plt.tight_layout()
plt.savefig('output/statystyki_zabka.png', dpi=300, bbox_inches='tight')
print("Wykresy zapisane do: output/statystyki_zabka.png")
plt.show()

# ============================================================================
# ANALIZA PRZESTRZENNA I DEMOGRAFICZNA
# ============================================================================

print("\n=== TWORZENIE MAPY INTERAKTYWNEJ ===")

# Mapa Polski
map_of_poland = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles=None)

# Dodanie warstw tła (spójnie z mapą zaawansowaną)
folium.TileLayer(
    tiles="CartoDB positron",
    name="Tło mapy (Jasne)",
    control=True,
    show=True
).add_to(map_of_poland)

folium.TileLayer(
    tiles="OpenStreetMap",
    name="Tło mapy (OSM)",
    control=True,
    show=False
).add_to(map_of_poland)

folium.TileLayer(
    tiles="CartoDB dark_matter",
    name="Tło mapy (Ciemne)",
    control=True,
    show=False
).add_to(map_of_poland)

# Warstwa z lokalizacjami sklepów Żabka
layer_name = "Sklepy Żabka (klastry)"
zabka_shops_layer = folium.FeatureGroup(name=layer_name, show=True)
marker_cluster = MarkerCluster().add_to(zabka_shops_layer)

# Dodanie markerów dla każdego sklepu
for row in df_clean_zabka_shop.itertuples():
    folium.Marker(
        location=[row.lat, row.lng],
        tooltip=f"Adres: {row.city}, {row.address}",
        icon=folium.Icon(color="green", icon="frog", prefix="fa")
    ).add_to(marker_cluster)

zabka_shops_layer.add_to(map_of_poland)

# ============================================================================
# WARSTWA CHOROPLETH - OBCIĄŻENIE (LUDZIE NA SKLEP)
# ============================================================================

if gdf_population is not None:
    print("Generowanie warstwy choropleth...")
    
    # Spatial Join - łączenie sklepów z siatką
    joined_data = gpd.sjoin(gdf_zabka_shops_2180, gdf_population, how="inner", predicate="within")
    
    # Liczba sklepów w każdym kwadracie
    shops_in_grid = joined_data['index_right'].value_counts()
    gdf_population['frog_shop_count'] = 0
    gdf_population.loc[shops_in_grid.index, 'frog_shop_count'] = shops_in_grid
    
    # Analiza: tylko kwadraty z min. 1 sklepem
    gdf_analysis = gdf_population[gdf_population['frog_shop_count'] > 0].copy()
    gdf_analysis['people_per_shop'] = gdf_analysis['RES'] / gdf_analysis['frog_shop_count']
    
    gdf_analysis_map = gdf_analysis.to_crs(epsg=4326)
    max_val = gdf_analysis['people_per_shop'].max()
    quantile_bins = gdf_analysis['people_per_shop'].quantile([0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]).tolist()
    quantile_bins = sorted(set([float(x) for x in quantile_bins]))
    if quantile_bins and quantile_bins[0] > 0:
        quantile_bins.insert(0, 0.0)
    if len(quantile_bins) < 4:
        custom_bins = [0, 2500, 5000, 7500, 10000, 12500, max_val + 1]
    else:
        custom_bins = quantile_bins
    
    # Tworzenie warstwy choropleth
    choropleth = folium.Choropleth(
        geo_data=gdf_analysis_map,
        data=gdf_analysis_map,
        columns=['CODE', 'people_per_shop'],
        key_on='feature.properties.CODE',
        fill_color='Blues',
        fill_opacity=0.85,
        line_opacity=0.2,
        legend_name='Liczba mieszkańców na 1 Żabkę',
        highlight=True,
        name="Obciążenie (ludzie na sklep)",
        show=False,
        bins=custom_bins
    )
    
    # Dodanie tooltipów
    tooltip = folium.GeoJsonTooltip(
        fields=['RES', 'frog_shop_count', 'people_per_shop'],
        aliases=['Liczba ludności:', 'Liczba Żabek:', 'Ludzi na sklep:'],
        localize=True,
        sticky=False,
        labels=True,
        style="background-color: white; border: 1px solid black; border-radius: 3px;"
    )
    
    choropleth.geojson.add_child(tooltip)
    choropleth.add_to(map_of_poland)

    class BindLegendToLayer(MacroElement):
        def __init__(self, layer, legend_selector=".legend"):
            super().__init__()
            self.layer = layer
            self.legend_selector = legend_selector
            self._template = Template(u"""
            {% macro script(this, kwargs) %}
                var map = {{this.layer._parent.get_name()}};
                var layer = {{this.layer.get_name()}};
                function getLegend() {
                    return document.querySelector('{{this.legend_selector}}');
                }
                function setLegendVisibility() {
                    var legend = getLegend();
                    if (!legend) { return; }
                    if (map.hasLayer(layer)) {
                        legend.style.display = 'block';
                    } else {
                        legend.style.display = 'none';
                    }
                }
                map.on('overlayadd', function(e) {
                    if (e.layer === layer) { setLegendVisibility(); }
                });
                map.on('overlayremove', function(e) {
                    if (e.layer === layer) { setLegendVisibility(); }
                });
                setTimeout(setLegendVisibility, 200);
            {% endmacro %}
            """)

    map_of_poland.add_child(BindLegendToLayer(choropleth))

# ============================================================================
# WARSTWA BĄBELKÓW - DOSTĘPNOŚĆ
# ============================================================================

if gdf_population is not None:
    print("Generowanie warstwy z bąbelkami dostępności...")
    
    # Klasyfikacja stref
    gdf_population['zone_class'] = 'C (Daleko)'
    gdf_population.loc[gdf_population['frog_shop_count'] > 0, 'zone_class'] = 'A (W zasięgu)'
    
    # Strefa B - w sąsiedztwie (bufor 1.1 km)
    zone_A_geo = gdf_population[gdf_population['zone_class'] == 'A (W zasięgu)'].geometry
    buffer_A = zone_A_geo.buffer(1100).unary_union
    mask_B = (gdf_population['zone_class'] == 'C (Daleko)') & \
             (gdf_population['RES'] > 200) & \
             (gdf_population.intersects(buffer_A))
    gdf_population.loc[mask_B, 'zone_class'] = 'B (Sąsiedztwo)'
    
    # Przygotowanie danych dla bąbelków (min. 200 osób)
    gdf_bubbles = gdf_population[gdf_population['RES'] > 200].copy()
    gdf_bubbles_4326 = gdf_bubbles.to_crs(epsg=4326)
    gdf_bubbles_4326['centroid'] = gdf_bubbles_4326.geometry.centroid
    gdf_bubbles_4326 = gdf_bubbles_4326.set_geometry('centroid')
    gdf_bubbles_4326['lat'] = gdf_bubbles_4326.geometry.y
    gdf_bubbles_4326['lon'] = gdf_bubbles_4326.geometry.x
    
    bubbles_layer = folium.FeatureGroup(
        name="Dostępność (bąbelki)",
        show=False
    )
    
    def get_bubble_color(zone):
        if 'A' in zone: return '#00ff00'  # Zielony (Jest sklep)
        if 'B' in zone: return '#ffff00'  # Żółty (Blisko)
        return '#ff0000'                  # Czerwony (Daleko)
    
    for row in gdf_bubbles_4326.itertuples():
        radius = row.RES / 200
        if radius < 5: radius = 5
        if radius > 20: radius = 20
        
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=radius,
            color=get_bubble_color(row.zone_class),
            fill=True,
            fill_color=get_bubble_color(row.zone_class),
            fill_opacity=0.5,
            weight=0,
            tooltip=f"Ludność: {int(row.RES)}<br>Strefa: {row.zone_class}"
        ).add_to(bubbles_layer)
    
    bubbles_layer.add_to(map_of_poland)

# ============================================================================
# STYLIZACJA I KONTROLKI
# ============================================================================

# Dodanie niestandardowego CSS dla kontrolki warstw
custom_css = """
<style>
    .leaflet-control-layers {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
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

map_of_poland.get_root().html.add_child(folium.Element(custom_css))
folium.LayerControl(collapsed=True).add_to(map_of_poland)

# ============================================================================
# ZAPISYWANIE MAPY DO PLIKU HTML
# ============================================================================

print("\n=== ZAPISYWANIE MAPY ===")
output_file = 'output/mapa_zabka_interaktywna.html'
map_of_poland.save(output_file)
print(f"Mapa zapisana do: {output_file}")

print("\n=== ANALIZA ZAKOŃCZONA POMYŚLNIE ===")
