import numpy as np
import pandas as pd
import geopandas as gpd
import folium 
import requests
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from shapely.geometry import Point
from dataFrameUtil import *

def getRevenus() -> pd.DataFrame:
    """
    Lit la base de données "revenus" dans un DataFrame.

    Returns:
        DataFrame: le contenu de "revenus" en renommant les colonnes
    """
    revenus = pd.read_csv("data/revenus.csv", sep=";")
    revenus.columns = (
        revenus.columns
        .str.replace("^DISP_", "", regex=True)
        .str.replace("18$", "", regex=True)
        .str.lower()
        )
    return revenus

def getIris(revenus: pd.DataFrame) -> gpd.GeoDataFrame:
    iris = gpd.read_file("data/contours-iris-pe.gpkg")
    iris["type_iris_label"] = iris["type_iris"].map(column_mapping_iris)
    iris = iris.merge(
            revenus,
            left_on="code_iris",
            right_on="iris",
            how="left"
        )
    
    iris = iris.merge(
        getPopulation(),
        left_on="code_iris",
        right_on="COD_MOD",
        how="left"
    )

    iris = iris.to_crs(epsg=4326)
    return iris

def getPopulation() -> pd.DataFrame:
    population = pd.read_csv("data/population.csv", sep=";")
    population.columns = (
        population.columns
        .str.replace("^P21_", "", regex=True)
        .str.replace("^C21_", "", regex=True)
        .str.lower()
        )

    population["iris"] = population["iris"].astype(str)

    # Ajout des nouvelles colonnes à "population"
    population = population.merge(
        getMetaPopulation(),
        left_on="iris",
        right_on="COD_MOD",
        how="left"
    )
    return population

def getParcoursup(iris: gpd.GeoDataFrame) -> pd.DataFrame:
    parcoursup = pd.read_csv("data/parcoursup.csv", sep=";")
    parcoursup = parcoursup.rename(columns=column_mapping_parcoursup)
    parcoursup = parcoursup[columns_to_keep_parcoursup]
    parcoursup[['latitude', 'longitude']] = parcoursup['coord_gps'].str.split(',', expand=True)
    parcoursup['latitude'] = parcoursup['latitude'].astype(float)
    parcoursup['longitude'] = parcoursup['longitude'].astype(float)

    parcoursup_total = gpd.sjoin(
            gpd.GeoDataFrame(
                parcoursup,
                geometry=gpd.points_from_xy(parcoursup.longitude, parcoursup.latitude),
                crs="EPSG:4326"
            ), 
            iris[columns_to_add_to_parcoursup], 
            how="left",
            predicate="within"
        )

    parcoursup_total["part_memeac"] = parcoursup_total["nb_memeac"] / parcoursup_total["nb_etud"]
    parcoursup_total["part_memeac2"] = parcoursup_total["nb_memeac2"] / parcoursup_total["nb_etud"]

    parcoursup_total["part_memeac2_decile"] = pd.qcut(parcoursup_total["part_memeac2"], 
                                                q=len(Q_list), 
                                                labels=Q_list)
    return parcoursup_total

def getMetaPopulation() -> pd.DataFrame:
    meta = pd.read_csv("data/meta_population.csv", sep=";")
    # dans la base de données meta, on garde seulement les lignes correspondant à la variable IRIS
    meta = meta[meta["COD_VAR"] == "IRIS"]

    # ne garder que le code et le nom
    meta = meta[["COD_MOD", "LIB_MOD"]]

    # On enlève les "0" au début des codes IRIS pour qu'ils correspondent aux codes des autres bases
    meta["COD_MOD"] = (
        meta["COD_MOD"].astype(str)
                            .apply(lambda x: x[1:] if x.startswith("0") else x)
    )

    return meta

def getRevenusCah(revenus: pd.DataFrame) -> pd.DataFrame:
    # conserver uniquement les colonnes pertinentes
    rev_cah = revenus.drop(columns=columns_to_remove_revenus)

    # imputation des valeurs manquantes
    for col in rev_cah.columns:
        rev_cah[col] = rev_cah[col].fillna(rev_cah[col].median())

    # La normalisation transforme les données pour que chaque colonne ait moyenne 0 et écart type 1, ce qui évite qu’une variable domine les autres et permet des analyses plus fiables.
    
    return StandardScaler().fit_transform(rev_cah)

def doCAH(rev_cah: pd.DataFrame):
    return linkage(rev_cah, method='ward')

def plotDendrogramme(cah):
    plt.figure(figsize=(12, 6))
    dendrogram(cah, truncate_mode="level", p=5)
    plt.title("Dendrogramme CAH")
    plt.show()

    ## La méthode de Ward est utilisée pour minimiser la variance à l’intérieur des clusters.

def plotCoude(cah):
    last_rev = cah[:, 2][::-1]

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 16), last_rev[:15], marker='o')
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Distance de fusion")
    plt.title("Méthode du coude (1 à 15 clusters)")
    plt.grid(True)
    plt.show() 

def createClusters(revenus: pd.DataFrame, cah):
    revenus["cluster"] = fcluster(cah, 5, criterion='maxclust')

def analyzeClusters(revenus: pd.DataFrame) -> pd.DataFrame:
    summary = revenus.groupby("cluster")[variables_to_analyze_in_clusters].mean()
    total = revenus[variables_to_analyze_in_clusters].mean()
    summary_with_total = pd.concat([summary, total.to_frame().T], axis=0)
    summary_with_total.index = list(summary.index) + ["Total"]
    return summary_with_total

def renameClusters(revenus: pd.DataFrame):
    cluster_order = revenus.groupby("cluster")["med"].median().sort_values()
    mapping = {cluster: clusters_labels[i] for i, cluster in enumerate(cluster_order.index)}
    revenus["cluster_label"] = revenus["cluster"].map(mapping)

def plotParcoursupMobilitesBoursiers(parcoursup: pd.DataFrame):
    moyennes = (
        parcoursup
        .groupby("part_memeac2_decile")["admis_boursier"]
        .mean()
    )

    plt.figure(figsize=(8,5))
    plt.bar(moyennes.index, moyennes.values)
    plt.ylabel("Taux moyen de boursiers dans la formation")
    plt.xlabel("Décile de la part d'étudiants venant de la même académie")
    plt.title("Part moyenne d’admis boursiers de la formation selon la part d'étudiants issus de la même académie")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.ylim(0, 35)
    plt.show()

def updateIris(iris: pd.DataFrame, parcoursup_total: pd.DataFrame) -> pd.DataFrame:
    # Nombre total de formations par IRIS
    total_form = (
        parcoursup_total.groupby("code_iris")
        .size()
        .reset_index(name=NB_FORMATIONS)
    )

    total_form[NB_FORMATIONS] = total_form[NB_FORMATIONS].fillna(0).astype(int)
    iris = iris.merge(total_form, on="code_iris", how="left")
    iris[NB_FORMATIONS] = iris[NB_FORMATIONS].fillna(0).astype(int)
    iris[NB_FORMATIONS] = (iris[NB_FORMATIONS] > 0).astype(int)
    return iris

def printCarteFranceMetropolitaine(iris: pd.DataFrame, parcoursup_total: pd.DataFrame):
    # =====================================================
    # 1) Préparer les IRIS – France métropolitaine uniquement
    # =====================================================
    iris = iris.copy()

    iris["code_iris"] = iris["code_iris"].astype(str)
    iris["code_dept"] = iris["code_iris"].str[:2]

    deps_metro = (
        [f"{i:02d}" for i in range(1, 96)]
        + ["2A", "2B"]
    )

    gdf_metro = iris[iris["code_dept"].isin(deps_metro)].copy()

    print("Nombre d'IRIS en France métropolitaine :", gdf_metro.shape[0])

    # Sécurisation JSON (important pour Folium)
    for col in gdf_metro.columns:
        if isinstance(gdf_metro[col].dtype, (pd.CategoricalDtype, pd.IntervalDtype)):
            gdf_metro[col] = gdf_metro[col].astype(str)

    # =====================================================
    # 2) Filtrer les formations situées en France métropolitaine
    # =====================================================
    metro_iris_codes = set(gdf_metro["code_iris"])

    df_points_metro = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(metro_iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    print("Nombre de formations en France métropolitaine :", df_points_metro.shape[0])

    # =====================================================
    # 3) Palette de couleurs pour les quartiers
    # =====================================================
    cluster_colors = {
        "tres_pauvre": "#b30000",
        "pauvre":      "#fc8d59",
        "moyen":       "#fee08b",
        "riche":       "#91bfdb",
        "tres_riche":  "#4575b4",
    }

    def style_cluster(feature):
        label = feature["properties"].get("cluster_label")
        return {
            "fillColor": cluster_colors.get(label, "#cccccc"),
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.6,
        }

    # =====================================================
    # 4) Créer la carte – France entière
    # =====================================================

    m = folium.Map(
        location=[46.6, 2.5],  # centre France
        zoom_start=6,
        min_zoom=5,            # empêche le dézoom au-delà de la France
        max_zoom=13,
        zoomControl=True
    )

    # =====================================================
    # 5) Ajouter les polygones IRIS
    # =====================================================
    folium.GeoJson(
        gdf_metro,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
            localize=True
        ),
    ).add_to(m)

    # =====================================================
    # 6) Ajouter les formations
    # =====================================================
    for _, row in df_points_metro.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="darkred",
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # =====================================================
    # 7) Contrôles et affichage
    # =====================================================
    folium.LayerControl().add_to(m)
    #m

