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

def get_revenus() -> pd.DataFrame:
    revenus = pd.read_csv("data/revenus.csv", sep=";")

    # simplification du nom des variables : retirer les préfixes et suffixes communs à toutes les variables
    revenus.columns = (
        revenus.columns
        .str.replace("^DISP_", "", regex=True)
        .str.replace("18$", "", regex=True)
        .str.lower()
        )
    return revenus

def get_iris(revenus: pd.DataFrame) -> gpd.GeoDataFrame:
    iris = gpd.read_file("data/contours-iris-pe.gpkg")

    # associer le nom des différents types d'iris selon leur encodage
    iris["type_iris_label"] = iris["type_iris"].map(column_mapping_iris)
    
    # fusion de la base "iris" (données géographiques des IRIS) avec la base "revenus"
    iris = iris.merge(
            revenus,
            left_on="code_iris",
            right_on="iris",
            how="left"
        )
    
    # fusion de la base "iris" avec la base "population" (données démographiques des IRIS)
    iris = iris.merge(
        get_population(),
        left_on="code_iris",
        right_on="COD_MOD",
        how="left"
    )

    # choix du système de coordonnées
    iris = iris.to_crs(epsg=4326)

    # construire une variable catégorielle du nombre d'habitants par quartier
    iris["pop_bin"] = pd.cut(
        iris["pop"],
        bins= range(0, 5000, 100),
        include_lowest=True,
        right=False
    )

    # construire des variables sur la part des jeunes dans la population du quartier
    iris["part1117"] = (iris["pop1117"] / iris["pop"]) * 100
    iris["part1824"] = (iris["pop1824"] / iris["pop"]) * 100

    # construire une variable catégorielle de la part de 18-24 ans dans le quartier
    iris["part1824_bin"] = pd.cut(
        iris["part1824"],
        bins=range(0, 30, 2),
        include_lowest=True,
        right=False
    )

    # construire une variable catégorielle de la part de 11-17 ans dans le quartier
    iris["part1117_bin"] = pd.cut(
        iris["part1117"],
        bins=range(0, 30, 2),
        include_lowest=True,
        right=False
    )

    # créer une variable avec le numéro du département
    iris["code_iris"] = iris["code_iris"].astype(str)
    iris["code_dept"] = iris["code_iris"].str[:2]
    return iris

def get_population() -> pd.DataFrame:
    # ouverture de la base "population" (données démographiques au niveau des IRIS)
    population = pd.read_csv("data/population.csv", sep=";")

    # simplification du nom des variables : retirer les préfixes communs à (presque) toutes les variables
    population.columns = (
        population.columns
        .str.replace("^P21_", "", regex=True)
        .str.replace("^C21_", "", regex=True)
        .str.lower()
        )

    population["iris"] = population["iris"].astype(str)

    # Ajout des nouvelles colonnes à "population" qui viennent des métadonnées de population (base "meta")
    population = population.merge(
        get_meta_population(),
        left_on="iris",
        right_on="COD_MOD",
        how="left"
    )
    return population

def get_parcoursup(iris: gpd.GeoDataFrame) -> pd.DataFrame:
    # ouverture du fichier
    parcoursup = pd.read_csv("data/parcoursup.csv", sep=";")
    # changer les noms de colonnes et garder seulement les colonnes utiles
    parcoursup = parcoursup.rename(columns=column_mapping_parcoursup)
    parcoursup = parcoursup[columns_to_keep_parcoursup]

    # préparation des coordonnées pour faire le merge avec les iris
    parcoursup[['latitude', 'longitude']] = parcoursup['coord_gps'].str.split(',', expand=True)
    parcoursup['latitude'] = parcoursup['latitude'].astype(float)
    parcoursup['longitude'] = parcoursup['longitude'].astype(float)

    # fusion de la base parcoursup avec la base des iris
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

    # créer des colonnes sur la part des élèves qui viennent de la même académie que la formation
    parcoursup_total["part_memeac"] = parcoursup_total["nb_memeac"] / parcoursup_total["nb_etud"]
    parcoursup_total["part_memeac2"] = parcoursup_total["nb_memeac2"] / parcoursup_total["nb_etud"]

    parcoursup_total["part_memeac2_decile"] = pd.qcut(parcoursup_total["part_memeac2"], 
                                                q=len(Q_list), 
                                                labels=Q_list)
    
    # créer une colonne pour les formations très sélectives
    parcoursup_total["tres_select"] = (parcoursup_total["taux_acces"] < 50).astype(int)

    # créer une colonne sur le type de formation : BTS/CPGE/Autre
    conditions = [
        parcoursup_total["type_form"] == "BTS",
        parcoursup_total["type_form"].isin(["CPGE", "Ecole d'Ingénieur", "Ecole de commerce"])
    ]
    choices = [
        "BTS",
        "CPGE / Grande Ecole"
    ]
    parcoursup_total["type_form_agg"] = np.select(conditions, choices, default="Autre")
    return parcoursup_total

def get_meta_population() -> pd.DataFrame:
    # ouverture de la base "meta", une base associée à la base "population" nécessaire pour fusionner les bases "iris" et "population"
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

def get_revenus_cah(revenus: pd.DataFrame) -> pd.DataFrame:
    # conserver uniquement les colonnes pertinentes
    rev_cah = revenus.drop(columns=columns_to_remove_revenus)

    # imputation des valeurs manquantes
    for col in rev_cah.columns:
        rev_cah[col] = rev_cah[col].fillna(rev_cah[col].median())
    # normalisation des variables
    return StandardScaler().fit_transform(rev_cah)

def do_cah(rev_cah: pd.DataFrame):
    return linkage(rev_cah, method='ward')

def plot_dendrogramme(cah):
    plt.figure(figsize=(12, 6))
    dendrogram(cah, truncate_mode="level", p=5)
    plt.title("Dendrogramme CAH")
    plt.show()

def plot_coude(cah):
    last_rev = cah[:, 2][::-1]
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 16), last_rev[:15], marker='o')
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Distance de fusion")
    plt.title("Méthode du coude (1 à 15 clusters)")
    plt.grid(True)
    plt.show() 

def create_clusters(revenus: pd.DataFrame, cah):
    revenus["cluster"] = fcluster(cah, 5, criterion='maxclust')

def analyze_clusters(revenus: pd.DataFrame) -> pd.DataFrame:
    summary = revenus.groupby("cluster")[variables_to_analyze_in_clusters].mean()
    total = revenus[variables_to_analyze_in_clusters].mean()
    summary_with_total = pd.concat([summary, total.to_frame().T], axis=0)
    summary_with_total.index = list(summary.index) + ["Total"]
    return summary_with_total

def rename_clusters(revenus: pd.DataFrame):
    cluster_order = revenus.groupby("cluster")["med"].median().sort_values()
    mapping = {cluster: clusters_labels[i] for i, cluster in enumerate(cluster_order.index)}
    revenus["cluster_label"] = revenus["cluster"].map(mapping)

def update_iris(iris: pd.DataFrame, parcoursup_total: pd.DataFrame) -> pd.DataFrame:
    # Ajouter des données issues de la base "parcoursup_total" à la base "iris"
    # Nombre total de formations par IRIS
    total_form = (
        parcoursup_total.groupby("code_iris")
        .size()
        .reset_index(name=NB_FORMATIONS)
    )
    iris = iris.merge(total_form, on="code_iris", how="left")
    iris[NB_FORMATIONS] = iris[NB_FORMATIONS].fillna(0).astype(int)
    # rendre la variable binaire
    iris[NB_FORMATIONS] = (iris[NB_FORMATIONS] > 0).astype(int)

    # Nombre de formations très sélectives par IRIS
    result = parcoursup_total.groupby("code_iris")[[TRES_SELECT]].sum().reset_index()
    iris = iris.merge(result, on="code_iris", how="left")
    iris[TRES_SELECT] = iris[TRES_SELECT].fillna(0)
    # rendre la variable binaire
    iris[TRES_SELECT] = (iris[TRES_SELECT] > 0).astype(int)
    return iris

def plot_parcoursup_mobilites_boursiers(parcoursup: pd.DataFrame):
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

def print_carte_ville(
    iris: pd.DataFrame,
    parcoursup_total: pd.DataFrame,
    code_dept: str,
    center: tuple,
    zoom: int = 11
):
    # Filtrer le département
    gdf = iris[iris["code_dept"] == code_dept].copy()

    # Rendre compatible JSON
    for col in gdf.columns:
        if gdf[col].dtype.name in ["interval", "category"]:
            gdf[col] = gdf[col].astype(str)

    # Palette clusters
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

    # Filtrer les formations
    iris_codes = set(gdf["code_iris"].astype(str))
    df_points = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(iris_codes)
    ].dropna(subset=["latitude", "longitude"])

    # Carte
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        max_zoom=zoom,
        min_zoom=zoom,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # Polygones IRIS
    folium.GeoJson(
        gdf,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
        ),
    ).add_to(m)

    # Points formations (une seule couleur)
    for _, row in df_points.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="red",
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    return m


def print_carte_france_metropolitaine(iris: pd.DataFrame, parcoursup_total: pd.DataFrame):
    # Préparer les IRIS – France métropolitaine uniquement
    deps_metro = (
        [f"{i:02d}" for i in range(1, 96)]
        + ["2A", "2B"]
    )

    gdf_metro = iris[iris["code_dept"].isin(deps_metro)].copy()

    # Sécurisation JSON (important pour Folium)
    for col in gdf_metro.columns:
        if isinstance(gdf_metro[col].dtype, (pd.CategoricalDtype, pd.IntervalDtype)):
            gdf_metro[col] = gdf_metro[col].astype(str)

    # Filtrer les formations situées en France métropolitaine
    metro_iris_codes = set(gdf_metro["code_iris"])

    df_points_metro = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(metro_iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()
    
    # Créer la carte – France entière
    m = folium.Map(
        location=[46.6, 2.5],  # centre France
        zoom_start=6,
        min_zoom=6,
        max_zoom=6,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # Ajouter les formations
    for _, row in df_points_metro.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="darkred",
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # Contrôles
    folium.LayerControl().add_to(m)
    return m

def plot_formation_demographie(
    iris: pd.DataFrame,
    bin_col: str,
    value_col: str,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    figsize=(10, 5),
    grid: bool = True
):
    prop = (
        iris
        .groupby(bin_col)[value_col]
        .mean()
        .reset_index()
    )

    prop["bin_center"] = prop[bin_col].apply(lambda x: x.mid)

    plt.figure(figsize=figsize)
    plt.plot(prop["bin_center"], prop[value_col], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    if grid:
        plt.grid(True)

    plt.show()

def construct_logit_model(
    df: pd.DataFrame,
    y: str,
    terms: list[str]
) -> pd.DataFrame:

    # Sous-ensemble des variables utiles
    df_model = df[[y] + logit_variable].copy()

    # Nettoyage
    df_model = df_model.dropna()

    # Modèle logit
    formula = f"{y} ~ " + " + ".join(terms)
    res = smf.logit(formula=formula, data=df_model).fit()

    # Table des résultats
    table = pd.DataFrame({
        "coef_logit": res.params,
        "odds_ratio": np.exp(res.params),
        "p_value": res.pvalues
    })

    return table

def print_carte_ville_select(
    iris: pd.DataFrame,
    parcoursup_total: pd.DataFrame,
    code_depts: list[str],
    center: tuple[float, float],
    zoom_start: int = 12,
    min_zoom: int | None = None,
    max_zoom: int | None = None
):
    # Filtrer la zone
    gdf = iris[iris["code_dept"].isin(code_depts)].copy()

    # Préparation JSON
    for col in gdf.columns:
        if gdf[col].dtype.name in ["interval", "category"]:
            gdf[col] = gdf[col].astype(str)

    # Palette clusters
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

    # Filtrer les formations
    iris_codes = set(gdf["code_iris"].astype(str))
    df_points = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    # Variable très sélective
    df_points[TRES_SELECT] = (df_points["taux_acces"] < 50)

    # Carte
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        min_zoom=min_zoom or zoom_start,
        max_zoom=max_zoom or zoom_start,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # Polygones IRIS
    folium.GeoJson(
        gdf,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
        ),
    ).add_to(m)

    # Points formations
    for _, row in df_points.iterrows():
        color = "darkred" if row[TRES_SELECT] else "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

def plot_boursier_quartier(parcoursup_total: pd.DataFrame):
    moyennes = (
        parcoursup_total
        .groupby("cluster_label")["admis_boursier"]
        .mean()
        .reindex(ordre_clusters)
    )

    plt.figure(figsize=(8,5))
    plt.bar(moyennes.index, moyennes.values)
    plt.ylabel("Taux moyens de boursiers dans la formation")
    plt.xlabel("Caractéristiques économiques du quartier de la formation")
    plt.title("Part moyenne d’admis boursiers selon le niveau de richesse du quartier")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.ylim(0, 35)
    plt.show()

def construct_model_boursier(
    df: pd.DataFrame,
    terms: list[str],
    y: str = "admis_boursier"
):
    formula = f"{y} ~ " + " + ".join(terms)
    model = smf.ols(formula, data=df).fit()
    return model

def print_carte_type_formation(
    iris: pd.DataFrame,
    parcoursup_total: pd.DataFrame,
    deps: list[str],
    center: tuple[float, float],
    zoom: int = 12
):
    # Filtrer la zone
    gdf = iris[iris["code_dept"].isin(deps)].copy()

    # Sécuriser les colonnes catégorielles / Interval (GeoDataFrame)
    for col in gdf.columns:
        if (
            pd.api.types.is_categorical_dtype(gdf[col])
            or pd.api.types.is_interval_dtype(gdf[col])
        ):
            gdf[col] = gdf[col].astype(str)

    # Sécuriser les colonnes catégorielles / Interval (points)
    for col in parcoursup_total.columns:
        if (
            pd.api.types.is_categorical_dtype(parcoursup_total[col])
            or pd.api.types.is_interval_dtype(parcoursup_total[col])
        ):
            parcoursup_total[col] = parcoursup_total[col].astype(str)

    # Palette clusters
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

    # Filtrer les formations dans les IRIS de la zone
    iris_codes = set(gdf["code_iris"].astype(str).unique())
    df_points = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    # Palette type de formation
    form_colors = {
        "BTS": "darkgreen",
        "CPGE / Grande Ecole": "darkred",
        "Autre": "darkblue"
    }

    # Carte
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        min_zoom=zoom,
        max_zoom=zoom,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # Polygones IRIS
    folium.GeoJson(
        gdf,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
            localize=True
        ),
    ).add_to(m)

    # Points formations
    for _, row in df_points.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=form_colors.get(row["type_form_agg"], "black"),
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m

