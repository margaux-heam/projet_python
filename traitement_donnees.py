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
    # La méthode de Ward est utilisée pour minimiser la variance à l’intérieur des clusters.

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

def printCarteFranceMetropolitaine(iris: pd.DataFrame, parcoursup_total: pd.DataFrame):
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

    print("Nombre de formations en France métropolitaine :", df_points_metro.shape[0])
    
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

def plotFormationPopulation(iris: pd.DataFrame):
    prop = (
        iris
        .groupby("pop_bin")[NB_FORMATIONS]
        .mean()
        .reset_index()
    )

    prop["bin_center"] = prop["pop_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(10,5))
    plt.plot(prop["bin_center"], prop[NB_FORMATIONS], marker="o")
    plt.xlabel("Nombre d'habitants")
    plt.ylabel("Proportion de formations")

def plotFormation1824(iris: pd.DataFrame):
    prop = (
        iris
        .groupby("part1824_bin")[NB_FORMATIONS]
        .mean()
        .reset_index()
    )

    prop["bin_center"] = prop["part1824_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(10,5))
    plt.plot(prop["bin_center"], prop[NB_FORMATIONS], marker="o")
    plt.xlabel("Part des 18-24 ans (%)")
    plt.ylabel("Probabilité d'avoir une formations")
    plt.title("Probabité d'avoir une formation selon la part des 18-24 ans")
    plt.grid(True)
    plt.show()

def plotFormation1117(iris: pd.DataFrame):
    prop = (
        iris
        .groupby("part1117_bin")[NB_FORMATIONS]
        .mean()
        .reset_index()
    )

    prop["bin_center"] = prop["part1117_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(10,5))
    plt.plot(prop["bin_center"], prop[NB_FORMATIONS], marker="o")
    plt.xlabel("Part des 11–17 ans (%)")
    plt.ylabel("Proportion de formations")
    plt.title("Proportion de formations selon la part des 11–17 ans")
    plt.grid(True)
    plt.show()

def constructModelFormation(iris: pd.DataFrame):
    df_model = iris[[NB_FORMATIONS, "pop", "cluster_label", "type_iris_label"]].copy()

    # Nettoyage : on enlève les lignes avec NA sur les variables du modèle
    df_model = df_model.dropna()

    logit_model = smf.logit(formula=f"{NB_FORMATIONS} ~ " + " + ".join(terms1), data=df_model)
    res = logit_model.fit()
    table = pd.DataFrame({
        "coef_logit": res.params,
        "odds_ratio": np.exp(res.params),
        "p_value": res.pvalues
    })
    return table

def printCarteParis1(iris: pd.DataFrame, parcoursup_total : pd.DataFrame):

    # 2) Filtrer uniquement Paris (75)
    gdf_idf = iris[iris["code_dept"].isin(idf_deps)].copy()
    print("Nombre d'IRIS en région parisienne :", gdf_idf.shape[0])

    for col in gdf_idf.columns:
        if gdf_idf[col].dtype.name in ["interval", "category"]:
            gdf_idf[col] = gdf_idf[col].astype(str)

    # 3) Palette de couleurs pour les types de quartiers (clusters)
    cluster_colors = {
        "tres_pauvre": "#b30000",
        "pauvre":      "#fc8d59",
        "moyen":       "#fee08b",
        "riche":       "#91bfdb",
        "tres_riche":  "#4575b4",
    }

    def style_cluster(feature):
        label = feature["properties"].get("cluster_label")
        color = cluster_colors.get(label, "#cccccc")  # gris si NaN
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.6,
        }

    # 4) Filtrer les formations qui sont dans un IRIS IDF
    idf_iris_codes = set(gdf_idf["code_iris"].astype(str).unique())
    df_points_idf = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(idf_iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    # 4b) Créer la colonne tres_select
    df_points_idf[TRES_SELECT] = df_points_idf["taux_acces"] < 50
    df_points_idf[TRES_SELECT] = df_points_idf[TRES_SELECT].astype(bool)

    print("Nombre de formations en IDF :", df_points_idf.shape[0])

    # 5) Créer une carte centrée sur Paris
    m = folium.Map(
        location=[48.8566, 2.3522],
        zoom_start=12,
        max_zoom=12,
        min_zoom=12,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # 6) Ajouter les polygones IRIS colorés selon le type de quartier
    folium.GeoJson(
        gdf_idf,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
            localize=True
        ),
    ).add_to(m)

    # 7) Ajouter les formations en points
    for _, row in df_points_idf.iterrows():
        color = "darkred" if row["tres_select"] else "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # 8) Ajouter le LayerControl
    folium.LayerControl().add_to(m)

    return m

def printCarteStrasbourg(iris: pd.DataFrame, parcoursup_total: pd.DataFrame):

    # 2) Filtrer uniquement Strasbourg (67)
    idf_deps = ["67"]
    gdf_stras = iris[iris["code_dept"].isin(idf_deps)].copy()
    print("Nombre d'IRIS dans le Bas-Rhin :", gdf_stras.shape[0])

    for col in gdf_stras.columns:
        if gdf_stras[col].dtype.name in ["interval", "category"]:
            gdf_stras[col] = gdf_stras[col].astype(str)

    # 3) Palette de couleurs pour les types de quartiers (clusters)
    cluster_colors = {
        "tres_pauvre": "#b30000",
        "pauvre":      "#fc8d59",
        "moyen":       "#fee08b",
        "riche":       "#91bfdb",
        "tres_riche":  "#4575b4",
    }

    def style_cluster(feature):
        label = feature["properties"].get("cluster_label")
        color = cluster_colors.get(label, "#cccccc")  # gris si NaN
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.6,
        }

    # 4) Filtrer les formations qui sont dans un IRIS IDF
    stras_iris_codes = set(gdf_stras["code_iris"].astype(str).unique())
    df_points_stras = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(stras_iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    # 4b) Créer la colonne tres_select
    df_points_stras[TRES_SELECT] = df_points_stras["taux_acces"] < 50
    df_points_stras[TRES_SELECT] = df_points_stras[TRES_SELECT].astype(bool)

    print("Nombre de formations dans le 67 :", df_points_stras.shape[0])

    # 5) Créer une carte centrée sur Strasbourg
    m = folium.Map(
        location=[48.583, 7.745],
        zoom_start=12,
        max_zoom=13,
        min_zoom=12,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # 6) Ajouter les polygones IRIS colorés selon le type de quartier
    folium.GeoJson(
        gdf_stras,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
            localize=True
        ),
    ).add_to(m)

    # 7) Ajouter les formations en points
    for _, row in df_points_stras.iterrows():
        color = "darkred" if row["tres_select"] else "red"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # 8) Ajouter le LayerControl
    folium.LayerControl().add_to(m)
    return m

def plotFormationSelectPopulation(iris: pd.DataFrame):
    prop = (
        iris
        .groupby("pop_bin")[TRES_SELECT]
        .mean()
        .reset_index()
    )

    prop["bin_center"] = prop["pop_bin"].apply(lambda x: x.mid)

    plt.figure(figsize=(10,5))
    plt.plot(prop["bin_center"], prop[TRES_SELECT], marker="o")
    plt.xlabel("Nombre d'habitants")
    plt.ylabel("Proportion de formations sélectives")
    plt.title("Proportion de formations sélectives selon le nombre d'habitants du quartier")
    plt.grid(True)
    plt.show()

def constructModelSelect1(iris: pd.DataFrame):
    df_model = iris[[TRES_SELECT, "pop", "cluster_label", "type_iris_label"]].copy()

    # Nettoyage : on enlève les lignes avec NA sur les variables du modèle
    df_model = df_model.dropna()
    
    logit_model = smf.logit(formula=f"{TRES_SELECT} ~ " + " + ".join(terms1), data=df_model)
    res = logit_model.fit()
    table = pd.DataFrame({
        "coef_logit": res.params,
        "odds_ratio": np.exp(res.params),
        "p_value": res.pvalues
    })
    return table

def constructModelSelect2(parcoursup_total: pd.DataFrame):
    df_model = parcoursup_total[[TRES_SELECT, "pop", "cluster_label", "type_iris_label"]].copy()

    # Nettoyage : on enlève les lignes avec NA sur les variables du modèle
    df_model = df_model.dropna()
    
    logit_model = smf.logit(formula=f"{TRES_SELECT} ~ " + " + ".join(terms1), data=df_model)
    res = logit_model.fit()
    table = pd.DataFrame({
        "coef_logit": res.params,
        "odds_ratio": np.exp(res.params),
        "p_value": res.pvalues
    })
    return table

def plotBoursierQuartier(parcoursup_total: pd.DataFrame):
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

def constructModelBoursier1(parcoursup_total: pd.DataFrame):
    model = smf.ols(
        "admis_boursier ~ C(cluster_label) + C(selectivite) + pop",
        data=parcoursup_total
    ).fit()
    return model

def constructModelBoursier2(parcoursup_total: pd.DataFrame):
    model = smf.ols(
        "admis_boursier ~ C(cluster_label) + C(selectivite) + pop + C(type_form, Treatment(reference='Licence'))",
        data=parcoursup_total
    ).fit()
    return model

def printCarteParis2(iris: pd.DataFrame, parcoursup_total: pd.DataFrame):
    # Filtrer uniquement Paris (75)
    gdf_idf = iris[iris["code_dept"].isin(idf_deps)].copy()

    # Sécuriser les colonnes catégorielles / Interval
    for col in gdf_idf.columns:
        if pd.api.types.is_categorical_dtype(gdf_idf[col]) or pd.api.types.is_interval_dtype(gdf_idf[col]):
            gdf_idf[col] = gdf_idf[col].astype(str)

    for col in parcoursup_total.columns:
        if pd.api.types.is_categorical_dtype(parcoursup_total[col]) or pd.api.types.is_interval_dtype(parcoursup_total[col]):
            parcoursup_total[col] = parcoursup_total[col].astype(str)

    # Palette de couleurs pour les types de quartiers (clusters)
    cluster_colors = {
        "tres_pauvre": "#b30000",
        "pauvre":      "#fc8d59",
        "moyen":       "#fee08b",
        "riche":       "#91bfdb",
        "tres_riche":  "#4575b4",
    }

    def style_cluster(feature):
        label = feature["properties"].get("cluster_label")
        color = cluster_colors.get(label, "#cccccc")  # gris si NaN
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.6,
        }

    # Filtrer les formations qui sont dans un IRIS IDF
    idf_iris_codes = set(gdf_idf["code_iris"].astype(str).unique())
    df_points_idf = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(idf_iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    # Palette de couleurs pour type_form_agg
    form_colors = {
        "BTS": "darkgreen",
        "CPGE / Grande Ecole": "darkred",
        "Autre": "darkblue"
    }

    # Créer la carte centrée sur Paris
    m = folium.Map(
        location=[48.8566, 2.3522],
        zoom_start=12,
        max_zoom=12,
        min_zoom=12,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # Ajouter les polygones IRIS
    folium.GeoJson(
        gdf_idf,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
            localize=True
        ),
    ).add_to(m)

    # Ajouter les points des formations
    for _, row in df_points_idf.iterrows():
        color = form_colors.get(row["type_form_agg"], "black")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # Ajouter LayerControl
    folium.LayerControl().add_to(m)

    return m

def printCarteLille(iris: pd.DataFrame, parcoursup_total: pd.DataFrame):
    # Filtrer uniquement Lille (59)
    gdf_lille = iris[iris["code_dept"].isin(lille_deps)].copy()

    # Sécuriser les colonnes catégorielles / Interval
    for col in gdf_lille.columns:
        if pd.api.types.is_categorical_dtype(gdf_lille[col]) or pd.api.types.is_interval_dtype(gdf_lille[col]):
            gdf_lille[col] = gdf_lille[col].astype(str)

    for col in parcoursup_total.columns:
        if pd.api.types.is_categorical_dtype(parcoursup_total[col]) or pd.api.types.is_interval_dtype(parcoursup_total[col]):
            parcoursup_total[col] = parcoursup_total[col].astype(str)

    # Palette de couleurs pour les types de quartiers (clusters)
    cluster_colors = {
        "tres_pauvre": "#b30000",
        "pauvre":      "#fc8d59",
        "moyen":       "#fee08b",
        "riche":       "#91bfdb",
        "tres_riche":  "#4575b4",
    }

    def style_cluster(feature):
        label = feature["properties"].get("cluster_label")
        color = cluster_colors.get(label, "#cccccc")  # gris si NaN
        return {
            "fillColor": color,
            "color": "black",
            "weight": 0.3,
            "fillOpacity": 0.6,
        }

    # Filtrer les formations qui sont dans un IRIS IDF
    lille_iris_codes = set(gdf_lille["code_iris"].astype(str).unique())
    df_points_lille = parcoursup_total[
        parcoursup_total["code_iris"].astype(str).isin(lille_iris_codes)
    ].dropna(subset=["latitude", "longitude"]).copy()

    # Palette de couleurs pour type_form_agg
    form_colors = {
        "BTS": "darkgreen",
        "CPGE / Grande Ecole": "darkred",
        "Autre": "darkblue"
    }

    # Créer la carte centrée sur Lille
    m = folium.Map(
        location=[50.632, 3.057],
        zoom_start=12,
        max_zoom=12,
        min_zoom=12,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        zoomControl=False
    )

    # Ajouter les polygones IRIS
    folium.GeoJson(
        gdf_lille,
        name="Quartiers (IRIS)",
        style_function=style_cluster,
        tooltip=folium.GeoJsonTooltip(
            fields=["nom_iris", "nom_commune", "cluster_label"],
            aliases=["IRIS", "Commune", "Type de quartier"],
            localize=True
        ),
    ).add_to(m)

    # Ajouter les points des formations
    for _, row in df_points_lille.iterrows():
        color = form_colors.get(row["type_form_agg"], "black")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.8,
        ).add_to(m)

    # Ajouter LayerControl
    folium.LayerControl().add_to(m)

    return m

