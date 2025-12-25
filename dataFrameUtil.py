column_mapping_iris = {
    "H": "habitat",
    "A": "activité",
    "D": "divers",
    "Z": "autre"
}

column_mapping_parcoursup = {
    "Statut de l’établissement de la filière de formation (public, privé…)": "secteur",
    "Établissement": "etab",
    "Code départemental de l’établissement": "dep",
    "Région de l’établissement": "reg",
    "Académie de l’établissement": "academie",
    "Académie de l’établissement": "academie",
    "Commune de l’établissement": "commune",
    "Filière de formation détaillée": "filiere_det",
    "Sélectivité": "selectivite",
    "Filière de formation très agrégée": "type_form",
    "Filière de formation détaillée bis": "filiere",
    "Coordonnées GPS de la formation": "coord_gps",
    "Capacité de l’établissement par formation": "nb_etud",
    "Effectif total des candidats pour une formation": "nb_cand",
    "Effectif total des candidats ayant accepté la proposition de l’établissement (admis)": "nb_admis",
    "Dont effectif des candidates admises": "nb_fille",
    "Dont effectif des admis boursiers néo bacheliers": "nb_boursier",
    "Effectif des admis néo bacheliers généraux": "nb_general",
    "Effectif des admis néo bacheliers technologiques": "nb_techno",
    "Effectif des admis néo bacheliers professionnels": "nb_pro",
    "Dont effectif des admis néo bacheliers sans mention au bac": "nb_sansmention",
    "Dont effectif des admis néo bacheliers avec mention Assez Bien au bac": "nb_abien",
    "Dont effectif des admis néo bacheliers avec mention Bien au bac": "nb_bien",
    "Dont effectif des admis néo bacheliers avec mention Très Bien au bac": "nb_tbien",
    "Dont effectif des admis néo bacheliers avec mention Très Bien avec félicitations au bac": "nb_felicitations",
    "Dont effectif des admis issus de la même académie": "nb_memeac",
    "Dont effectif des admis issus de la même académie (Paris/Créteil/Versailles réunies)": "nb_memeac2",
    "% d’admis néo bacheliers boursiers": "admis_boursier",
    "Taux d’accès": "taux_acces"
}

columns_to_keep_parcoursup = [
    "secteur", "etab", "dep", "reg", "academie", "commune",
    "filiere_det", "selectivite", "type_form", "filiere",
    "coord_gps", "nb_etud", "nb_cand", "nb_admis",
    "nb_fille", "nb_boursier", "nb_general", "nb_techno", "nb_pro",
    "nb_sansmention", "nb_abien", "nb_bien", "nb_tbien", "nb_felicitations",
    "nb_memeac", "nb_memeac2", "admis_boursier", "taux_acces"
]

columns_to_remove_revenus = ["iris", "tp60", "note", "d2", "d3", "d4", "d6", "d7", "d8"]

variables_to_analyze_in_clusters = ["tp60", "med", "q1", "q3", "rd", "gi", "pact", "ptsa", "pcho", "pben", "ppen", "ppat", "ppsoc", "ppmini", "pimpot"]

clusters_labels = ["tres_pauvre", "pauvre", "moyen", "riche", "tres_riche"]

columns_to_add_to_parcoursup = ['code_iris', 'nom_iris', 'geometry', 'nom_commune', 'type_iris', 'type_iris_label', "med", "rd", "ppsoc", "cluster_label", "pop", "pop1117", "pop1824", "pop6074", "pop75p", "pop15p_cs3", "pop15p_cs5", "pop15p_cs6", "pop_imm"]

Q_list = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]

NB_FORMATIONS = "nb_formations"
TRES_SELECT = "tres_select"

terms1 = ["C(cluster_label, Treatment(reference='moyen'))", "C(type_iris_label, Treatment(reference='habitat'))", "pop"]

ordre_clusters = ["tres_pauvre", "pauvre", "moyen", "riche", "tres_riche"]

idf_deps = ["75"]
lille_deps = ["59"]