# Localisation des formations post-bac et inégalités spatiales d'accès à l'enseigement supérieur

## Projet Python
*Par Blanche Choueiry, Margaux Héam et Ana Laherrere.*

# Contextualisation

Selon leur origine sociale et géographique, les jeunes ne disposent pas du même accès aux formations post-bac. Ces inégalités tiennent aussi bien aux écarts concernant la proximité aux établissements qu'à des possibilités inégales de mobilité, conditionnant l'accès à une offre de formation plus ou moins étendue. Les familles d’origine populaire en milieu rural se distinguent ainsi par un triple éloignement — social, spatial et symbolique — vis-à-vis de l’enseignement supérieur. Le système d'enseigement supérieur français reste de fait profondément segmenté et hiérarchisé. De nombreux travaux révèlent les déterminants sociaux et genrés qui structurent l'accès aux formations post-bac, mais aussi les inégalités sur le plan spatial (Frouillou, 2017). Par exemple, les élèves issus de familles aisées s’orientent prioritairement vers les établissements affichant les meilleurs taux de réussite, une stratégie rendue possible par leur plus grande mobilité spatiale, tandis que des élèves de niveau scolaire comparable mais disposant de ressources financières plus limitées privilégient des choix de proximité (Michault *et al.*, 2021).

# Objectif

Notre projet a pour but d'étudier la localisation des formations post-bac et d'en analyser les déterminants pour essayer de comprendre dans quelle mesure la localisation des formations participe aux inégalités d'accès aux différentes filières de l'enseignement supérieur.

# Sources

Pour cela, nous utilisons 4 bases de données (et les métadonnées associées à ces bases) :
- La base `contours-iris-pe.gpkg`, une base données qui contient les contours géographiques des IRIS (source : https://geoservices.ign.fr/contoursiris ; https://data.geopf.fr/telechargement/download/CONTOURS-IRIS-PE/CONTOURS-IRIS-PE_3-0__GPKG_WGS84G_FRA_2025-01-01/CONTOURS-IRIS-PE_3-0__GPKG_WGS84G_FRA_2025-01-01.7z) ;
- La base `parcoursup.csv`, qui contient des données sur chacune des formations post-bac de la plateforme parcoursup, comme le taux d'accès, le taux de boursiers, le nombre de candidats... (source : https://www.data.gouv.fr/datasets/parcoursup-2023-voeux-de-poursuite-detudes-et-de-reorientation-dans-lenseignement-superieur-et-reponses-des-etablissements) ;
- La base `population.csv`, une base qui renseigne sur le profil démographique de chaque IRIS et les métadonnées `meta_population.csv` associées à cette base (source : https://www.insee.fr/fr/statistiques/8268806) ;
- La base `revenus.csv`, une base de données sur les revenus disponibles des ménages à l'échelle des IRIS (source : https://www.data.gouv.fr/datasets/revenus-et-pauvrete-des-menages-aux-niveaux-national-et-local-revenus-localises-sociaux-et-fiscaux).

# Présentation du dépôt

Nos résultats et analyses se trouvent dans le document `code.ipynb`.

Les fonctions utilisées dans ce notebook de façon à le rendre plus lisible se trouvent dans le document `traitement_donnees.py` et les constantes utilisées (souvent des listes de noms de colonnes ou de longs textes) se trouvent dans `dataFrameUtil.py`.

Le dossier `data` contient les différentes bases de données utilisées pour réaliser ce projet.

# Bibliographie

Frouillou L. (2017), Ségrégations universitaires en Île-de-France. Inégalités d’accès et trajectoires étudiantes, Paris, La Documentation française, coll. « Études & recherches de l’Observatoire national de la vie étudiante », 207 p.

Michaut, C., Lanéelle, X. et Dutercq, Y. (2021). Les stratégies socio-spatiales des candidats aux classes préparatoires aux grandes écoles. Formation emploi, 155(3), 97-116.