import numpy as np
import pandas as pd
import geopandas as gpd
import folium

def importFiles(file_name, verbose = False) :
    df = pd.read_csv(file_name, sep=";")
    if verbose : 
        df.head()
    return df
 

def norm(mydata) :
    mydata.columns = [
    col.lower()  # tout en minuscules
       .replace(' ', '_')       # espaces → _
       .replace("’", "_")       # apostrophes → _
       .replace('(', '')        # supprimer (
       .replace(')', '')        # supprimer )
       .replace('.', '')        # supprimer .
       .replace('…', '')
       .replace('é','e')        # accents
       .replace('è','e')
       .replace('à','a')
       .replace('ê','e')
       .replace('ç','c')
       .replace('%','percent')
       for col in mydata.columns
       ]

