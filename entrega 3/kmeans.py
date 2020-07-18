#*************************************************************************
# Autor: John Atkinson
# Fecha: 24/07/2019
# Contenidos:  Programa y funciones para realizar Clustering K-means
#*************************************************************************


#**********************************************************************
# Cargar bibliotecas relevantes
#**********************************************************************


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd  
from sklearn.preprocessing import StandardScaler


#**********************************************************************
# Funciones utilizadas por el programa
#**********************************************************************


#*******************************************************************************************
# CargarDatos:carga datos (con biblioteca Panda) desde un archivo CSV (FileName) y retorna 
#             una tabla con datos "escalados" pero con dos columnas solamente (para poder 
#             graficar posteriormente: UnitCost y TotalRevenue
#*******************************************************************************************

def CargarDatos(FileName):   
    datatable = pd.read_csv(FileName)  
    # Seleccionar 2 columnas  de la tabla 'Data'
    df = pd.DataFrame(datatable,columns=['UnitCost','TotalRevenue'])
    return(df)


#*******************************************************************************************
# Clustering_Kmeans: realiza clustering K-means a partir de la tabla de datos y 
#             un cierto número de clusters (NumClust)
# Retorna:  clusters y centroides
#*******************************************************************************************

def Clustering_Kmeans(Datos,NumClust):    
   clusters = KMeans(n_clusters=NumClust).fit(Datos)
   centroids = clusters.cluster_centers_
   # Opcionalmente podemos mostrar los centroides: print(centroids)
   return((clusters,centroids))
   
   
#*******************************************************************************************
# GraficarClusters: grafica los clusters a partir de la tabla de datos y los centroides
#*******************************************************************************************
   
def GraficarClusters(df,clusters,centroids):
   plt.scatter(df['UnitCost'], df['TotalRevenue'], c= clusters.labels_.astype(float), s=50, alpha=0.5)
   plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
 


# *******************************************   
# Programa Principal
# *******************************************

#NombreArchivo = 'C:/Users/atkin/Desktop/Ejemplos-BI/muestra.csv'
NombreArchivo = 'E:\JOHN\CURSOS\Business Intelligence\Ejercicios-BI\muestra.csv'


Datos = CargarDatos(NombreArchivo)
# print(Datos.describe())
K = 3     # Num. de clusters
(clusters,centroids) = Clustering_Kmeans(Datos,K)
GraficarClusters(Datos,clusters,centroids)
print(clusters.inertia_) # Mostrar el error SSE
print(clusters.labels_)  # Mostrar los rótulos de cluster de cada elemento del dataset


# Graficar SSE versus el num. de clusters
#plt.figure(figsize=(6, 6))
#plt.plot(list_k, sse, '-o')
#plt.xlabel(r'Número de clusters *k*')
#plt.ylabel('Suma de distancias cuadradas');

