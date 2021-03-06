#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:13:24 2019

@author: carlamardonesburgos
"""

#*************************************************************************
# Autor: John Atkinson
# Fecha: 24/07/2019
# Contenidos:  Programa y funciones para realizar Clustering jerárquico Aglomerativo
#*************************************************************************


#**********************************************************************
# Cargar bibliotecas relevantes
#**********************************************************************


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage  
import matplotlib.pyplot as plt
import pandas as pd  

#**********************************************************************
# Funciones utilizadas por el programa
#**********************************************************************


#*******************************************************************************************
# CargarDatos:carga datos (con biblioteca Panda) desde un archivo CSV (FileName) y retorna 
#             una tabla con datos "escalados" pero con dos columnas solamente (para poder 
#             graficar posteriormente: UnitCost y TotalRevenue
#*******************************************************************************************
pd.set_option('display.max_columns', 20)
def CargarDatos(FileName):
    datatable = pd.read_csv(FileName)
    df = pd.DataFrame(datatable,columns=['age','workclass','fnlwgt','education','education.num','marital.status','occupation','relationship','race','sex','capital.gain','capital.loss','hours.per.week','native.country','income'])
    df['age'] = ConvertirTipoDato(df,'age')
    df['workclass'] = ConvertirTipoDato(df,'workclass')
#    df['fnlwgt'] = ConvertirTipoDato(df,'fnlwgt')
    df['education'] = ConvertirTipoDato(df,'education')
#    df['education.num'] = ConvertirTipoDato(df,'education.num')
    df['marital.status'] = ConvertirTipoDato(df,'marital.status')
    df['occupation'] = ConvertirTipoDato(df,'occupation')
    df['relationship'] = ConvertirTipoDato(df,'relationship')
    df['race'] = ConvertirTipoDato(df,'race')
    df['sex'] = ConvertirTipoDato(df,'sex')
#    df['capital.gain'] = ConvertirTipoDato(df,'capital.gain')
#    df['capital.loss'] = ConvertirTipoDato(df,'capital.loss')
    df['hours.per.week'] = ConvertirTipoDato(df,'hours.per.week')
    df['native.country'] = ConvertirTipoDato(df,'native.country')
    df['income'] = ConvertirTipoDato(df,'income')
    return(df)


#*******************************************************************************************
# GraficarDatos: Grafica una lista de pares de puntos de coordendas (workclass, age) (UnitCost,TotalRevenue)
#*******************************************************************************************   
def GraficarDatos(Puntos):    
  labels = range(1, len(Puntos))  
  plt.figure(figsize=(10, 7))  
  plt.subplots_adjust(bottom=0.1)  
  plt.scatter(Puntos['workclass'],Puntos['age'], label='True Position')
  for label, x, y in zip(labels, Puntos['workclass'],Puntos['age']):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
  plt.show()  
 
    
#*******************************************************************************************
# DibujarDendograma: Genera dendograma a partir de una lista de pares de puntos de datos
#*******************************************************************************************
def DibujarDendograma(Puntos):
    linked = linkage(Puntos, 'single')  #Distancia del tipo single-link
    plt.figure(figsize=(10, 7))  
    dendrogram(linked,  
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()  

#*******************************************************************************************
# Clustering_Aglomerativo: genera clusters jerárquicos agomerativos a partir de 
#             una lista de puntos (X) y el numero de clusters (NumClus)
#              retorna los clusters generados
#*******************************************************************************************
# Posibles parámetros de clustering están acá:
#       https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
def Clustering_Aglomerativo(X,NumClus):
    # Linkage tipo "ward" minimiza la varianza de cluster al mezclarlos (merge)
    clusters = AgglomerativeClustering(n_clusters=NumClus, affinity='euclidean', linkage='ward')  
    clusters.fit_predict(X) 
    return(clusters)


# *******************************************   
# Programa Principal
# *******************************************

#NombreArchivo = 'F:\JOHN\CURSOS\Business Intelligence\Ejercicios-BI\muestra.csv'
NombreArchivo ="/Users/carlamardonesburgos/Desktop/muestra.csv"

Datos = CargarDatos(NombreArchivo)
NumClusters = 3
GraficarDatos(Datos)
DibujarDendograma(Datos)
clusters = Clustering_Aglomerativo(Datos,NumClusters)
#plt.scatter(Datos['UnitCost'],Datos['TotalRevenue'], c=clusters.labels_, cmap='rainbow')  
