#*******************************************************************************************
# Autor: John Atkinson
# Fecha: 24/07/2019
# Contenidos:  Programa y funciones para realizar Clasificación por arboles de decision
#******************************************************************************************
#
# IMPORTANTE:  DEBE instalar una biblioteca:
#              conda install graphviz   o bien    pip  install graphviz
#
#**********************************************************************
# Cargar bibliotecas relevantes
#**********************************************************************

import pandas as pd  
from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.tree.export import export_text
from sklearn.tree import DecisionTreeClassifier 
import graphviz

#**********************************************************************
# Funciones utilizadas por el programa
#**********************************************************************

#*******************************************************************************************
# ConvertirTipoDato: Dada una tabla de datos, convierte el tipo "categorial" de una columna 
#                    a datos de tipo numéricos para poderlos manipular (asigna códigos numéricos)
# Retorna: Datos de una columna categorial convertidos a datos numéricos
#*******************************************************************************************

def ConvertirTipoDato(Datos,NombreColumna):
    Datos[NombreColumna] = Datos[NombreColumna].astype('category')
    Datos[NombreColumna] = Datos[NombreColumna].cat.codes
    return(Datos[NombreColumna])   
    
#*******************************************************************************************
# CargarDatos: carga datos solamente con 4 columnas de interés
# Retorna: tabla de datos panda
#*******************************************************************************************
    
def CargarDatos(FileName):   
    datatable = pd.read_csv(FileName)  
    df = pd.DataFrame(datatable,columns=['age','relationship','sex','hours.per.week','income'])
    df['age'] = ConvertirTipoDato(df,'age')
    #df['education.num'] = ConvertirTipoDato(df,'education.num')
    df['relationship'] = ConvertirTipoDato(df,'relationship')
    df['sex'] = ConvertirTipoDato(df,'sex')
    df['hours.per.week'] = ConvertirTipoDato(df,'hours.per.week')
    df['income'] = ConvertirTipoDato(df,'income')
    return(df)
    
#*******************************************************************************************
# Train_DecisionTrees: entrena un modelo de arboles de decision a partir de datos de entrenamiento (X,Y)
# Retorna: modelo entrenado
#*******************************************************************************************

def Train_DecisionTrees(X_train,Y_train):
    modelo = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    modelo = modelo.fit(X_train,Y_train)
    return(modelo)

#*******************************************************************************************
# SepararDatos: Toma una tabla de datos y los separa en una proporcion "Prop" en dos conjuntos:
#               uno de entrenamiento y otro de prueba
# Retorna: Datos separados (training, testing)
#*******************************************************************************************

def SepararDatos(Prop,Data):
    # "Prop" es la proporcion de datos para entrenamiento
    largo = len(Data)-1
    pos = int(Prop*largo)
    return(Data[0:pos],Data[pos+1:largo])
    

#*******************************************************************************************
# MostrarArbol: Dado un arbol de decision, las columnas de la tabla y las variables dependientes (decision)
#               visualiza el árbol obtenido
#*******************************************************************************************  

def MostrarArbol(modelo,columnas,targetnames):
    data = export_graphviz(modelo,out_file=None,
                         feature_names=columnas,
                         class_names=targetnames,   
                         filled=True, rounded=True,  
                         special_characters=True)
    graph = graphviz.Source(data)
    return(graph)
    
    
# *******************************************   
# Programa Principal
# *******************************************

#NombreArchivo = 'F:\JOHN\CURSOS\Business Intelligence\Ejercicios-BI\muestra.csv'
NombreArchivo ="/Users/carlamardonesburgos/Desktop/DataEntrega3.csv"



Datos = CargarDatos(NombreArchivo)


Train,Test = SepararDatos(0.4,Datos)  # 70% de datos para entrenamiento

Xtrain = Train.drop('income', axis=1)
Ytrain = Train.income

Xtest = Test.drop('income', axis=1)
Ytest = Test.income

Clasif = Train_DecisionTrees(Xtrain,Ytrain)
Ypred = Clasif.predict(Xtest)

print("Accuracy:",metrics.accuracy_score(Ytest, Ypred))
#print(metrics.classification_report(Ytest, Ypred))
#print(metrics.confusion_matrix(Ytest, Ypred))

targetnames = [str(i) for i in list(set(Ytrain))]
arbol = MostrarArbol(Clasif,list(Xtrain.columns),targetnames)
#arbol.view()
texto_arbol = export_text(Clasif, feature_names=list(Xtrain.columns))
print(texto_arbol)