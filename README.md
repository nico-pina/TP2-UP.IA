# TP2-UP.IA
Este repositorio contiene un pipeline completo orientado a predecir el estado de fumador de un conjunto de datos de personas  que contienen variables  antropométicas y analíticas médicas ( exámenes de laboratorios).  

## 1. Introducción y Objetivo del Proyecto

El objetivo principal de este proyecto fue construir y optimizar dos modelos predictivos para clasificar si un individuo es o no fumador. Para ello, se utilizó un conjunto de datos inicial etiquetados (`smoking_prediction.csv`) con **50,000 registros y 27 variables** para el entrenamiento y validación de los modelos. Seguidamente, los modelos entrenados se utilizaron para precedir el estado de tabaquismo de un conjunto de datos (`smoking_prediction_entrega.csv`) que carecía de la variable objetivo ("smoking").

El proyecto se desarrolló principalmente en dos etapas:

1. **Entrenamiento y selección de modelo**, comparando dos modelos de clasificación — **Random Forest** y **XGBoost** — entrenados sobre un dataset con la etiqueta `smoking` conocida, evaluando cada uno en su versión base y en su versión optimizada con `GridSearchCV`.
2. **Predicción sobre un dataset "de entrega"**, con las mismas columnas pero **sin** la etiqueta `smoking` (5.692 registros), aplicando el pipeline de preprocesamiento ya ajustado (encoder + scaler) y los mejores modelos de Random Forest y XGBoost, comparando además el grado de coincidencia entre ambos.
3. Finalmente, el modelo seleccionado como predictor fue el XGboost optimizado según las siguientes razones:

   a). Mayor capacidad de generalización. En el proceso de optimziación, XGboost logró "frenar" el overfitting, dejando las métodas de train y test muy cercanas una de la otra (Train:0.79 ; Test:0.7). Esto garantiza que el modelo será estable y confiable cuando reciba datos reales del mundo exterior.

