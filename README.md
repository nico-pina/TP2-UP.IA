# TP2-UP.IA
Este repositorio contiene un pipeline completo orientado a predecir el estado de fumador de un conjunto de datos de personas  que contienen variables  antropométicas y analíticas médicas ( exámenes de laboratorios).  

## 1. Introducción y Objetivo del Proyecto

El objetivo principal de este proyecto fue construir y optimizar dos modelos predictivos para clasificar si un individuo es o no fumador. Para ello, se utilizó un conjunto de datos inicial etiquetados (`smoking_prediction.csv`) con **50,000 registros y 27 variables** para el entrenamiento y validación de los modelos. Seguidamente, los modelos entrenados se utilizaron para precedir el estado de tabaquismo de un conjunto de datos (`smoking_prediction_entrega.csv`) que carecía de la variable objetivo ("smoking").

El proyecto se desarrolló principalmente en dos etapas:

1. **Entrenamiento y selección de modelo**, comparando dos modelos de clasificación — **Random Forest** y **XGBoost** — entrenados sobre un dataset con la etiqueta `smoking` conocida, evaluando cada uno en su versión base y en su versión optimizada con `GridSearchCV`.
2. **Predicción sobre un dataset "de entrega"**, con las mismas columnas pero **sin** la etiqueta `smoking` (5.692 registros), aplicando el pipeline de preprocesamiento ya ajustado (encoder + scaler) y los mejores modelos de Random Forest y XGBoost, comparando además el grado de coincidencia entre ambos.

El modelo final usado para la evaluación de cierre (notebook `06_Evaluación de resultado`) es **XGBoost optimizado**, por ser el que mostró el mejor balance entre exactitud (accuracy) y F1-score sobre la clase minoritaria (fumadores).
 
## 2. Estructura del repositorio
 
El repositorio sigue la organización estándar definida para el proyecto:

```
TP2/
│
├── data/
│   ├── raw/
│   │   ├── smoking_prediction.xlsx - smoking_prediction.csv            # Dataset con etiqueta (50.000 filas, 27 cols)
│   │   └── smoking_prediction_entrega.xlsx - smoking_prediction.csv    # Dataset sin etiqueta a predecir (5.692 filas)
│   ├── processed/
│   │   ├── X_train.csv / X_test.csv                            # Features preprocesadas (encoded + escaladas)
│   │   ├── y_train.csv / y_test.csv                            # Etiquetas de train/test
│   └── external/                                               # No se utilizaron datos externos en este proyecto.
├── objetos entrenados/
│   ├── ohe.pkl                                                 # OneHotEncoder ajustado (solo con train)
│   ├── scaler.pkl                                              # StandardScaler ajustado (solo con train)
│   ├── num_cols.pkl / cat_cols.pkl                             # Listas de columnas numéricas / categóricas
│   ├── feature_columns.pkl                                     # Orden final de columnas usado en el entrenamiento

├── models/
│   ├── smoking_model_forest.pkl                                # Random Forest base
│   ├── smoking_model_forest_best.pkl                           # Random Forest optimizado
│   ├── smoking_model_xgb.pkl                                   # XGBoost base
│   └── smoking_model_xgb_best.pkl                              # XGBoost optimizado ← modelo final usado en notebooks 05 y 06
│
├── notebooks/
│   ├── 01_lectura_y_discovery.ipynb                            # Carga inicial y exploración estructural del dataset
│   ├── 02_eda.ipynb                                            # Análisis Exploratorio de Datos y visualizaciones
│   ├── 03_preprocesamiento.ipynb                               # Limpieza, encoding, escalado (feature engineering)
│   ├── 04_entrenamiento_y_optimizacion-RandomForest.ipynb      # Random Forest + GridSearchCV
│   ├── 04_entrenamiento_y_optimizacion-XGboost.ipynb           # XGBoost + GridSearchCV
│   ├── 05_predicción.ipynb                                     # Genera las predicciones finales sobre el dataset de entrega
│   └── 06_Evaluación_de_resultados.ipynb                       # Evaluación final del mejor modelo (matriz de confusión)
│
├── predictions_cvs/
│   ├── smoking_predictions_comparacion.csv                     # Comparación de las predicciones por XGBoost y Randomforest Optimizados.
│   ├── smoking_predictions_rf.csv                              # Predicciones del modelo optimizado de Randomforest.
│   ├── smoking_predictions_xgb.csv                             # Predicciones del modelo optimizado de XGBoost.
```

### 3.2 Crear entorno e instalar dependencias
 
```bash
# Crear y activar un entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / Mac
 
# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### 3.3 Orden de ejecución
Antes de la ejecución de las notebook, asegurarse que las rutas declaradas en el apartado 2 coincidan, en caso contrario, no olviden modificar las rutas relativas a cada archivo que se lee (.csv o .pkl). 
Los notebooks **deben ejecutarse en el orden que se detalla a continuación**, ya que cada uno depende de archivos generados por el anterior:


| # | Notebook | Genera | Carpeta destino |
|---|----------|--------|------------------|
| 1 | `01_lectura_y_discovery.ipynb` | Solo exploración, no genera archivos | — |
| 2 | `02_eda.ipynb` | Solo gráficos, no genera archivos | — |
| 3 | `03_preprocesamiento.ipynb` | `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` | `data/processed/` |
| 3 | `03_preprocesamiento.ipynb` | `ohe.pkl`, `scaler.pkl`, `num_cols.pkl`, `cat_cols.pkl`, `feature_columns.pkl` | `models/` |
| 4a | `04_entrenamiento_y_optimizacion-RandomForest.ipynb` | `smoking_model_forest.pkl`, `smoking_model_forest_best.pkl` | `models/` |
| 4b | `04_entrenamiento_y_optimizacion-XGboost.ipynb` | `smoking_model_xgb.pkl`, `smoking_model_xgb_best.pkl` | `models/` |
| 5 | `05_predicción.ipynb` | `smoking_predictions_rf.csv`, `smoking_predictions_xgb.csv`, `smoking_predictions_comparacion.csv` | `data/processed/` |
| 6 | `06_Evaluación_de_resultados.ipynb` | Matriz de confusión del modelo final (XGBoost optimizado) | — (se muestra en el notebook) |

