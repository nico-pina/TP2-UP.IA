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
## 3. Instrucciones para reproducir el entorno y ejecutar el código
 
### 3.1 Requisitos
- Python 3.10+ (los notebooks fueron desarrollados en Windows con Anaconda)
- Jupyter Notebook / JupyterLab
  
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
| 5 | `05_predicción.ipynb` | `smoking_predictions_rf.csv`, `smoking_predictions_xgb.csv`, `smoking_predictions_comparacion.csv` | `data/predictions_csv/` |
| 6 | `06_Evaluación_de_resultados.ipynb` | Matriz de confusión del modelo final (XGBoost optimizado) | — (se muestra en el notebook) |

Todas las carpetas detalladas en el apartado 2 deben existir antes de ejecutar (o crearse manualmente), ya que los notebooks no las crean automáticamente.


# 4. Diccionario de datos

### 4.1. Dataset usado para entrenar los modelos.
50.000 filas × 27 columnas, sin nulos ni duplicados.
 
| Categoría | Columnas | Rango / Valores |
|---|---|---|
| Identificador | `ID` | 50.000 valores únicos (se descarta del modelado) |
| Demográficas | `gender`, `age` | F/M · 20–85 años |
| Antropométricas | `height(cm)`, `weight(kg)`, `waist(cm)` | 130–190 cm · 30–135 kg · 2.1–5.4 |
| Visión / audición | `eyesight(left/right)`, `hearing(left/right)` | 0–0.38 · 0.04–0.08 |
| Signos vitales | `systolic`, `relaxation` | 3.0–10.0 · 1.7–6.1 |
| Bioquímica sanguínea | `fasting blood sugar`, `Cholesterol`, `triglyceride`, `HDL`, `LDL`, `hemoglobin`, `Urine protein`, `serum creatinine`, `AST`, `ALT`, `Gtp` | rangos variables por analito (glicemia, perfil lipídico, función hepática/renal) |
| Salud bucal | `oral`, `dental caries`, `tartar` | `oral`=`Y` constante (se descarta) · caries 0/1 (21%) · tartar Y/N |
| **Objetivo** | `smoking` | 0/1 |

### 4.2. Archivos generados 

- **`data/processed/X_train.csv` / `X_test.csv`** (40.000 / 10.000 filas × 26 cols): variables numéricas escaladas + `gender_F/M`, `tartar_N/Y` (one-hot). **`y_train.csv` / `y_test.csv`**: columna `smoking`. Generados por `03_preprocesamiento.ipynb`.
- **`data/processed/smoking_predictions_rf.csv`** y **`smoking_predictions_xgb.csv`** (5.692 filas: `ID` + predicción 0/1) y **`smoking_predictions_comparacion.csv`** (`ID`, ambas predicciones y columna `coinciden`), generados por `05_predicción.ipynb` sobre el dataset de entrega (sin `smoking` real).


## 5. Resumen de experimentos, pruebas y cambios realizados
 
- **Lectura y EDA (`01`–`02`):** dataset sin nulos; se detecta que `oral` es constante (un solo valor) y se descarta. Clases desbalanceadas (63% no fumador / 37% fumador). Hemoglobina y Gtp muestran la mayor separación visual entre clases. Además, en un principio de analizó la existencia de datos duplicados, los cuales se identificaron alrededor de 5000, sin embargo, no se descartaron, ya que se asume que son datos de usuarios distintos y que sus métricas son aproximadas ( por eso la igualdad de métricas antropométricas y médicas). Se mantuvieron la totalidad de los datos.
- **Preprocesamiento (`03`):** split estratificado 80/20; `OneHotEncoder` y `StandardScaler` ajustados **solo con train** (evita data leakage) y serializados con `joblib` para reutilizarlos tal cual en la predicción.
- **Entrenamiento y optimización (`04`, x2):** En un principio de entrenaron tres modelos (Decision Tree, RandomForest y XGBoost), sin embargo, se descartó por un bajo desempeño en las métricas). Por este motivo, solo se compararon Random Forest y XGBoost, cada uno en versión base y optimizada con `GridSearchCV` (5-fold, `scoring="f1"`):


| Modelo | Accuracy (test) | F1 — Fumador | Precisión | Recall |
  |---|---|---|---|---|
  | Random Forest (base) | 0.76 | 0.69 | 0.65 | 0.73 |
  | Random Forest optimizado | 0.72 | 0.71 | 0.58 | **0.94** |
  | XGBoost (base) | 0.77 | 0.69 | 0.68 | 0.70 |
  | **XGBoost optimizado** | **0.77** | **0.70** | 0.68 | 0.73 |

- **Predicción (`05`):** se aplica el mismo pipeline del notebook `03` al dataset de entrega (5.692 registros sin la columna `smoking`). RF y XGBoost optimizados coinciden en el **79.73%** de las predicciones.
- **Evaluación final (`06`):** matriz de confusión del XGBoost optimizado sobre el set de test.


# 6. Conclusiones principales

 Las evaluaciones realizadas muestran que XGBoost optimizado se posiciona como el mejor modelo entre los cuatro comparados (RF base, RF optimizado, XGB base y XGB optimizado). Aunque mantiene el mismo nivel de accuracy que su versión base (0.77), logra un mejor desempeño en la clase 1 (fumador), elevando su F1‑score a 0.70 sin sacrificar de manera significativa la precisión. Esto en contraste con el visto en Random Forest: al optimizarlo mediante GridSearchCV con scoring="f1", el modelo incrementó el recall de la clase 0 (no fumador) , pero a costa de una caída importante en precisión, generando más falsos positivos y reduciendo su accuracy global respecto a la versión base ( de 0.76 a 0.72)  . Este resultado evidencia que optimizar con F1 no garantiza necesariamente un mejor rendimiento general.

El análisis exploratorio también mostró que variables fisiológicas como hemoglobina y GTP presentan una separación visual clara entre fumadores y no fumadores, lo que coincide con hallazgos clínicos previos y sugiere que estas características aportan buena parte del poder predictivo de los modelos.

La matriz de confusión del XGBoost optimizado confirma este comportamiento: el modelo acierta más en la clase 0 (no fumadores), pero mantiene un buen equilibrio entre precisión y recall en la clase 1 (fumadores), logrando identificar a la mayoría de los fumadores sin inflar excesivamente los falsos positivos.

Finalmente, al comparar las predicciones de Random Forest y XGBoost sobre el conjunto de entrega, ambos modelos coinciden en aproximadamente el 80% de los casos. El 20% restante corresponde a casos límites, donde los modelos discrepan y donde sería valioso profundizar, por ejemplo, revisando si estos casos coinciden con predicciones de baja confianza o probabilidades cercanas al umbral de decisión.

