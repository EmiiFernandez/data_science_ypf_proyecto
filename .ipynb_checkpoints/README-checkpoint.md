# Proyecto Final - Data Science YPF

Proyecto final del curso de Data Science (EnergIA Digital), aplicando de punta a punta el flujo de trabajo aprendido: AED, feature engineering, modelo supervisado y modelo no supervisado.

## Dataset

El dataset utilizado es **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)**, una competencia publica de Kaggle. Contiene ~150.000 registros historicos de clientes de entidades financieras, con variables de comportamiento crediticio (utilizacion de credito, ingresos, atrasos de pago, cantidad de creditos abiertos, etc.). Se incluye en este repositorio en `dataset/` (`cs-training.csv`, `cs-test.csv`, `sampleEntry.csv` y el diccionario de datos).

## Resumen del proyecto

Se trabajo sobre un problema de **scoring de riesgo crediticio**: a partir del historial de un cliente, estimar la probabilidad de que caiga en una mora seria en los proximos dos anos. Es un caso de uso real de la industria financiera, con un fuerte desbalance de clases que obligo a elegir metricas de evaluacion cuidadosamente (ROC-AUC y F1 en vez de accuracy).

## Objetivo

Predecir la variable `SeriousDlqin2yrs` (1 = el cliente tuvo un atraso de pago >= 90 dias en los siguientes 2 anos) y, ademas, segmentar a los clientes en perfiles de riesgo sin usar esa etiqueta, para validar si el comportamiento crediticio por si solo ya separa a los clientes de alto y bajo riesgo.

## Hallazgos del analisis exploratorio (AED)

- Fuerte desbalance de clases: solo ~6.7% de los clientes tuvo mora seria.
- Dos columnas con valores faltantes (`MonthlyIncome`, `NumberOfDependents`), imputadas con la mediana/0 y un flag de "faltante" para no perder esa senal.
- Valores anomalos conocidos del dataset: edades en 0 (reemplazadas por la mediana) y un codigo especial (96/98) en las columnas de atraso de pago, que se conservo como flag en vez de eliminarlo.
- Las variables de historial de atraso (30-59, 60-89 y 90+ dias) son las que mas correlacionan con la variable objetivo: el comportamiento pasado es el mejor predictor de riesgo futuro.

Detalle completo en [`notebooks/01_analisis_exploratorio.ipynb`](notebooks/01_analisis_exploratorio.ipynb).

## Modelos ajustados

### Modelo supervisado (clasificacion)

Se compararon varios algoritmos sobre el dataset limpio, usando ROC-AUC como metrica principal por el desbalance de clases:

| Modelo | ROC-AUC |
|---|---|
| Dummy (clase mayoritaria) | referencia — no discrimina clases |
| Regresion Logistica | 0.824 |
| Arbol de Decision | 0.850 |
| Random Forest (base) | 0.846 |
| **Random Forest + GridSearchCV** | **0.867** (test) / 0.863 (CV) |

El mejor modelo fue un **Random Forest** ajustado con `GridSearchCV` (`n_estimators=400`, `max_depth=10`, `min_samples_leaf=20`). Limitar la profundidad y el minimo de muestras por hoja evito el overfitting frente al Random Forest sin ajustar. Las variables mas importantes fueron el historial de atrasos y la utilizacion de credito revolvente.

Detalle completo en [`notebooks/02_modelo_supervisado.ipynb`](notebooks/02_modelo_supervisado.ipynb).

### Modelo no supervisado (clustering)

Se aplico `KMeans` (sin usar la etiqueta objetivo) sobre variables de comportamiento crediticio, eligiendo `k=4` por metodo del codo y coeficiente de silhouette. Al cruzar los clusters resultantes con la tasa real de mora, se encontraron 4 perfiles claramente diferenciados:

| Cluster | Clientes | Perfil | Tasa real de mora |
|---|---|---|---|
| 1 | 61.962 | Mayores (~63 anos), baja utilizacion de credito | 2.4% |
| 2 | 39.002 | Ingresos mas altos, mas lineas de credito abiertas | 5.5% |
| 0 | 48.767 | Alta utilizacion de credito revolvente y debt ratio | 12.9% |
| 3 | 269 | Casos con codigo de atraso especial (96/98) | 54.6% |

**Hallazgo clave:** la segmentacion no supervisada, sin ver la etiqueta, aislo por su cuenta un grupo minoritario (cluster 3) con una tasa de mora real de mas del 50%, coincidiendo con los registros que tienen el codigo anomalo de atraso. Esto sugiere que ese codigo especial no es solo un error de carga de datos, sino que marca a los clientes de mayor riesgo real.

Detalle completo en [`notebooks/03_modelo_no_supervisado.ipynb`](notebooks/03_modelo_no_supervisado.ipynb).

## Estructura del repositorio

```
dataset/        Dataset original (Kaggle) y version limpia generada en el AED
notebooks/      01_analisis_exploratorio.ipynb, 02_modelo_supervisado.ipynb, 03_modelo_no_supervisado.ipynb
```
