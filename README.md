# Proyecto Final - Data Science YPF

Proyecto final del curso de Data Science (EnergIA Digital), aplicando de punta a punta el flujo de trabajo aprendido: AED, limpieza de datos, modelo supervisado y modelo no supervisado.

## Dataset

El dataset utilizado es **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)**, una competencia pública de Kaggle. Contiene 150.000 registros históricos de clientes de entidades financieras, con 10 variables de comportamiento crediticio (utilización de crédito, ingresos, atrasos de pago, cantidad de créditos abiertos, etc.). Se incluye en `dataset/`: `cs-training.csv` (datos con etiqueta, los que se usan en el proyecto), `cs-test.csv` y `sampleEntry.csv` (archivos de la competencia, sin etiqueta, no se usan) y el diccionario de datos (`dataset.xls`). `cs-training-clean.csv` es la versión limpia que genera el notebook 01.

## Objetivo

Predecir la variable `SeriousDlqin2yrs` (1 = el cliente tuvo un atraso de pago de 90 días o más en los siguientes 2 años) y, además, segmentar a los clientes en perfiles sin usar esa etiqueta, para validar si el comportamiento crediticio por sí solo ya separa a los clientes de alto y bajo riesgo.

Es un problema de **scoring de riesgo crediticio** con un fuerte desbalance de clases: solo el 6,7% de los clientes cae en mora. Por eso la evaluación usa ROC-AUC, PR-AUC, precisión, recall y F2 en lugar de accuracy.

## Criterio metodológico

- **El test se usa una única vez, al final.** La elección del modelo, los hiperparámetros y el umbral de decisión se toman con validación cruzada (5 folds estratificados) sobre el set de entrenamiento.
- **Sin fuga de información.** El notebook 01 solo aplica reglas fijas que no aprenden de los datos (flags y valores imposibles a NaN). La imputación con la mediana se hace en el notebook 02, dentro de un `Pipeline`, por lo que se calcula solo con datos de entrenamiento.
- **División estratificada:** train (120.000) y test (30.000) conservan la misma proporción de morosos.

## Hallazgos del análisis exploratorio (AED)

- **Desbalance de clases:** 6,7% de mora seria. Un modelo que predice siempre "no mora" tiene 93% de accuracy y no detecta a ningún moroso.
- **Faltantes:** `MonthlyIncome` (19,8%) y `NumberOfDependents` (2,6%). Se marcan con un flag y se dejan como NaN para imputarlos en el modelado.
- **Valores anómalos tratados con reglas fijas (sin eliminar filas):**

| Problema | Registros | Tratamiento |
|---|---|---|
| `age = 0` | 1 | NaN: error de carga |
| Códigos 96/98 en las tres variables de atraso | 269 | Flag `atraso_codigo_especial` + NaN. No son cantidades reales (no hay valores entre 17 y 96), pero esos clientes tienen 54,6% de mora |
| `RevolvingUtilizationOfUnsecuredLines` > 10 | 241 | NaN: errores de carga. Los valores entre 1 y 10 se conservan (clientes pasados de su límite, de alto riesgo) |
| `DebtRatio` sin ingreso real (ingreso faltante, 0 o 1) | 31.970 | Flag `DebtRatio_invalido` + NaN: en esos casos el valor es la deuda absoluta, no un cociente |
| `DebtRatio` > 5 con ingreso real | 321 | NaN: criterio de negocio |

- **La limpieza cambió las conclusiones:** con los datos crudos, la utilización de crédito parecía no tener relación con la mora (correlación 0,00) y las variables de atraso parecían redundantes entre sí (0,98-0,99). Con los datos limpios, la utilización correlaciona 0,28 con la mora, los atrasos 0,27-0,31, y entre sí 0,22-0,31.

Detalle completo en [`notebooks/01_analisis_exploratorio.ipynb`](notebooks/01_analisis_exploratorio.ipynb).

## Modelo supervisado (clasificación)

Se compararon varios modelos con la ROC-AUC de validación cruzada sobre train:

| Modelo | ROC-AUC (CV en train) |
|---|---|
| Dummy (clase mayoritaria) | 0,5 (no discrimina) |
| Regresión Logística | 0,855 |
| Árbol de Decisión | 0,842 |
| Random Forest (sin ajustar) | 0,839 |
| **Random Forest + GridSearchCV** | **0,862** |

El mejor modelo es un **Random Forest** ajustado con `GridSearchCV` (`n_estimators=400`, `max_depth=10`, `min_samples_leaf=20`). El hiperparámetro con más impacto fue `min_samples_leaf`: exigir al menos 20 clientes por hoja evita que los árboles memoricen el entrenamiento.

**Umbral de decisión:** elegido con probabilidades "fuera de fold" (`cross_val_predict`) sobre train, maximizando F2 (el recall pesa el doble, porque no detectar a un moroso cuesta más que rechazar a un buen cliente). Umbral: **0,56**.

**Resultados en test (usado una única vez):**

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,866 (0,862 en validación cruzada: sin sobreajuste) |
| PR-AUC | 0,396 (un modelo al azar daría 0,067) |
| Recall | 0,694: detecta a 1.392 de los 2.005 morosos |
| Precisión | 0,263: 1 de cada 4 alertas es un moroso real |
| F2 | 0,523 (0,519 en validación cruzada) |

Las variables más importantes son la utilización de crédito revolvente (35%) y las tres variables de atraso (90+ días 19%, 30-59 días 17%, 60-89 días 10%).

Detalle completo en [`notebooks/02_modelo_supervisado.ipynb`](notebooks/02_modelo_supervisado.ipynb).

## Modelo no supervisado (clustering)

Se aplicó `KMeans` sin usar la etiqueta. Antes del clustering, los 269 clientes con código 96/98 se separan para analizarlos aparte, se imputan los faltantes con la mediana, se aplica `log1p` a las variables con cola larga y se estandarizan todas.

Las métricas de calidad no señalan un número de clusters claramente mejor (silhouette entre 0,18 y 0,21 para k de 2 a 8), lo que indica que los clientes no forman grupos naturales bien separados. Se eligió **k = 4 por interpretabilidad**: la segmentación se usa como herramienta de negocio. Al cruzar con la tasa real de mora:

| Segmento | Clientes | Perfil | Tasa real de mora |
|---|---|---|---|
| Mayores con poco uso de crédito | 51.393 | 65 años en promedio, usan el 12% de su límite | 1,6% |
| Ingresos altos y más productos | 56.995 | Mayor ingreso, 11 líneas de crédito | 5,6% |
| Jóvenes con uso medio | 33.512 | 40 años, usan el 53% de su límite | 7,7% |
| Con atrasos graves previos | 7.831 | Usan el 78% de su límite, 1,7 atrasos de 90+ días | 41,7% |
| Código 96/98 (fuera del clustering) | 269 | Atrasos desconocidos | 54,6% |

**Hallazgo clave:** sin ver la etiqueta, la segmentación separa perfiles con riesgos de 1,6% a 41,7% (promedio: 6,7%). El segmento de mayor riesgo está definido por el historial de atrasos de 90+ días, en línea con el modelo supervisado.

### Sensibilidad a valores extremos y estabilidad

Para comprobar que los segmentos no estén definidos por valores anómalos, el notebook 03 compara dos versiones de KMeans (`k = 4`) sobre los mismos 149.731 clientes (sin los 269 con código 96/98):

- **Sin tratamiento:** variables crudas, solo imputación y estandarización, sin `log1p` ni reglas para outliers.
- **Tratada:** reglas de limpieza del notebook 01, imputación, `log1p` y estandarización.

Como los números de cluster son etiquetas arbitrarias, antes de comparar se alinea cada cluster sin tratamiento con el cluster tratado con el que más clientes comparte (Hungarian matching).

| Cluster tratado | Clientes sin tratamiento | Mora sin tratamiento | Clientes tratado | Mora tratado | Cambio de mora |
|---|---|---|---|---|---|
| Jóvenes con uso medio | 55.414 | 7,81% | 33.512 | 7,67% | −0,14 pp |
| Mayores con poco uso de crédito | 52.873 | 3,08% | 51.393 | 1,64% | −1,44 pp |
| Ingresos altos y más productos | 38.620 | 6,12% | 56.995 | 5,62% | −0,51 pp |
| Con atrasos graves previos | 2.824 | 55,17% | 7.831 | 41,72% | −13,45 pp |

- **El tratamiento reasigna al 29,1% de los clientes:** ese porcentaje cambia de cluster al pasar de la versión sin tratamiento a la tratada. Los perfiles se mantienen reconocibles, pero las fronteras cambian.
- **Sin tratamiento, KMeans aísla un grupo chico de valores extremos:** 2.824 clientes con 55% de mora, aun habiendo sacado los códigos 96/98. Con el tratamiento, el segmento de riesgo pasa a 7.831 clientes definidos por su comportamiento (alta utilización del crédito y atrasos de 90+ días). Esto respalda el tratamiento de outliers.
- **Las métricas internas no son comparables entre versiones:** sin tratamiento, silhouette 0,209 y Davies-Bouldin 1,326; tratada, 0,182 y 1,536. Cada versión mide distancias en un espacio distinto (con y sin logaritmo), por lo que estas diferencias no indican cuál es mejor.
- **Estabilidad:** repitiendo KMeans con cinco semillas distintas, el Adjusted Rand Index (ARI) entre corridas es 0,974 en promedio (mínimo 0,942): la segmentación casi no depende de la inicialización. Esto mide estabilidad del algoritmo, no que los grupos sean naturales.
- **Perfiles robustos:** el notebook complementa los promedios con la mediana y los percentiles 95 y 99 de cada variable por cluster. En ningún cluster quedan valores imposibles: el percentil 99 de la utilización llega a 1,87 y el de `DebtRatio` a 2,37.

Detalle completo en [`notebooks/03_modelo_no_supervisado.ipynb`](notebooks/03_modelo_no_supervisado.ipynb).

## Cambios respecto de la primera entrega

A partir de la devolución recibida:

1. **Umbral de decisión:** antes se elegía probando valores sobre el test; ahora se elige con validación cruzada sobre train y el test se usa una sola vez. Con el mismo criterio, la comparación de modelos pasó a hacerse con la AUC de validación cruzada.
2. **Imputación antes del split:** antes la mediana se calculaba con todo el dataset; ahora se calcula dentro de un `Pipeline`, solo con train.
3. **KMeans y valores extremos:** en la versión anterior, el cluster de mayor riesgo (269 clientes) eran exactamente los registros con código 98, y los perfiles mostraban promedios imposibles (13 en `RevolvingUtilizationOfUnsecuredLines`, 674 en `DebtRatio`). Al tratar los códigos aparte, corregir los ratios y aplicar logaritmo, aparece un segmento de riesgo de 7.831 clientes definido por su comportamiento real. Un análisis de sensibilidad cuantifica el efecto del tratamiento (29,1% de clientes reasignados) y la estabilidad de la solución (ARI promedio 0,974).
4. **Notebooks ejecutados de principio a fin** y carpetas `.ipynb_checkpoints` excluidas con `.gitignore`.

La limpieza también mejoró a la Regresión Logística, que pasó de 0,824 a 0,862 de ROC-AUC en test.

## Limitaciones y próximos pasos

- La Regresión Logística no se ajustó y quedó cerca del Random Forest (0,855 contra 0,862 en validación cruzada).
- Con `class_weight="balanced"`, las probabilidades del modelo no son probabilidades reales de mora; habría que calibrarlas si se necesitan.
- El umbral maximiza F2; en un caso real se elegiría con el costo económico de cada tipo de error.
- **Validación temporal:** en scoring crediticio lo ideal es entrenar con clientes del pasado y evaluar con clientes más recientes, porque el comportamiento cambia con la economía y con el tipo de clientes (drift). El dataset no tiene una variable de fecha, por lo que no se pudo hacer esa validación; sería el primer paso con datos reales.
- Próximos pasos: probar Gradient Boosting, usar importancia por permutación y validar el modelo con datos fechados.

## Estructura del repositorio

```
dataset/        Dataset original (Kaggle) y versión limpia generada en el AED
notebooks/      01_analisis_exploratorio.ipynb, 02_modelo_supervisado.ipynb, 03_modelo_no_supervisado.ipynb
```

Para reproducir los resultados, ejecutar los notebooks en orden: el 01 genera `dataset/cs-training-clean.csv`, que leen el 02 y el 03.