# Proyecto Final - Data Science YPF

Proyecto final del curso de Data Science (EnergIA Digital), aplicando de punta a punta el flujo de trabajo aprendido: AED, limpieza de datos, modelo supervisado y modelo no supervisado.

## Dataset

El dataset utilizado es **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)**, una competencia pública de Kaggle. Contiene 150.000 registros históricos de clientes de entidades financieras, con 10 variables de comportamiento crediticio (utilización de crédito, ingresos, atrasos de pago, cantidad de créditos abiertos, etc.). Se incluye en `dataset/`: `cs-training.csv` (datos con etiqueta, los que se usan en el proyecto), `cs-test.csv` y `sampleEntry.csv` (archivos de la competencia, sin etiqueta, no se usan) y el diccionario de datos (`dataset.xls`). `cs-training-clean.csv` es la versión limpia que genera el notebook 01.

## Objetivo

Predecir la variable `SeriousDlqin2yrs` (1 = el cliente tuvo un atraso de pago de 90 días o más en los siguientes 2 años) y, además, segmentar a los clientes en perfiles sin usar esa etiqueta, para validar si el comportamiento crediticio por sí solo ya separa a los clientes de alto y bajo riesgo.

Es un problema de **scoring de riesgo crediticio** con un fuerte desbalance de clases: solo el 6,7% de los clientes cae en mora. Por eso la evaluación usa ROC-AUC, PR-AUC, precisión, recall y F2 en lugar de accuracy.

## Criterio metodológico

- **El test se usa una única vez, al final.** La elección del modelo, los hiperparámetros y el umbral de decisión se toman sobre train. La evaluación principal usa validación cruzada anidada de 3 folds: una CV interna selecciona hiperparámetros y umbral dentro de cada fold, y una CV externa estima el rendimiento fuera de muestra. Se usa una grilla reducida para que el experimento sea reproducible en equipos modestos. Esta reducción es una decisión práctica para poder avanzar con un equipo limitado, no la configuración ideal para un caso real.
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
| **Random Forest + GridSearchCV** | Se actualiza al ejecutar la grilla reducida |

El mejor modelo se selecciona con un **Random Forest** ajustado mediante `GridSearchCV`. Para mantener el experimento reproducible en equipos modestos, la grilla reducida usa `n_estimators=100`, `max_depth` entre 8 y 10, y `min_samples_leaf` entre 10 y 20, con 3 folds en el ajuste y en la validación cruzada anidada. En la configuración original se exploraba una grilla más amplia con 5 folds, que es la alternativa preferible en una práctica real porque permite una búsqueda más exhaustiva y una estimación más estable, a costa de un tiempo de ejecución considerablemente mayor. El hiperparámetro con más impacto esperado es `min_samples_leaf`, porque exigir más clientes por hoja reduce el sobreajuste. Los valores finales deben actualizarse después de ejecutar la grilla reducida.

**Umbral de decisión:** se define como una política de negocio dentro del flujo de validación cruzada sobre train. Se explicitan el costo relativo de un falso negativo (`4`) y de un falso positivo (`1`), además de un recall mínimo (`0,65`) y una precisión mínima (`0,20`). Entre los umbrales que cumplen esas restricciones se elige el de menor costo esperado. Estos valores son supuestos iniciales y deben reemplazarse por costos reales.

**Calibración de probabilidades:** como el Random Forest usa `class_weight="balanced"`, se comparan `sigmoid` e `isotonic` con predicciones out-of-fold sobre train. El método se selecciona por menor Brier score y se visualiza con curvas de calibración. El test no interviene en la selección.

**Robustez temporal:** el dataset no contiene una fecha de observación. El notebook incorpora un análisis de sensibilidad por bloques ordenados (primer 80% para entrenar y último 20% para evaluar), pero se reporta explícitamente como proxy y no como validación temporal real. Para una conclusión definitiva se necesita una fecha y una división cronológica, idealmente con ventanas rolling.

**Resultados en test (usado una única vez):**

| Métrica | Valor |
|---|---|
| ROC-AUC | Se actualiza al ejecutar la evaluación final |
| PR-AUC | Se actualiza al ejecutar la evaluación final |
| Recall | Se actualiza al ejecutar la evaluación final |
| Precisión | Se actualiza al ejecutar la evaluación final |
| F2 | Se actualiza al ejecutar la evaluación final |

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

### Sensibilidad a valores extremos

Para comprobar que los segmentos no estén definidos solamente por valores anómalos, el notebook 03 incorpora una comparación sobre los mismos clientes:

- **Sin tratamiento:** variables crudas, imputación necesaria para ejecutar KMeans y estandarización, sin `log1p` ni reglas para outliers.
- **Tratado:** reglas de limpieza del notebook 01, imputación, transformaciones `log1p` y estandarización.

La comparación informa las métricas internas de clustering (silhouette, Davies-Bouldin y Calinski-Harabasz), los tamaños de los grupos, la participación, la tasa de mora y el porcentaje de clientes cuya asignación cambia. Como los números de cluster son etiquetas arbitrarias, las asignaciones se alinean mediante Hungarian matching antes de calcular ese porcentaje. También se muestran los centroides en unidades originales para identificar qué variables forman cada perfil.

En la ejecución actual de `k = 4`, la solución sin tratamiento obtiene silhouette **0,209**, Davies-Bouldin **1,326** y Calinski-Harabasz **1780,912**. La solución tratada obtiene silhouette **0,182**, Davies-Bouldin **1,536** y Calinski-Harabasz **1929,452**. No hay una mejora uniforme; estos valores no son estrictamente comparables porque cada solución usa un espacio transformado distinto.

El tratamiento reasigna **60,446%** de los clientes. Los clusters tratados quedan con **33.512 (22,38%)**, **51.393 (34,32%)**, **56.995 (38,06%)** y **7.831 (5,23%)** clientes, con tasas de mora de **7,67%**, **1,64%**, **5,62%** y **41,72%**, respectivamente. Frente a los clusters alineados sin tratamiento, la mora cambia **-0,14**, **-1,44**, **-0,51** y **-13,45 puntos porcentuales**.

Los centroides muestran que los perfiles tratados se forman principalmente por utilización del crédito, edad, cantidad de productos e historial de atrasos. El cluster de mayor mora combina utilización **0,78** y **1,50 atrasos de 90+ días**; el de menor mora corresponde a clientes mayores (edad **65,11**) con utilización **0,12**; el perfil joven presenta utilización **0,53**; y el perfil de mayores ingresos concentra **10,27 líneas de crédito**. El tratamiento cambia fuertemente las fronteras, pero los clusters no se explican únicamente por los outliers.

También se evalúa la estabilidad de la solución tratada repitiendo KMeans con cinco semillas y calculando el **Adjusted Rand Index (ARI)** para todos los pares de corridas. El ARI promedio es **0,974** y el mínimo **0,942**, lo que indica alta estabilidad frente a la inicialización, aunque no demuestra por sí solo que los grupos sean naturales ni que sean la única segmentación de negocio válida. Los perfiles se complementan con mediana, percentil 95 y percentil 99 para evitar que los promedios oculten colas largas en `RevolvingUtilizationOfUnsecuredLines`, `DebtRatio` o `MonthlyIncome`.

Detalle completo en [`notebooks/03_modelo_no_supervisado.ipynb`](notebooks/03_modelo_no_supervisado.ipynb).

## Cambios respecto de la primera entrega

A partir de la devolución recibida:

1. **Umbral de decisión basado en costos:** antes se elegía probando valores sobre el test; ahora se elige dentro de los folds internos de una validación cruzada anidada, minimizando el costo esperado sujeto a mínimos de recall y precisión. El test se usa una sola vez.
2. **Imputación antes del split:** antes la mediana se calculaba con todo el dataset; ahora se calcula dentro de un `Pipeline`, solo con train.
3. **KMeans y valores extremos:** en la versión anterior, el cluster de mayor riesgo (269 clientes) eran exactamente los registros con código 98, y los perfiles mostraban promedios imposibles (13 en `RevolvingUtilizationOfUnsecuredLines`, 674 en `DebtRatio`). Al tratar los códigos aparte, corregir los ratios y aplicar logaritmo, aparece un segmento de riesgo de 7.831 clientes definido por su comportamiento real. La nueva sección de sensibilidad cuantifica que se reasigna el 60,446% de los clientes, documenta los cambios de tamaño y mora por cluster y muestra que la solución tratada es estable (ARI promedio 0,974; mínimo 0,942).
4. **Notebooks ejecutados de principio a fin** y carpetas `.ipynb_checkpoints` excluidas con `.gitignore`.
5. **Calibración de probabilidades:** se comparan calibración sigmoidal e isotónica con Brier score y curvas de calibración usando únicamente train; el método ganador se evalúa después en el test reservado.
6. **Robustez temporal:** se agrega una evaluación por bloques ordenados como proxy, junto con la limitación explícita de que el dataset no tiene una variable temporal real.

La limpieza también mejoró a la Regresión Logística, que pasó de 0,824 a 0,862 de ROC-AUC en test.

## Limitaciones y próximos pasos

- La Regresión Logística no se ajustó y quedó cerca del Random Forest (0,855 contra 0,862 en validación cruzada).
- La calibración se selecciona con train; sus valores de Brier y el método final deben actualizarse ejecutando el notebook.
- Los costos y mínimos de recall/precisión son supuestos iniciales y deben validarse con el área de negocio.
- El proxy temporal no reemplaza una validación cronológica: el dataset no tiene fecha. El próximo paso es conseguir datos con período de observación y evaluar drift con ventanas temporales.
- Próximos pasos: probar Gradient Boosting y usar importancia por permutación.

## Estructura del repositorio

```
dataset/        Dataset original (Kaggle) y versión limpia generada en el AED
notebooks/      01_analisis_exploratorio.ipynb, 02_modelo_supervisado.ipynb, 03_modelo_no_supervisado.ipynb
```

Para reproducir los resultados, ejecutar los notebooks en orden: el 01 genera `dataset/cs-training-clean.csv`, que leen el 02 y el 03.