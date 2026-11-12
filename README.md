# Proyecto Final - Data Science YPF

Proyecto final del curso de Data Science de la Fundación YPF (EnergIA Digital), aplicando de punta a punta el flujo de trabajo aprendido: AED, limpieza de datos, modelo supervisado y modelo no supervisado.

## Integrantes

- Emilia Fernández
- Noelia Navarro
- Oscar Sánchez
- Rodrigo Villablanca

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
5. **Conexión con el sector energético:** se desarrolla en la sección siguiente, con casos de YPF y fuentes.

La limpieza también mejoró a la Regresión Logística, que pasó de 0,824 a 0,862 de ROC-AUC en test.

## Conexión con el sector energético

El dataset es de crédito, pero el problema de fondo, anticipar qué clientes no van a pagar, es habitual en las empresas de energía. Lo que se traslada es el método, no los datos: con datos del sector habría que reentrenar los modelos.

- **YPF ya mide este riesgo.** En sus estados financieros al 31/12/2025, YPF estima la pérdida crediticia esperada con el enfoque de la NIIF 9, mediante una matriz por tramos de días de atraso. Este proyecto coincide en que el atraso es la señal principal (las tres variables de atraso suman el 46% de la importancia) y muestra que, combinado con la utilización del crédito y otras variables, permite estimar el riesgo cliente por cliente y no solo por tramo.
- **YPF Agro es el caso más cercano.** Financia a productores la compra de combustibles e insumos y acepta el pago con granos (canje), una modalidad que la compañía decidió mantener en 2026. Son muchos clientes con crédito a plazo, la misma estructura que *Give Me Some Credit*: un scoring sirve para fijar límites y condiciones, y la segmentación para diferenciar la estrategia comercial.
- **La cobrabilidad impacta en el resultado.** En 2024, parte de la deuda de CAMMESA con las generadoras se canceló con bonos AE38, con una quita importante. En la venta de Metrogas (más de 2,4 millones de usuarios), el valor de la distribuidora depende en parte de cuánto de su cartera se cobra efectivamente.
- **Crédito para energías renovables.** Financiar paneles solares, termotanques solares o equipos eólicos domésticos es un crédito con inversión inicial alta que se repaga con el ahorro en la factura. Con las variables del dataset:
  - `MonthlyIncome` y `DebtRatio` permiten verificar que el cliente soporte la cuota sin sobreendeudarse. En este modelo pesan poco (2,4% y 4,1% de la importancia), por lo que funcionan mejor como control de capacidad de pago que como predictores.
  - El historial de atrasos permite reservar una tasa preferencial ("crédito verde") para el tramo de menor riesgo **estimado por el modelo**. `SeriousDlqin2yrs` no sirve para esto: es lo que ocurre después del otorgamiento.
  - `NumberRealEstateLoansOrLines` es un indicio de propiedad, no una prueba: un propietario sin deuda tiene 0. En un caso real habría que confirmar la titularidad del inmueble, necesaria para instalar.
  - Contexto: la Ley 27.424 creó el régimen de generación distribuida (2.520 usuarios-generadores a febrero de 2025, según la Secretaría de Energía); en 2024, el Banco Nación y la Secretaría de Energía lanzaron líneas en cuotas para paneles y termotanques solares; y en octubre de 2026 YPF aprobó tomar el 100% de YPF Luz.
- **Otros usos del mismo método:** mantenimiento predictivo (una falla de una bomba o un compresor es un evento raro, como la mora: mismo desbalance, misma importancia del recall y del umbral) y segmentación de clientes de estaciones de servicio por patrón de consumo.

**Límites de la traslación:** con datos reales del sector habría que incorporar variables propias (consumo, estacionalidad, actualizaciones tarifarias, inflación) y validar en forma temporal, porque en Argentina el contexto cambia el comportamiento de pago de un año a otro.

### Fuentes

| Dato | Fuente |
|---|---|
| YPF estima la pérdida crediticia esperada con NIIF 9 y una matriz por tramos de atraso | [Estados financieros de YPF S.A. al 31/12/2025](https://inversores.ypf.com/documents/asamblea-accionistas/2026/Asamblea30-04-2026-EEFFdeYPFSAal31-12-2025.pdf) |
| YPF Agro financia a productores y mantiene el canje de granos en 2026 | [La Nación, 05/05/2026](https://www.lanacion.com.ar/economia/campo/ypf-suspendio-la-busqueda-de-un-socio-para-su-negocio-agro-y-ahora-lo-relanzaria-nid05052026/) · [LM Neuquén](https://masp.lmneuquen.com/agricultura/ypf-dio-marcha-atras-la-venta-agro-y-apuesta-una-transformacion-total-del-negocio-n1239843) |
| Pago de la deuda de CAMMESA con bonos AE38 (2024) | [Infobae, 10/05/2024](https://www.infobae.com/economia/2024/05/10/algunas-empresas-de-energia-aceptaron-la-propuesta-de-caputo-para-el-pago-de-la-deuda-de-cammesa/) |
| Venta de Metrogas a Edenor (US$780 millones, sujeta a condiciones de cierre) | [Hecho relevante de YPF ante la SEC (6-K), 10/08/2026](https://www.sec.gov/Archives/edgar/data/0000904851/000155485526001764/MainDocument.htm) |
| Usuarios-generadores de la Ley 27.424 (2.520 a febrero de 2025) | [Secretaría de Energía, reporte de avance de febrero de 2025](https://argentina.gob.ar/sites/default/files/ppt_reporte_de_avance_febrero_2025.pdf) |
| Líneas del Banco Nación para paneles y termotanques solares (2024) | [Ámbito](https://www.ambito.com/economia/banco-nacion-lanza-creditos-comprar-electrodomesticos-tasa-del-43-y-plazo-5-anos-n6045754/amp) · [La Nación, 08/08/2024](https://www.lanacion.com.ar/economia/televisores-heladeras-y-lavarropas-como-es-el-programa-oficial-de-creditos-con-cuotas-a-cinco-anos-nid08082024/) |
| YPF toma el 100% de YPF Luz (octubre de 2026) | [Forbes Argentina](https://www.forbesargentina.com/negocios/ypf-pago-us-430-millones-25-ypf-luz-ya-controla-total-compania-n99853) |
| Importancia de variables del modelo (46% atrasos, 2,4% ingreso, 4,1% `DebtRatio`, 1,9% inmuebles) | [`notebooks/02_modelo_supervisado.ipynb`](notebooks/02_modelo_supervisado.ipynb), sección 7 |

## Presentación final

La presentación *¿Quién no va a pagar?* expone el proyecto en 11 bloques:

1. **Introducción:** el problema y el dataset.
2. **Objetivo:** predecir la mora, segmentar sin usar la etiqueta y proponer qué hacer con cada perfil.
3. **Hipótesis:** H1, el atraso y la utilización del crédito predicen la mora; H2, la segmentación separa el riesgo.
4. **Metodología:** el test se usa una sola vez y las decisiones se toman con validación cruzada.
5. **Limpieza de datos:** valores anómalos tratados con reglas fijas, sin eliminar filas.
6. **Preparación para el análisis:** división estratificada, métrica F2 y `class_weight="balanced"`.
7. **Random Forest:** comparación de modelos y ajuste con `GridSearchCV`.
8. **Predicción vs. realidad:** matriz de confusión con el umbral 0,56.
9. **KMeans:** cuatro segmentos con mora de 1,6% a 41,7%.
10. **Conclusiones:** la calidad de los datos pesó más que el algoritmo.
11. **Propuesta y sector energético:** riesgo comercial (YPF Agro y crédito verde para renovables), mantenimiento predictivo y segmentación de clientes de estaciones, con la NIIF 9 como base.

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

```
pip install -r requirements.txt
```

Para reproducir los resultados, ejecutar los notebooks en orden: el 01 genera `dataset/cs-training-clean.csv`, que leen el 02 y el 03. Los resultados se obtuvieron con Python 3.13.9. Los notebooks 01 y 03 son estables entre versiones de las librerías; el 02 puede variar levemente con otra versión de scikit-learn (con la 1.9.1, el umbral elegido es 0,57 y la ROC-AUC en test, 0,867).