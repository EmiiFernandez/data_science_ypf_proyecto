# Guía paso a paso para mejorar el modelo supervisado

Este documento recoge la mejora recomendada para el notebook de modelado supervisado y está pensado para aplicarse de forma ordenada en `notebooks/02_modelo_supervisado.ipynb` sin perder la lógica original del proyecto.

---

## Objetivo

Mejorar la utilidad práctica del modelo de riesgo crediticio, no solo el ROC-AUC. En problemas desbalanceados como este, el punto más importante suele ser:

- detectar mejor a la clase positiva (`SeriousDlqin2yrs = 1`),
- evaluar con métricas reales para desbalance,
- ajustar el umbral de decisión,
- comparar modelos más fuertes como boosting,
- y decidir el modelo final según el negocio.

---

## 1. Mantener el baseline actual

Antes de hacer cambios, la base actual debe seguir existiendo:

- carga del dataset limpio,
- train/test split con estratificación,
- DummyClassifier como piso mínimo,
- comparación de modelos base,
- Random Forest con GridSearchCV.

Eso ya está bien. Lo que falta es agregar una segunda capa de evaluación y decisión basada en el negocio.

---

## 2. Agregar importaciones nuevas

Reemplaza la celda de imports del notebook con esta versión:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
    average_precision_score,
    precision_recall_curve,
)

sns.set_theme(style="whitegrid")
```

### Por qué
Estas métricas permiten evaluar no solo el ranking del modelo, sino también la capacidad real de detectar morosos en un problema desbalanceado.

---

## 3. Reafirmar la división train/test

La celda actual puede quedar igual, pero conviene dejarla explícita y robusta:

```python
df = pd.read_csv("../dataset/cs-training-clean.csv")
X = df.drop(columns="SeriousDlqin2yrs")
y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Proporción clase positiva train:", y_train.mean().round(4))
print("Proporción clase positiva test:", y_test.mean().round(4))
```

---

## 4. Agregar una función de evaluación con métricas de desbalance

Coloca esta función justo después de la celda de resultados base:

```python
resultados = {}

def evaluar_modelo(nombre, modelo, X_tr, X_te, umbral=0.5):
    modelo.fit(X_tr, y_train)
    y_pred = modelo.predict(X_te)
    y_proba = modelo.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, (y_proba >= umbral).astype(int), zero_division=0)
    recall = recall_score(y_test, (y_proba >= umbral).astype(int), zero_division=0)
    f2 = fbeta_score(y_test, (y_proba >= umbral).astype(int), beta=2, zero_division=0)
    f1 = f1_score(y_test, y_pred)

    resultados[nombre] = {
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "F2": f2,
        "Umbral": umbral,
    }

    print(f"--- {nombre} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Precision (umbral {umbral}): {precision:.4f}")
    print(f"Recall (umbral {umbral}): {recall:.4f}")
    print(f"F2 (umbral {umbral}): {f2:.4f}")
    print()
    return modelo
```

### Uso
Luego puedes seguir usando:

```python
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
])
log_reg = evaluar_modelo("Regresion Logistica", log_reg, X_train, X_test, umbral=0.5)
```

---

## 5. Agregar búsqueda del mejor umbral

Este es el paso más importante para la mejora práctica del modelo. Inserta esta función antes de comparar modelos finales:

```python
def buscar_mejor_umbral(y_true, y_proba, objetivo="f2"):
    thresholds = np.linspace(0.1, 0.9, 81)
    resultados_umbral = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        if objetivo == "f2":
            score = f2
        elif objetivo == "recall":
            score = recall
        elif objetivo == "precision":
            score = precision
        else:
            score = f2

        resultados_umbral.append({
            "threshold": t,
            "precision": precision,
            "recall": recall,
            "f2": f2,
            "score": score,
        })

    df_umbral = pd.DataFrame(resultados_umbral)
    return df_umbral.sort_values("score", ascending=False).head(10)
```

### Cómo usarla

```python
# Modelo final ya entrenado
# por ejemplo: mejor_modelo
# y_proba = mejor_modelo.predict_proba(X_test)[:, 1]

mejores_umbrales = buscar_mejor_umbral(y_test, y_proba, objetivo="f2")
print(mejores_umbrales.head(10))
```

### Qué se espera
Te dirá qué umbral maximiza F2, que suele ser útil para detectar casos de riesgo con buen equilibrio.

---

## 6. Añadir validación cruzada por PR-AUC

Esto complementa la validación que ya hace GridSearchCV. Inserta esta sección después del ajuste de hiperparámetros del Random Forest.

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

scores_pr = cross_val_score(
    rf_cv,
    X_train,
    y_train,
    cv=cv,
    scoring="average_precision",
    n_jobs=-1,
)

print("PR-AUC CV mean:", round(scores_pr.mean(), 4))
print("PR-AUC CV std:", round(scores_pr.std(), 4))
```

### Importante
Si quieres usar `cross_val_score`, debes importar también `cross_val_score` desde `sklearn.model_selection`.

---

## 7. Comparar con un modelo boosting

Inserta esta celda después del Random Forest base y antes del ajuste final:

```python
hgb = HistGradientBoostingClassifier(
    max_depth=8,
    learning_rate=0.05,
    max_iter=300,
    random_state=42,
)

hgb.fit(X_train, y_train)
y_proba_hgb = hgb.predict_proba(X_test)[:, 1]

auc_hgb = roc_auc_score(y_test, y_proba_hgb)
pr_auc_hgb = average_precision_score(y_test, y_proba_hgb)

print("HistGradientBoostingClassifier")
print(f"ROC-AUC: {auc_hgb:.4f}")
print(f"PR-AUC: {pr_auc_hgb:.4f}")
```

### Si XGBoost está instalado

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=13,
    random_state=42,
)

xgb.fit(X_train, y_train)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

print("XGBClassifier")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba_xgb):.4f}")
```

### Recomendación
No reemplazar el Random Forest por fuerza; comparar ambos y elegir según recall/precision bajo el umbral definitivo.

---

## 8. Añadir calibración de probabilidades

Agrega esta sección después de entrenar el mejor modelo final:

```python
from sklearn.calibration import CalibratedClassifierCV

rf_calibrado = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

calibrated = CalibratedClassifierCV(rf_calibrado, method="sigmoid", cv=3)
calibrated.fit(X_train, y_train)

y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
print("ROC-AUC calibrado:", round(roc_auc_score(y_test, y_proba_cal), 4))
print("PR-AUC calibrado:", round(average_precision_score(y_test, y_proba_cal), 4))
```

### Cuándo usarlo
En sistemas reales de scoring, la probabilidad debe ser interpretable y consistente. Si el ranking es bueno pero la probabilidad no está bien calibrada, esto ayuda.

---

## 9. Diseñar la comparación final en una tabla

Reemplaza la sección final del notebook por una tabla más útil:

```python
resultados_df = pd.DataFrame(resultados).T.sort_values("AUC", ascending=False)
print(resultados_df.round(4))
```

Y si quieres ver un ranking por objetivo real:

```python
# Elegir la métrica que más te importa según el negocio
# Ejemplo: F2
resultado_final = resultados_df.sort_values("F2", ascending=False)
print(resultado_final.round(4))
```

### Consejo práctico
Si el objetivo es detectar morosos, la métrica final no debe ser solo ROC-AUC; debe ser la que mejor refleje la capacidad de detectar la clase positiva.

---

## 10. Decisión final recomendada para el notebook

La mejor práctica para este proyecto es esta:

1. Mantener Random Forest como baseline fuerte.
2. Añadir PR-AUC, Precision, Recall y F2.
3. Buscar el mejor umbral sobre validación.
4. Comparar con HistGradientBoosting o XGBoost.
5. Elegir el modelo por negocio y no solo por ROC-AUC.
6. Guardar el modelo final junto con el umbral seleccionado.

El criterio sugerido es:

- maximizar recall de la clase positiva,
- con precisión razonable,
- y mantener una buena métrica F2.

---

## 11. Bloque de resumen para pegar al final

Este bloque resume la interpretación del proyecto:

```python
print("=== Resumen de mejora del modelo ===")
print("1. El modelo base ya funciona y supera al Dummy.")
print("2. ROC-AUC no es suficiente por sí solo en un problema desbalanceado.")
print("3. La optimización del umbral suele mejorar la utilidad práctica del modelo.")
print("4. PR-AUC y F2 son métricas más orientadas a la detección de la clase positiva.")
print("5. Modelos boosting como HistGradientBoosting o XGBoost pueden competir o superar al Random Forest.")
print("6. La decisión final debe basarse en recall/precision y en el costo de error del negocio.")
```

---

## 12. Recomendación final de implementación

Para aplicar esto en el notebook:

1. agrega primero las importaciones nuevas,
2. luego la función `evaluar_modelo` ampliada,
3. después `buscar_mejor_umbral`,
4. añade la comparación con boosting,
5. y al final, elige el modelo según F2 o recall a un umbral de negocio.

Esto cambia la calidad del proyecto de “modelos comparados” a “modelo listo para decisión real de riesgo crediticio”.

---

## 13. Resultado esperado

Al final, el notebook debería permitir:

- comparar modelos con AUC y PR-AUC,
- revisar precision/recall/F2 para cada umbral,
- identificar el corte que mejor responde al caso de uso,
- y elegir un modelo que no solo “logre buen AUC”, sino que realmente detecte a los clientes en riesgo.

Este es el siguiente paso correcto para volcar el proyecto de una versión académica a una versión más aplicada y de negocio.
