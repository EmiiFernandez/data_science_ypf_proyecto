# Diferencias entre la versión previa y la versión actual del notebook

## Notebook no supervisado: clustering de clientes

La versión previa del notebook de clustering se apoyaba sobre todo en la inercia y en la visualización del método del codo para elegir `k`. Eso es útil, pero no es suficiente para validar que la segmentación sea realmente buena.

### Cambios aplicados

- Se agregaron métricas de calidad de clustering: `silhouette`, `davies_bouldin` y `calinski_harabasz`.
- Se mejoró la comparación entre distintos valores de `k` con una tabla resumen, no solo con gráficos.
- Se reforzó la decisión de `k` con criterios cuantitativos y no solo visuales.
- Se dejó un perfil más accionable por cluster, incluyendo cantidad de clientes, participación y tasa de mora.

### Diferencia clave

La versión previa evaluaba principalmente la forma del codo. La versión actual valida la calidad de la segmentación con métricas que ayudan a distinguir si los grupos son compactos, bien separados y interpretables.

Esto hace que la elección de `k=4` quede más fundamentada y más reproducible para una presentación o un análisis más serio.
