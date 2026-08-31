# StreamView Analytics — EP1
## Visual Analytics para la Toma de Decisiones Estratégicas
**ADY1104 — Visualización de Datos | Duoc UC 2026**

---

## Estructura del proyecto

```
streamview/
├── data/
│   ├── netflix_movies_detailed_up_to_2025.csv   (16.000 registros, 18 variables)
│   └── netflix_tv_shows_detailed_up_to_2025.csv (16.000 registros, 16 variables)
├── notebooks/
│   └── EP1_visualizaciones.py   ← script principal
├── images/                      ← visualizaciones estáticas (PNG)
├── dashboard/                   ← visualizaciones interactivas (HTML)
└── README.md
```

## Cómo ejecutar

```bash
pip install pandas matplotlib seaborn plotly kaleido
python notebooks/EP1_visualizaciones.py
```

## Visualizaciones generadas

| Archivo | Descripción |
|---|---|
| viz1_kpis | KPIs generales del catálogo |
| viz2_tipo | Distribución películas vs. series |
| viz3_generos_volumen | Top 15 géneros por volumen |
| viz4_temporal | Evolución temporal del catálogo |
| viz5_paises | Top 15 países productores |
| viz6_idiomas | Distribución por idioma |
| viz7_popularidad_genero | Popularidad promedio por género |
| viz8_scatter_pop_val | Scatter popularidad vs. valoración |
| viz9_clasificacion_etaria | Distribución por clasificación etaria |
| viz10_top_popularidad | Top 10 contenidos más populares |

## Reglas de negocio aplicadas

- `budget` y `revenue` se usan exclusivamente para películas
- `popularity` es un índice relativo TMDB, no representa reproducciones
- `vote_average` se interpreta en escala 0–10
- Contenidos con `budget = 0` o `revenue = 0` se excluyen de análisis financieros
- `duration` en Movies no disponible (columna vacía en el dataset)
- Dataset muestreado: 1.000 registros por año (2010–2025) por categoría
