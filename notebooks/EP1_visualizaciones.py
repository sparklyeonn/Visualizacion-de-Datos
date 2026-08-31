"""
=============================================================================
STREAMVIEW ANALYTICS — EP1
Visual Analytics para la Toma de Decisiones Estratégicas
ADY1104 — Visualización de Datos | Duoc UC 2026
=============================================================================
Genera:
  - Imágenes estáticas (Seaborn/Matplotlib) → carpeta images/
  - Gráficos interactivos (Plotly)          → carpeta dashboard/
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
IMG_DIR    = os.path.join(BASE_DIR, "images")
DASH_DIR   = os.path.join(BASE_DIR, "dashboard")

os.makedirs(IMG_DIR,  exist_ok=True)
os.makedirs(DASH_DIR, exist_ok=True)

# Paleta corporativa StreamView
C_MOVIE    = "#E50914"   # rojo (películas)
C_SHOW     = "#1A73E8"   # azul (series)
C_ACCENT   = "#F5A623"   # naranja acento
C_BG       = "#F8F9FA"   # fondo claro
C_DARK     = "#1C1C1C"   # texto oscuro
C_GRAY     = "#6C757D"   # gris secundario
C_PALETTE  = [C_MOVIE, C_SHOW, C_ACCENT, "#2ECC71", "#9B59B6",
              "#1ABC9C", "#E67E22", "#3498DB", "#E74C3C", "#95A5A6"]

sns.set_theme(style="whitegrid", palette=C_PALETTE)
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.facecolor":   C_BG,
    "figure.facecolor": "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

NOTA_POPULARITY = "⚠ El índice de popularidad es relativo (TMDB) y no representa cantidad de reproducciones."
NOTA_MUESTRA    = "ℹ Los datos presentan 1.000 registros por año (2010–2025), lo que refleja un muestreo balanceado del catálogo."


# ─────────────────────────────────────────────
# 1. CARGA Y LIMPIEZA DE DATOS
# ─────────────────────────────────────────────
def cargar_datos():
    movies = pd.read_csv(os.path.join(DATA_DIR, "netflix_movies_detailed_up_to_2025.csv"))
    shows  = pd.read_csv(os.path.join(DATA_DIR, "netflix_tv_shows_detailed_up_to_2025.csv"))

    # ── Películas ──────────────────────────────
    movies["date_added"]  = pd.to_datetime(movies["date_added"], errors="coerce")
    movies["year_added"]  = movies["date_added"].dt.year
    movies["type_label"]  = "Película"
    # duration está completamente vacía → se descarta
    movies = movies.drop(columns=["duration"], errors="ignore")
    # budget/revenue: solo válidos cuando > 0
    movies["budget_valid"]  = movies["budget"].where(movies["budget"]  > 0)
    movies["revenue_valid"] = movies["revenue"].where(movies["revenue"] > 0)

    # ── Series ────────────────────────────────
    shows["date_added"] = pd.to_datetime(shows["date_added"], errors="coerce")
    shows["year_added"] = shows["date_added"].dt.year
    shows["type_label"] = "Serie"
    # director tiene 68 % nulos → se documenta y no se usa en análisis agregado

    # ── Dataset combinado (solo variables comunes) ──
    common_cols = ["show_id","type","type_label","title","country","date_added",
                   "year_added","release_year","rating","genres","language",
                   "popularity","vote_count","vote_average"]
    combined = pd.concat([movies[common_cols], shows[common_cols]], ignore_index=True)

    return movies, shows, combined

movies, shows, combined = cargar_datos()
print(f"✓ Datos cargados — Películas: {len(movies):,} | Series: {len(shows):,} | Total: {len(combined):,}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def guardar(fig, nombre, dpi=150):
    path = os.path.join(IMG_DIR, f"{nombre}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Guardado: images/{nombre}.png")

def guardar_plotly(fig, nombre):
    path = os.path.join(DASH_DIR, f"{nombre}.html")
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  ✓ Guardado: dashboard/{nombre}.html")

def nota_fig(ax, texto, y=-0.13, fontsize=8):
    ax.annotate(texto, xy=(0, y), xycoords="axes fraction",
                fontsize=fontsize, color=C_GRAY, style="italic",
                wrap=True)


# ═════════════════════════════════════════════
# VIZ 1 — KPIs GENERALES (Seaborn/Matplotlib)
# ═════════════════════════════════════════════
def viz1_kpis():
    print("\n[VIZ 1] KPIs Generales del Catálogo")

    total        = len(combined)
    total_movies = len(movies)
    total_shows  = len(shows)
    n_paises     = combined["country"].dropna().str.split(", ").explode().nunique()
    n_idiomas    = combined["language"].nunique()
    n_generos    = combined["genres"].dropna().str.split(", ").explode().nunique()

    kpis = [
        ("Total\nContenidos",   f"{total:,}",        C_DARK),
        ("Películas",           f"{total_movies:,}",  C_MOVIE),
        ("Series",              f"{total_shows:,}",   C_SHOW),
        ("Países\nRepresentados", f"{n_paises}",      C_ACCENT),
        ("Idiomas\nDisponibles",  f"{n_idiomas}",     "#2ECC71"),
        ("Géneros",             f"{n_generos}",        "#9B59B6"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    fig.suptitle("El catálogo de StreamView Analytics en cifras",
                 fontsize=15, fontweight="bold", y=1.05, color=C_DARK)

    for ax, (label, valor, color) in zip(axes, kpis):
        ax.set_facecolor(color + "18")
        ax.text(0.5, 0.58, valor, ha="center", va="center",
                fontsize=26, fontweight="bold", color=color,
                transform=ax.transAxes)
        ax.text(0.5, 0.20, label, ha="center", va="center",
                fontsize=10, color=C_GRAY, transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    guardar(fig, "viz1_kpis")

viz1_kpis()


# ═════════════════════════════════════════════
# VIZ 2 — DISTRIBUCIÓN POR TIPO (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz2_tipo():
    print("\n[VIZ 2] Distribución por Tipo de Contenido")

    counts = combined["type_label"].value_counts().reset_index()
    counts.columns = ["Tipo", "Cantidad"]
    counts["Porcentaje"] = (counts["Cantidad"] / counts["Cantidad"].sum() * 100).round(1)
    counts["Color"] = counts["Tipo"].map({"Película": C_MOVIE, "Serie": C_SHOW})

    # ── Seaborn ──
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(counts["Tipo"], counts["Cantidad"],
                   color=counts["Color"].values, height=0.5, edgecolor="white")
    for bar, (_, row) in zip(bars, counts.iterrows()):
        ax.text(bar.get_width() + 150, bar.get_y() + bar.get_height()/2,
                f"{row['Cantidad']:,}  ({row['Porcentaje']}%)",
                va="center", fontsize=11, fontweight="bold", color=C_DARK)
    ax.set_xlim(0, counts["Cantidad"].max() * 1.18)
    ax.set_xlabel("Cantidad de títulos")
    ax.set_title("Distribución del catálogo: Películas vs. Series", pad=12)
    ax.set_facecolor(C_BG)
    nota_fig(ax, NOTA_MUESTRA)
    plt.tight_layout()
    guardar(fig, "viz2_tipo")

    # ── Plotly ──
    fig_p = px.bar(counts, x="Cantidad", y="Tipo", orientation="h",
                   color="Tipo", color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
                   text=counts.apply(lambda r: f"{r['Cantidad']:,} ({r['Porcentaje']}%)", axis=1),
                   title="Distribución del catálogo: Películas vs. Series")
    fig_p.update_traces(textposition="outside")
    fig_p.update_layout(showlegend=False, plot_bgcolor=C_BG,
                        xaxis_title="Cantidad de títulos", yaxis_title="")
    guardar_plotly(fig_p, "viz2_tipo")

viz2_tipo()


# ═════════════════════════════════════════════
# VIZ 3 — TOP 15 GÉNEROS POR VOLUMEN (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz3_generos_volumen():
    print("\n[VIZ 3] Top 15 Géneros por Volumen")

    def top_generos(df, label):
        g = df["genres"].dropna().str.split(", ").explode()
        top = g.value_counts().head(15).reset_index()
        top.columns = ["Género", "Cantidad"]
        top["Tipo"] = label
        return top

    top_m = top_generos(movies, "Películas")
    top_s = top_generos(shows,  "Series")

    # ── Seaborn — dos paneles ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, data, color, titulo in [
        (ax1, top_m, C_MOVIE, "Películas — Top 15 géneros"),
        (ax2, top_s, C_SHOW,  "Series — Top 15 géneros"),
    ]:
        data_s = data.sort_values("Cantidad")
        bars = ax.barh(data_s["Género"], data_s["Cantidad"],
                       color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height()/2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=9, color=C_DARK)
        ax.set_title(titulo, pad=10)
        ax.set_xlabel("Cantidad de contenidos")
        ax.set_facecolor(C_BG)

    fig.suptitle("Géneros predominantes en el catálogo de StreamView Analytics",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    guardar(fig, "viz3_generos_volumen")

    # ── Plotly — combinado con selector ──
    combined_g = pd.concat([top_m, top_s])
    fig_p = px.bar(combined_g, x="Cantidad", y="Género", color="Tipo",
                   facet_col="Tipo", orientation="h",
                   color_discrete_map={"Películas": C_MOVIE, "Series": C_SHOW},
                   title="Géneros predominantes — Películas vs. Series (Top 15)")
    fig_p.update_layout(plot_bgcolor=C_BG, showlegend=False)
    guardar_plotly(fig_p, "viz3_generos_volumen")

viz3_generos_volumen()


# ═════════════════════════════════════════════
# VIZ 4 — EVOLUCIÓN TEMPORAL (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz4_temporal():
    print("\n[VIZ 4] Evolución Temporal del Catálogo")

    evol = combined.groupby(["year_added", "type_label"]).size().reset_index(name="Cantidad")

    # ── Seaborn ──
    fig, ax = plt.subplots(figsize=(11, 5))
    for tipo, color, marker in [("Película", C_MOVIE, "o"), ("Serie", C_SHOW, "s")]:
        d = evol[evol["type_label"] == tipo]
        ax.plot(d["year_added"], d["Cantidad"], color=color,
                linewidth=2.5, marker=marker, markersize=6, label=tipo)
        for _, row in d.iterrows():
            ax.annotate(f"{int(row['Cantidad'])}", (row["year_added"], row["Cantidad"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=color)

    ax.set_title("Crecimiento del catálogo por año de incorporación", pad=12)
    ax.set_xlabel("Año de incorporación a la plataforma")
    ax.set_ylabel("Cantidad de contenidos incorporados")
    ax.legend(title="Tipo de contenido")
    ax.set_facecolor(C_BG)
    nota_fig(ax, NOTA_MUESTRA)
    plt.tight_layout()
    guardar(fig, "viz4_temporal")

    # ── Plotly ──
    fig_p = px.line(evol, x="year_added", y="Cantidad", color="type_label",
                    markers=True,
                    color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
                    labels={"year_added": "Año de incorporación",
                            "Cantidad": "Contenidos incorporados",
                            "type_label": "Tipo"},
                    title="Crecimiento del catálogo por año de incorporación")
    fig_p.update_layout(plot_bgcolor=C_BG)
    guardar_plotly(fig_p, "viz4_temporal")

viz4_temporal()


# ═════════════════════════════════════════════
# VIZ 5 — TOP 15 PAÍSES PRODUCTORES (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz5_paises():
    print("\n[VIZ 5] Top 15 Países Productores")

    paises = combined["country"].dropna().str.split(", ").explode()
    top_p = paises.value_counts().head(15).reset_index()
    top_p.columns = ["País", "Cantidad"]
    top_p_s = top_p.sort_values("Cantidad")

    # ── Seaborn ──
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [C_MOVIE if i == len(top_p_s)-1 else C_SHOW + "BB"
              for i in range(len(top_p_s))]
    bars = ax.barh(top_p_s["País"], top_p_s["Cantidad"],
                   color=colors, edgecolor="white")
    for bar in bars:
        ax.text(bar.get_width() + 40, bar.get_y() + bar.get_height()/2,
                f"{int(bar.get_width()):,}", va="center", fontsize=9, color=C_DARK)
    ax.set_title("Países con mayor producción en el catálogo — Top 15", pad=12)
    ax.set_xlabel("Cantidad de títulos")
    ax.set_facecolor(C_BG)
    nota_fig(ax, "ℹ Un contenido con coproducción se contabiliza en cada país participante.")
    plt.tight_layout()
    guardar(fig, "viz5_paises")

    # ── Plotly ──
    fig_p = px.bar(top_p, x="Cantidad", y="País", orientation="h",
                   color="Cantidad", color_continuous_scale=["#AED6F1", C_MOVIE],
                   title="Países con mayor producción — Top 15",
                   labels={"Cantidad": "Títulos en el catálogo"})
    fig_p.update_layout(plot_bgcolor=C_BG, coloraxis_showscale=False)
    guardar_plotly(fig_p, "viz5_paises")

viz5_paises()


# ═════════════════════════════════════════════
# VIZ 6 — TOP IDIOMAS (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz6_idiomas():
    print("\n[VIZ 6] Distribución por Idioma")

    lang_map = {
        "en":"Inglés","fr":"Francés","ja":"Japonés","ko":"Coreano","es":"Español",
        "zh":"Chino","it":"Italiano","hi":"Hindi","de":"Alemán","ru":"Ruso",
        "tl":"Filipino","ar":"Árabe","pt":"Portugués","nl":"Holandés","tr":"Turco"
    }

    def top_lang(df, label, n=10):
        t = df["language"].map(lang_map).fillna(df["language"]).value_counts().head(n).reset_index()
        t.columns = ["Idioma","Cantidad"]
        t["Tipo"] = label
        return t

    top_m = top_lang(movies, "Películas")
    top_s = top_lang(shows,  "Series")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, color, titulo in [
        (ax1, top_m, C_MOVIE, "Películas — Top 10 idiomas"),
        (ax2, top_s, C_SHOW,  "Series — Top 10 idiomas"),
    ]:
        data_s = data.sort_values("Cantidad")
        bars = ax.barh(data_s["Idioma"], data_s["Cantidad"],
                       color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=9)
        ax.set_title(titulo, pad=10)
        ax.set_xlabel("Cantidad de contenidos")
        ax.set_facecolor(C_BG)

    fig.suptitle("Distribución del catálogo por idioma — Top 10",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    guardar(fig, "viz6_idiomas")

    # ── Plotly ──
    combined_l = pd.concat([top_m, top_s])
    fig_p = px.bar(combined_l, x="Cantidad", y="Idioma", color="Tipo",
                   facet_col="Tipo", orientation="h",
                   color_discrete_map={"Películas": C_MOVIE, "Series": C_SHOW},
                   title="Distribución por idioma — Películas vs. Series (Top 10)")
    fig_p.update_layout(plot_bgcolor=C_BG, showlegend=False)
    guardar_plotly(fig_p, "viz6_idiomas")

viz6_idiomas()


# ═════════════════════════════════════════════
# VIZ 7 — POPULARIDAD PROMEDIO POR GÉNERO (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz7_popularidad_genero():
    print("\n[VIZ 7] Popularidad Promedio por Género")

    def pop_genero(df, label, n=12):
        rows = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows.append({"Género": g, "popularity": row["popularity"]})
        gdf = pd.DataFrame(rows)
        top = gdf.groupby("Género")["popularity"].mean().sort_values(ascending=False).head(n).reset_index()
        top.columns = ["Género","Popularidad promedio"]
        top["Tipo"] = label
        return top

    top_m = pop_genero(movies, "Películas")
    top_s = pop_genero(shows,  "Series")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, color, titulo in [
        (ax1, top_m, C_MOVIE, "Películas — Popularidad promedio por género"),
        (ax2, top_s, C_SHOW,  "Series — Popularidad promedio por género"),
    ]:
        data_s = data.sort_values("Popularidad promedio")
        bars = ax.barh(data_s["Género"], data_s["Popularidad promedio"],
                       color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f"{bar.get_width():.1f}", va="center", fontsize=9)
        ax.set_title(titulo, pad=10)
        ax.set_xlabel("Índice de popularidad promedio (TMDB)")
        ax.set_facecolor(C_BG)

    fig.suptitle("Géneros con mayor popularidad promedio en el catálogo",
                 fontsize=14, fontweight="bold", y=1.02)
    nota_fig(ax2, NOTA_POPULARITY, y=-0.15)
    plt.tight_layout()
    guardar(fig, "viz7_popularidad_genero")

    # ── Plotly ──
    combined_pg = pd.concat([top_m, top_s])
    fig_p = px.bar(combined_pg, x="Popularidad promedio", y="Género", color="Tipo",
                   facet_col="Tipo", orientation="h",
                   color_discrete_map={"Películas": C_MOVIE, "Series": C_SHOW},
                   title="Géneros con mayor popularidad promedio (índice TMDB)")
    fig_p.update_layout(plot_bgcolor=C_BG, showlegend=False)
    guardar_plotly(fig_p, "viz7_popularidad_genero")

viz7_popularidad_genero()


# ═════════════════════════════════════════════
# VIZ 8 — SCATTER POPULARIDAD vs VALORACIÓN (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz8_scatter():
    print("\n[VIZ 8] Popularidad vs. Valoración (Scatter)")

    # Filtrar outliers extremos en popularity para mejor legibilidad
    p95 = combined["popularity"].quantile(0.95)
    df_plot = combined[combined["popularity"] <= p95].copy()
    # Filtrar vote_average == 0 (sin valoración real)
    df_plot = df_plot[df_plot["vote_average"] > 0]
    # Muestra para no saturar el gráfico estático
    sample = df_plot.sample(n=min(3000, len(df_plot)), random_state=42)

    color_map = {"Película": C_MOVIE, "Serie": C_SHOW}

    # ── Seaborn ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for tipo, color in color_map.items():
        d = sample[sample["type_label"] == tipo]
        ax.scatter(d["popularity"], d["vote_average"],
                   c=color, alpha=0.35, s=18, label=tipo, edgecolors="none")

    ax.set_xlabel("Índice de popularidad (TMDB)")
    ax.set_ylabel("Calificación promedio del público (0–10)")
    ax.set_title("Popularidad vs. Calificación del público\nen el catálogo de StreamView Analytics", pad=12)
    ax.legend(title="Tipo de contenido")
    ax.set_facecolor(C_BG)
    ax.set_ylim(0, 10.5)

    # Cuadrantes de referencia
    med_pop = df_plot["popularity"].median()
    med_vot = df_plot["vote_average"].median()
    ax.axvline(med_pop, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(med_vot, color=C_GRAY, linestyle="--", linewidth=1, alpha=0.6)
    ax.text(med_pop*1.05, 9.5, "Alta popularidad\nAlta valoración", fontsize=8, color=C_GRAY)
    ax.text(0.5, 9.5, "Baja popularidad\nAlta valoración", fontsize=8, color=C_GRAY)

    nota_fig(ax, NOTA_POPULARITY + "  |  Se excluye el 5% de valores extremos de popularidad para mayor legibilidad.")
    plt.tight_layout()
    guardar(fig, "viz8_scatter_pop_val")

    # ── Plotly (muestra mayor, interactivo) ──
    sample_p = df_plot.sample(n=min(5000, len(df_plot)), random_state=42)
    fig_p = px.scatter(sample_p, x="popularity", y="vote_average",
                       color="type_label", size="vote_count",
                       size_max=15, opacity=0.5,
                       hover_name="title",
                       hover_data={"popularity":":.1f","vote_average":":.2f","vote_count":True},
                       color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
                       labels={"popularity":"Popularidad (TMDB)",
                               "vote_average":"Calificación promedio (0–10)",
                               "type_label":"Tipo"},
                       title="Popularidad vs. Calificación del público — catálogo StreamView")
    fig_p.update_layout(plot_bgcolor=C_BG)
    guardar_plotly(fig_p, "viz8_scatter_pop_val")

viz8_scatter()


# ═════════════════════════════════════════════
# VIZ 9 — DISTRIBUCIÓN POR CLASIFICACIÓN ETARIA (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz9_clasificacion():
    print("\n[VIZ 9] Distribución por Clasificación Etaria")

    combined["rating_cat"] = combined["rating"].apply(lambda x:
        "TV-MA / R (Adultos)" if x in [8.0, 7.2, 7.0, 6.0] else str(x))

    # Reconstruir con etiquetas legibles
    rating_map = {
        8.0: "TV-MA",  7.2: "TV-14", 7.0: "R",
        6.0: "PG-13",  5.0: "PG",    4.0: "TV-PG",
        3.0: "TV-G",   2.0: "G",
    }
    combined["rating_label"] = combined["rating"].map(rating_map).fillna("Otra")

    counts = combined.groupby(["rating_label","type_label"]).size().reset_index(name="Cantidad")
    # Orden lógico de clasificación
    order = ["G","TV-G","PG","TV-PG","PG-13","TV-14","R","TV-MA","Otra"]
    counts["rating_label"] = pd.Categorical(counts["rating_label"], categories=order, ordered=True)
    counts = counts.sort_values("rating_label")

    # ── Seaborn ──
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = counts.pivot(index="rating_label", columns="type_label", values="Cantidad").fillna(0)
    pivot = pivot.reindex(order).dropna(how="all")
    pivot.plot(kind="bar", ax=ax, color=[C_MOVIE, C_SHOW], edgecolor="white",
               width=0.7, rot=0)
    ax.set_title("Distribución del catálogo por clasificación etaria", pad=12)
    ax.set_xlabel("Clasificación por edad")
    ax.set_ylabel("Cantidad de contenidos")
    ax.legend(title="Tipo de contenido")
    ax.set_facecolor(C_BG)
    plt.tight_layout()
    guardar(fig, "viz9_clasificacion_etaria")

    # ── Plotly ──
    fig_p = px.bar(counts, x="rating_label", y="Cantidad", color="type_label",
                   barmode="group",
                   color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
                   category_orders={"rating_label": order},
                   labels={"rating_label":"Clasificación","type_label":"Tipo"},
                   title="Distribución por clasificación etaria — Películas vs. Series")
    fig_p.update_layout(plot_bgcolor=C_BG)
    guardar_plotly(fig_p, "viz9_clasificacion_etaria")

viz9_clasificacion()


# ═════════════════════════════════════════════
# VIZ 10 — TOP 10 CONTENIDOS POR POPULARIDAD (Seaborn + Plotly)
# ═════════════════════════════════════════════
def viz10_top_popularidad():
    print("\n[VIZ 10] Top 10 Contenidos por Popularidad")

    top_m = movies.nlargest(10, "popularity")[["title","popularity","vote_average","type_label"]]
    top_s = shows.nlargest(10, "popularity")[["title","popularity","vote_average","type_label"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, color, titulo in [
        (ax1, top_m, C_MOVIE, "Películas más populares — Top 10"),
        (ax2, top_s, C_SHOW,  "Series más populares — Top 10"),
    ]:
        data_s = data.sort_values("popularity")
        bars = ax.barh(data_s["title"].str[:30], data_s["popularity"],
                       color=color, alpha=0.85, edgecolor="white")
        for bar, (_, row) in zip(bars, data_s.iterrows()):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{row['popularity']:.0f}  |  ★{row['vote_average']:.1f}",
                    va="center", fontsize=8.5, color=C_DARK)
        ax.set_title(titulo, pad=10)
        ax.set_xlabel("Índice de popularidad (TMDB)")
        ax.set_facecolor(C_BG)

    fig.suptitle("Contenidos con mayor popularidad en el catálogo",
                 fontsize=14, fontweight="bold", y=1.02)
    nota_fig(ax2, NOTA_POPULARITY, y=-0.15)
    plt.tight_layout()
    guardar(fig, "viz10_top_popularidad")

    # ── Plotly ──
    top_all = pd.concat([top_m, top_s])
    fig_p = px.bar(top_all, x="popularity", y="title", color="type_label",
                   facet_col="type_label", orientation="h",
                   color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
                   hover_data=["vote_average"],
                   title="Top 10 contenidos más populares — Películas y Series")
    fig_p.update_layout(plot_bgcolor=C_BG, showlegend=False)
    guardar_plotly(fig_p, "viz10_top_popularidad")

viz10_top_popularidad()


# ═════════════════════════════════════════════
# RESUMEN FINAL
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("✅  TODAS LAS VISUALIZACIONES GENERADAS")
print("="*60)
print(f"\n📁 Imágenes estáticas  →  images/   ({len(os.listdir(IMG_DIR))} archivos)")
print(f"📁 Gráficos interactivos → dashboard/ ({len(os.listdir(DASH_DIR))} archivos)")
print("\nArchivos generados:")
for f in sorted(os.listdir(IMG_DIR)):
    print(f"  images/{f}")
for f in sorted(os.listdir(DASH_DIR)):
    print(f"  dashboard/{f}")
