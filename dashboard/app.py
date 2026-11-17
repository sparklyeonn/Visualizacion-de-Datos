"""
STREAMVIEW ANALYTICS | DASHBOARD INTERACTIVO
Diseño Visualizacion de Datos | ADY1104 Duoc UC 2026
"""

import os, warnings, io, base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, dash_table, ALL, no_update
import dash_bootstrap_components as dbc
from functools import lru_cache

warnings.filterwarnings("ignore")

# ── COLORES DE INTERFAZ (chrome) ────────────────────────────────
BG_MAIN              = "var(--bg-page)"
BG_CARD              = "var(--bg-card)"
BG_SIDEBAR           = "var(--bg-sidebar)"
C_TEXT               = "var(--text-primary)"
C_TEXT_SIDEBAR       = "var(--text-on-sidebar)"
C_MUTED              = "var(--text-muted)"
C_MUTED_SIDEBAR      = "var(--text-muted-on-sidebar)"
C_BORDER             = "var(--border-color)"
C_ACCENT_PRIMARY     = "var(--accent-primary)"

# ── COLORES DE GRÁFICOS (Plotly) ────────────────────────────────
CHART_THEMES = {
    "dark":  dict(bg="#1A1D27", text="#E2E8F0", muted="#64748B", border="#2D3148"),
    "light": dict(bg="#FFFFFF", text="#0F172A", muted="#64748B", border="#E2E8F0"),
}

C_ACCENT1  = "#7C3AED"
C_ACCENT2  = "#06B6D4"
C_ACCENT3  = "#F472B6"
C_ACCENT4  = "#10B981"
C_ACCENT5  = "#3B82F6"
C_ACCENT6  = "#EF4444"
C_MOVIE    = "#7C3AED"
C_SHOW     = "#06B6D4"
C_HIGHLIGHT = "#F5A623"

# Colores de acción sobre fondo claro (incluida la descarga) y oscuro.
ACCIONES = {
    "1 · Destacar": ("Candidatos para destacar", "#127547", "#58D79A"),
    "2 · Descubrimiento": ("Explorar descubrimiento", "#205AB9", "#75A8FF"),
    "3 · Revisar expectativas": ("Revisar expectativas", "#975800", "#F9BC55"),
    "4 · Menor prioridad": ("Menor prioridad inicial", "#626D79", "#A8B2BE"),
}
NOTA_MEDIANAS = (
    "Los cuadrantes se calculan utilizando las medianas de popularidad y valoración del segmento filtrado. "
    "Las categorías representan posiciones relativas y no calificaciones absolutas de calidad o éxito comercial."
)

FONT = "Inter, system-ui, sans-serif"

LANG_MAP = {
    "en":"Inglés","fr":"Francés","ja":"Japonés","ko":"Coreano","es":"Español",
    "zh":"Chino","it":"Italiano","hi":"Hindi","de":"Alemán","ru":"Ruso",
    "tl":"Filipino","ar":"Árabe","pt":"Portugués","nl":"Holandés","tr":"Turco"
}

# ── CARGA ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

movies = pd.read_csv(os.path.join(DATA_DIR, "netflix_movies_detailed_up_to_2025.csv"))
shows  = pd.read_csv(os.path.join(DATA_DIR, "netflix_tv_shows_detailed_up_to_2025.csv"))

# 9 show_ids duplicados en shows (snapshots TMDB distintos, mismos metadatos).
# Criterio: conservar el registro con mayor popularidad.
if shows["show_id"].duplicated().any():
    shows = shows.sort_values("popularity", ascending=False).drop_duplicates(subset="show_id", keep="first")

for df, label in [(movies,"Película"), (shows,"Serie")]:
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["type_label"] = label
    df["lang_label"] = df["language"].map(LANG_MAP).fillna(df["language"])
    df["rating_bin"] = pd.cut(
        df["vote_average"],
        bins=[0,4,5,6,7,8,10],
        labels=["<4","4–5","5–6","6–7","7–8","8–10"],
        right=True
    )

COMMON = ["show_id","type","type_label","title","country","year_added",
          "release_year","genres","language","lang_label",
          "popularity","vote_count","vote_average","rating_bin"]
combined = pd.concat([movies[COMMON], shows[COMMON]], ignore_index=True)
# uid compuesto para no confundir películas y series con el mismo show_id
combined["uid"] = combined["type_label"].str[:3] + "_" + combined["show_id"].astype(str)

combined = combined.drop_duplicates("uid").reset_index(drop=True)
# Normalizar los separadores una sola vez para filtros, rankings y descargas.
for campo in ("genres", "country"):
    combined[campo] = (combined[campo].str.strip()
                       .str.replace(r"\s*,\s*", ", ", regex=True)
                       .replace("", np.nan))

YEAR_MIN = int(combined["year_added"].min())
YEAR_MAX = int(combined["year_added"].max())

# ── HELPERS ──────────────────────────────────────────────────────
def indice_etiquetas(columna):
    # Agrupar las posiciones una sola vez. Cada título conserva una fila en combined.
    etiquetas = combined[columna].dropna().str.split(",").explode().str.strip()
    etiquetas = etiquetas[etiquetas.ne("")]
    return {valor: grupo.index.unique().to_numpy()
            for valor, grupo in etiquetas.groupby(etiquetas)}


INDICES = {campo: indice_etiquetas(campo) for campo in ("genres", "country")}
TITULOS_POR_ANIO = combined["year_added"].value_counts().sort_index()


def seleccion(valores):
    if isinstance(valores, str):
        return [valores] if valores else []
    return valores or []


def filtrar(tipo="Todos", year_range=None, genero=None, pais=None, idioma=None, cruces=None):
    mascara = np.ones(len(combined), dtype=bool)
    if tipo and tipo != "Todos":
        mascara &= combined["type_label"].eq(tipo).to_numpy()
    if year_range:
        mascara &= combined["year_added"].between(*year_range).to_numpy()
    if idioma:
        mascara &= combined["language"].isin(seleccion(idioma)).to_numpy()

    # OR en el desplegable. Cada cruce es una restricción extra y reversible.
    activos = dict(cruces or [])
    for campo, valores in (("genres", genero), ("country", pais)):
        if valores:
            coincide = np.zeros(len(combined), dtype=bool)
            for valor in seleccion(valores):
                coincide[INDICES[campo].get(valor, [])] = True
            mascara &= coincide
        if campo in activos:
            coincide = np.zeros(len(combined), dtype=bool)
            coincide[INDICES[campo].get(activos[campo], [])] = True
            mascara &= coincide
    return combined.loc[mascara].copy()


def base_layout(title="", bg="#1A1D27", text="#E2E8F0", muted="#64748B", border="#2D3148"):
    return dict(
        title=dict(text=title, font=dict(size=13, color=text, family=FONT), x=0, pad=dict(l=4)),
        plot_bgcolor=bg,
        paper_bgcolor=bg,
        font=dict(family=FONT, size=11, color=text),
        margin=dict(l=8, r=16, t=44, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=10, color=muted),
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(bgcolor=bg, font_size=12, font_color=text,
                        bordercolor=border),
        xaxis=dict(showgrid=True, gridcolor=border, gridwidth=1,
                   zeroline=False, tickfont=dict(color=muted, size=10),
                   title_font=dict(color=muted, size=10)),
        yaxis=dict(showgrid=True, gridcolor=border, gridwidth=1,
                   zeroline=False, tickfont=dict(color=muted, size=10),
                   title_font=dict(color=muted, size=10)),
    )

def card(children, padding="20px", height=None):
    style = {
        "backgroundColor": BG_CARD,
        "borderRadius": "16px",
        "border": f"1px solid {C_BORDER}",
        "boxShadow": "var(--shadow-card)",
        "padding": padding,
        "height": height or "auto",
    }
    return html.Div(children, style=style)

def kpi_card(label, value, subtitle="", color=C_ACCENT1, icon=""):
    return card([
        html.Div([
            html.Span(icon + " ", style={"fontSize":"16px"}),
            html.Span(label, style={
                "fontSize":"10px","fontWeight":"700","color":C_MUTED,
                "textTransform":"uppercase","letterSpacing":"1px"
            }),
        ], style={"marginBottom":"8px"}),
        html.Div(value, style={
            "fontSize":"28px","fontWeight":"800","color":color,
            "lineHeight":"1","marginBottom":"4px",
            "fontVariantNumeric":"tabular-nums"
        }),
        html.Div(subtitle, style={"fontSize":"10px","color":C_MUTED}),
    ], padding="16px 18px")


# ══════════════════════════════════════════════════════════════════
# GENERADOR DE INFOGRAFÍA
# ══════════════════════════════════════════════════════════════════
def generar_infografia_png(tipo, year_range, top_n=5, min_votos=30, genero=None, pais=None, idioma=None, cruces=None):
    import base64
    import io
    import textwrap

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import FancyBboxPatch

    # ── DATOS ──────────────────────────────────────────────
    df = filtrar(tipo, year_range, genero, pais, idioma, cruces)
    df = df.drop_duplicates(["type_label", "show_id"])

    min_votos = max(0, int(min_votos or 0))
    for col in ["popularity", "vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    elegibles = df[
        (df["popularity"] > 0)
        & (df["vote_average"] > 0)
        & (df["vote_count"] > 0)
        & (df["vote_count"] >= min_votos)
    ].copy()

    nombres = [
        "Explorar descubrimiento",
        "Candidatos para destacar",
        "Menor prioridad inicial",
        "Revisar expectativas",
    ]
    conteos = dict.fromkeys(nombres, 0)
    candidatos = elegibles.iloc[:0]
    med_pop = med_val = None

    if not elegibles.empty:
        med_pop = elegibles["popularity"].median()
        med_val = elegibles["vote_average"].median()

        pop_alta = elegibles["popularity"] >= med_pop
        val_alta = elegibles["vote_average"] >= med_val

        elegibles["cuadrante"] = np.select(
            [
                ~pop_alta & val_alta,
                pop_alta & val_alta,
                ~pop_alta & ~val_alta,
                pop_alta & ~val_alta,
            ],
            nombres,
            default="Sin clasificar",
        )

        conteos.update(elegibles["cuadrante"].value_counts().to_dict())
        candidatos = (
            elegibles[
                elegibles["cuadrante"] == "Explorar descubrimiento"
            ]
            .sort_values(
                ["vote_count", "vote_average", "title", "show_id"],
                ascending=[False, False, True, True],
            )
            .head(3)
        )

    # Géneros: cada título cuenta una vez dentro de cada género.
    generos = df[
        ["type_label", "show_id", "genres", "popularity"]
    ].copy()
    generos["genero"] = generos["genres"].fillna("").str.split(",")
    generos = generos.explode("genero")
    generos["genero"] = generos["genero"].str.strip()
    generos = generos[generos["genero"].ne("")]
    generos = generos.drop_duplicates(
        ["type_label", "show_id", "genero"]
    )

    resumen = (
        generos.groupby("genero")
        .agg(
            cantidad=("show_id", "size"),
            popularidad=("popularity", "mean"),
        )
        .reset_index()
    )
    resumen["puesto_pop"] = resumen["popularidad"].rank(
        ascending=False, method="min"
    )
    resumen = resumen.sort_values(
        ["cantidad", "genero"], ascending=[False, True]
    )
    top_generos = resumen.head(3)

    traducciones = {
        "Comedy": "Comedia",
        "Animation": "Animación",
        "Thriller": "Suspenso",
        "Action": "Acción",
        "Family": "Familia",
        "Mystery": "Misterio",
        "Horror": "Terror",
        "Documentary": "Documental",
        "Adventure": "Aventura",
        "Science Fiction": "Ciencia ficción",
        "Fantasy": "Fantasía",
        "Unknown": "Sin clasificar",
    }

    def entero(valor):
        return f"{int(valor):,}".replace(",", ".")

    def decimal(valor, cifras=2):
        return f"{valor:.{cifras}f}".replace(".", ",")

    def envolver(texto, ancho):
        return textwrap.fill(str(texto), width=ancho)

    tipo_texto = tipo if tipo and tipo != "Todos" else "Películas y series"
    inicio, fin = year_range or [YEAR_MIN, YEAR_MAX]

    # ── DISEÑO ─────────────────────────────────────────────
    fondo = "#F5F6FA"
    tinta = "#192337"
    secundario = "#566177"
    morado = "#7C3AED"
    oscuro = "#17182E"

    detalles = []
    for nombre, valores in (("Género", genero), ("País", pais), ("Idioma", idioma)):
        if valores:
            etiquetas = seleccion(valores)
            if nombre == "Idioma":
                etiquetas = [LANG_MAP.get(v, v) for v in etiquetas]
            detalles.append(f"{nombre}: {', '.join(etiquetas)}")
    for campo, valor in dict(cruces or []).items():
        nombre = "género" if campo == "genres" else "país"
        detalles.append(f"Clic en {nombre}: {valor}")
    lineas_filtros = textwrap.wrap(" · ".join(detalles), width=95)
    alto_extra = len(lineas_filtros) * 0.22
    fig = plt.figure(figsize=(10, 14 + alto_extra), facecolor=fondo)
    if lineas_filtros:
        fig.text(0.065, 1 - 0.08 / (14 + alto_extra), "\n".join(lineas_filtros),
                 va="top", fontsize=9, color=tinta)
    ax = fig.add_axes([0, 0, 1, 14 / (14 + alto_extra)])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def caja(x, y, w, h, color="white", radio=0.012):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radio}",
                facecolor=color,
                edgecolor="none",
            )
        )

    def texto(x, y, contenido, size=10, color=tinta,
              bold=False, ha="left"):
        ax.text(
            x, y, contenido,
            fontsize=size,
            color=color,
            fontweight="bold" if bold else "normal",
            fontfamily="DejaVu Sans",
            ha=ha,
            va="top",
            linespacing=1.35,
        )

    try:
        # Encabezado
        caja(0.04, 0.845, 0.92, 0.12, oscuro)
        texto(0.065, 0.947, "STREAMVIEW ANALYTICS",
              10, "#C5B4FF", True)
        texto(0.065, 0.919,
              "¿Qué contenidos conviene promocionar?",
              19, "white", True)
        texto(0.065, 0.882,
              f"{tipo_texto} · {int(inicio)}–{int(fin)} · {len(df):,} títulos",
              10, "#D4D8E4")
        texto(0.065, 0.862,
              f"Oportunidades: mínimo {min_votos} votos y al menos 1 voto",
              9, "#D4D8E4")

        # Indicadores
        peliculas = int((df["type_label"] == "Película").sum())
        series = int((df["type_label"] == "Serie").sum())

        tarjetas = [
            (
                len(df),
                "Títulos en la muestra",
                f"{entero(peliculas)} películas · {entero(series)} series",
            ),
            (
                len(elegibles),
                "Títulos elegibles",
                "Cumplen criterios de votos,\nvaloración y popularidad",
            ),
            (
                conteos["Explorar descubrimiento"],
                "Para explorar descubrimiento",
                "Alta valoración y baja\npopularidad relativas",
            ),
        ]

        for i, (valor, etiqueta, detalle) in enumerate(tarjetas):
            x = 0.04 + i * 0.315
            caja(x, 0.725, 0.29, 0.102)
            texto(x + 0.018, 0.814, entero(valor), 25, morado, True)
            texto(x + 0.018, 0.779, etiqueta, 9, tinta, True)
            texto(x + 0.018, 0.758, detalle, 8, secundario)

        # Géneros y hallazgo
        caja(0.04, 0.495, 0.92, 0.213)
        texto(0.065, 0.692, "01  ¿Qué géneros predominan?", 13, tinta, True)

        if top_generos.empty:
            texto(0.065, 0.649, "Sin géneros disponibles para esta selección.")
        else:
            maximo = top_generos["cantidad"].max()

            for i, (_, fila) in enumerate(top_generos.iterrows()):
                y = 0.649 - i * 0.03
                nombre = traducciones.get(fila["genero"], fila["genero"])
                texto(0.065, y, envolver(nombre, 23), 9)
                caja(0.30, y - 0.014, 0.49, 0.009, "#EEEAF5", 0.003)
                caja(
                    0.30, y - 0.014,
                    0.49 * fila["cantidad"] / maximo,
                    0.009, morado, 0.003,
                )
                texto(0.93, y, entero(fila["cantidad"]), 10, ha="right")

            lider = resumen.iloc[0]
            nombre = traducciones.get(lider["genero"], lider["genero"])
            if pd.notna(lider["puesto_pop"]):
                hallazgo = (
                    f"{nombre}: #1 en cantidad y "
                    f"#{int(lider['puesto_pop'])} de {len(resumen)} "
                    "en popularidad promedio."
                )
            else:
                hallazgo = f"{nombre} lidera en cantidad de títulos."

            caja(0.065, 0.513, 0.87, 0.052, "#F0E9FC")
            texto(0.078, 0.555, envolver(hallazgo, 91), 9, tinta, True)

        texto(0.065, 0.507,
              "Un título puede pertenecer a varios géneros.",
              8, secundario)

        # ── PUNTO 2: CANTIDADES + EJEMPLOS ──────────────────
        caja(0.04, 0.265, 0.92, 0.214)

        texto(
            0.065, 0.463,
            "02  ¿Dónde están las oportunidades?",
            13, tinta, True,
        )

        descripciones = [
            "Alta valoración · Baja popularidad",
            "Alta valoración · Alta popularidad",
            "Baja valoración · Baja popularidad",
            "Baja valoración · Alta popularidad",
        ]

        acentos = {nombre: claro for nombre, claro, oscuro in ACCIONES.values()}

        # Conserva el orden de los cuatro grupos.
        grupos = [
            "Explorar descubrimiento",
            "Candidatos para destacar",
            "Menor prioridad inicial",
            "Revisar expectativas",
        ]

        for i, nombre in enumerate(grupos):
            # Cuatro filas: tarjeta izquierda + ejemplos derecha.
            y = 0.402 - i * 0.040
            alto = 0.038

            caja(
                0.065, y,
                0.425, alto,
                "#F2F4F7",
                0.006,
            )

            caja(0.065, y, 0.006, alto, acentos[nombre], 0.002)

            # Tarjeta: nombre, cantidad y descripción.
            texto(
                0.078, y + 0.035,
                nombre,
                8, tinta, True,
            )

            texto(
                0.078, y + 0.024,
                entero(conteos.get(nombre, 0)),
                14, tinta, True,
            )

            texto(
                0.078, y + 0.009,
                descripciones[i],
                7, secundario,
            )

            # Selección dinámica de ejemplos del mismo cuadrante.
            if (
                not elegibles.empty
                and "cuadrante" in elegibles.columns
            ):
                ejemplos = (
                    elegibles[
                        elegibles["cuadrante"] == nombre
                    ]
                    .sort_values(
                        [
                            "vote_count",
                            "vote_average",
                            "title",
                            "show_id",
                        ],
                        ascending=[False, False, True, True],
                    )
                    .head(2)
                )
            else:
                ejemplos = elegibles.iloc[:0]

            texto(
                0.52, y + 0.035,
                "EJEMPLOS DEL GRUPO",
                7, secundario,
            )

            if ejemplos.empty:
                texto(
                    0.52, y + 0.021,
                    "Sin títulos con los filtros actuales.",
                    8, secundario,
                )
            else:
                for j, (_, fila) in enumerate(ejemplos.iterrows()):
                    # Acorta títulos largos para que no se superpongan.
                    titulo = textwrap.shorten(
                        str(fila["title"]),
                        width=43,
                        placeholder="…",
                    )

                    texto(
                        0.52,
                        y + 0.023 - j * 0.012,
                        titulo,
                        8.5, tinta,
                    )

        # Nota breve: explica cómo se eligieron los ejemplos.
        if med_pop is None:
            nota = (
                "Sin títulos elegibles. Ajusta los años "
                "o el mínimo de votos."
            )
        else:
            nota = (
                f"Ejemplos por cantidad de votos. Cortes: "
                f"popularidad {decimal(med_pop)} "
                f"y valoración {decimal(med_val)}; igualdad = alto."
            )

        texto(
            0.065, 0.277,
            nota,
            7, secundario,
        )

        # Candidatos
        caja(0.04, 0.11, 0.92, 0.14)
        texto(0.065, 0.236,
              "03  Candidatos para una prueba de descubrimiento",
              12, tinta, True)
        texto(0.065, 0.212, "TÍTULO / FORMATO", 8, secundario)
        texto(0.77, 0.212, "NOTA /10", 8, secundario, ha="right")
        texto(0.93, 0.212, "VOTOS", 8, secundario, ha="right")

        if candidatos.empty:
            texto(0.065, 0.181,
                  "Sin candidatos en este cuadrante con los filtros actuales.",
                  9, secundario)
        else:
            for i, (_, fila) in enumerate(candidatos.iterrows()):
                y = 0.192 - i * 0.023
                titulo = textwrap.shorten(
                    str(fila["title"]), width=51, placeholder="…"
                )
                texto(0.065, y, f"{titulo} · {fila['type_label']}", 8.5)
                texto(0.77, y, decimal(fila["vote_average"]), 9, ha="right")
                texto(0.93, y, entero(fila["vote_count"]), 9, ha="right")

        texto(0.065, 0.124,
              "Ordenados por votos. “Descubrimiento” no significa desconocidos.",
              8, secundario)

        # Acción
        caja(0.04, 0.055, 0.92, 0.043, oscuro)
        texto(0.065, 0.089, "ACCIÓN SUGERIDA", 9, "#C5B4FF", True)
        accion = (
            "Probar una colección con estos candidatos y medir clics y reproducciones."
            if not candidatos.empty
            else "Revisar otros filtros para identificar candidatos a una prueba de promoción."
        )
        texto(0.065, 0.074, accion, 9, "white")

        # Fuente y límites
        texto(
            0.04, 0.043,
            "Fuente: datasets del proyecto con indicadores TMDB. "
            "Popularidad no equivale a reproducciones.\n"
            "Muestra seleccionada, no catálogo completo. "
            "Medir campañas requiere datos adicionales.",
            7.5, secundario,
        )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=150,
            facecolor=fondo,
        )
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    finally:
        plt.close(fig)


# ── ESTILOS CSS GLOBALES ─────────────────────────────────────────
external_css = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
]

app = Dash(__name__,
           external_stylesheets=external_css,
           title="StreamView Analytics")
server = app.server

# Scrollbar discreta para el sidebar — CSS inyectado sin archivo externo
app.index_string = app.index_string.replace(
    "</head>",
    """<style>
.sv-sidebar::-webkit-scrollbar { width: 4px; }
.sv-sidebar::-webkit-scrollbar-track { background: transparent; }
.sv-sidebar::-webkit-scrollbar-thumb {
    background: var(--border-color-sidebar, #2D3148);
    border-radius: 4px;
}
.sv-sidebar::-webkit-scrollbar-thumb:hover {
    background: var(--text-muted-on-sidebar, #64748B);
}
</style></head>"""
)

# ── SIDEBAR ──────────────────────────────────────────────────────
sidebar_header = html.Div([
    html.Div([
        html.Div("▶", style={
            "fontSize":"22px","color":C_TEXT_SIDEBAR,
            "lineHeight":"1","marginBottom":"2px"
        }),
        html.Div("StreamView", style={
            "fontSize":"15px","fontWeight":"800","color":C_TEXT_SIDEBAR,"letterSpacing":"-0.3px"
        }),
        html.Div("Analytics EP1", style={
            "fontSize":"10px","color":C_MUTED_SIDEBAR,"letterSpacing":"0.5px"
        }),
    ]),
    html.Button("Menú", id="mobile-menu-btn", n_clicks=0, className="sv-mobile-menu-btn"),
], className="sv-sidebar-header", style={"marginBottom":"32px"})

def selector_catalogo(id_control, etiqueta, opciones, placeholder):
    return html.Div([
        html.Label(etiqueta, htmlFor=id_control, style={
            "fontSize": "11px", "fontWeight": "600", "color": C_TEXT_SIDEBAR,
            "marginBottom": "6px",
        }),
        dcc.Dropdown(id=id_control, options=opciones, placeholder=placeholder,
                     multi=True, value=[], clearable=True, searchable=True, style={"color": "#192337"}),
    ], style={"marginBottom": "14px"})


sidebar_body = html.Div([
    html.Div("FILTROS", style={
        "fontSize":"9px","fontWeight":"700","color":C_MUTED_SIDEBAR,
        "letterSpacing":"1.5px","marginBottom":"14px"
    }),

    html.Div("Tipo de contenido", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT_SIDEBAR,"marginBottom":"8px"
    }),
    dcc.RadioItems(
        id="filtro-tipo",
        options=[{"label": f"  {t}", "value": t} for t in ["Todos","Película","Serie"]],
        value="Todos",
        labelStyle={"display":"block","marginBottom":"8px","fontSize":"12px","color":C_MUTED_SIDEBAR,"cursor":"pointer"},
        inputStyle={"marginRight":"8px","accentColor":C_ACCENT_PRIMARY},
        style={"marginBottom":"24px"}
    ),

    selector_catalogo("filtro-genero", "Género", sorted(INDICES["genres"]), "Todos los géneros"),
    selector_catalogo("filtro-pais", "País productor", sorted(INDICES["country"]), "Todos los países"),
    selector_catalogo("filtro-idioma", "Idioma", [
        {"label": LANG_MAP.get(codigo, codigo), "value": codigo}
        for codigo in sorted(combined["language"].dropna().unique(), key=lambda c: LANG_MAP.get(c, c))
    ], "Todos los idiomas"),

    dbc.Button("Deshacer último clic", id="deshacer-cruce", size="sm", outline=True,
               color="light", disabled=True, className="mb-2"),
    dbc.Button("Restablecer filtros", id="restablecer-filtros", size="sm",
               color="light", className="mb-3"),

    html.Div("Años de incorporación", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT_SIDEBAR,"marginBottom":"10px"
    }),
    html.Div("Filtra por date_added, no por año de estreno.", style={
        "fontSize":"9px","color":C_MUTED_SIDEBAR,"marginBottom":"6px","lineHeight":"1.4"
    }),
    dcc.RangeSlider(
        id="filtro-years",
        min=YEAR_MIN, max=YEAR_MAX,
        value=[YEAR_MIN, YEAR_MAX],
        marks={2010:"'10", 2015:"'15", 2020:"'20", 2025:"'25"},
        step=1,
        tooltip={"placement":"bottom","always_visible":False},
    ),
    html.Div(style={"marginBottom":"24px"}),

    html.Div("Ítems en rankings", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT_SIDEBAR,"marginBottom":"10px"
    }),
    dcc.Slider(
        id="filtro-top", min=5, max=20, step=5, value=10,
        marks={5:"5", 10:"10", 15:"15", 20:"20"},
    ),
    html.Div(style={"marginBottom":"20px"}),

    html.Div("Mín. votos (scatter)", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT_SIDEBAR,"marginBottom":"6px"
    }),
    html.Div("Umbral exploratorio, no garantía de representatividad.", style={
        "fontSize":"9px","color":C_MUTED_SIDEBAR,"marginBottom":"6px","lineHeight":"1.4"
    }),
    dcc.Slider(
        id="filtro-min-votos", min=0, max=200, step=10, value=30,
        marks={0:"0", 50:"50", 100:"100", 200:"200"},
        tooltip={"placement":"bottom","always_visible":False},
    ),
    html.Div(style={"marginBottom":"20px"}),

    html.Div("Mín. títulos/género", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT_SIDEBAR,"marginBottom":"6px"
    }),
    html.Div("Para la matriz de oportunidades.", style={
        "fontSize":"9px","color":C_MUTED_SIDEBAR,"marginBottom":"6px","lineHeight":"1.4"
    }),
    dcc.Slider(
        id="filtro-min-titulos-genero", min=5, max=100, step=5, value=30,
        marks={5:"5", 30:"30", 60:"60", 100:"100"},
        tooltip={"placement":"bottom","always_visible":False},
    ),
    html.Div(style={"marginBottom":"32px"}),

    html.Div("VISTAS", style={
        "fontSize":"9px","fontWeight":"700","color":C_MUTED_SIDEBAR,
        "letterSpacing":"1.5px","marginBottom":"14px"
    }),
    dcc.RadioItems(
        id="nav-tab",
        className="sv-nav-tab",
        options=[
            {"label":"Catálogo",              "value":"tab-catalogo"},
            {"label":"Geografía & Idiomas",   "value":"tab-geo"},
            {"label":"Popularidad",           "value":"tab-pop"},
        ],
        value="tab-catalogo",
        labelStyle={
            "display":"block","marginBottom":"6px","fontSize":"12px",
            "color":C_MUTED_SIDEBAR,"cursor":"pointer","padding":"6px 10px",
            "borderRadius":"8px"
        },
        inputStyle={"display":"none"},
        style={"marginBottom":"24px"}
    ),

    html.Div(style={"borderTop":"1px solid var(--border-color-sidebar)","marginBottom":"16px"}),
    html.Div("⚠ Popularidad = índice relativo TMDB.", style={
        "fontSize":"10px","color":C_MUTED_SIDEBAR,"lineHeight":"1.5","marginBottom":"6px"
    }),
    html.Div("ℹ 1.000 registros/año por categoría.", style={
        "fontSize":"10px","color":C_MUTED_SIDEBAR,"lineHeight":"1.5"
    }),
], id="sidebar-body", className="sv-sidebar-body")

sidebar = html.Div([
    sidebar_header,
    sidebar_body,
], className="sv-sidebar", style={
    "backgroundColor": BG_SIDEBAR,
    "width":"200px",
    "height":"100vh",          # altura fija = viewport → activa scroll interno
    "overflowY":"auto",        # scroll con rueda del mouse dentro del sidebar
    "overflowX":"hidden",
    "padding":"24px 16px",
    "position":"fixed","top":"0","left":"0",
    "borderRight":"1px solid var(--border-color-sidebar)",
    "fontFamily":FONT,
    # scrollbar discreta (webkit + scrollbar-width para Firefox)
    "scrollbarWidth":"thin",
    "scrollbarColor":"var(--border-color-sidebar) transparent",
})

# ── LAYOUT PRINCIPAL ─────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="theme-mode", storage_type="local", data="system"),
    dcc.Store(id="theme-resolved", data="dark"),
    dcc.Store(id="cruces", data=[]),
    dcc.Store(id="store-cuadrante", data="Todos"),

    dcc.Download(id="download-infografia"),
    dcc.Download(id="download-infografia-pdf"),

    html.Div([
        html.Div([
            html.P("Elige el formato de descarga", style={
                "fontWeight":"700","fontSize":"13px","marginBottom":"12px","color":"#0F172A"
            }),
            html.Button("⬇ PNG (imagen)", id="btn-dl-png", n_clicks=0, style={
                "display":"block","width":"100%","marginBottom":"8px",
                "padding":"9px","backgroundColor":"#7C3AED","color":"white",
                "border":"none","borderRadius":"8px","cursor":"pointer","fontWeight":"600"
            }),
            html.Button("⬇ PDF (documento)", id="btn-dl-pdf", n_clicks=0, style={
                "display":"block","width":"100%",
                "padding":"9px","backgroundColor":"#3B82F6","color":"white",
                "border":"none","borderRadius":"8px","cursor":"pointer","fontWeight":"600"
            }),
            html.Button("✕", id="btn-cerrar-modal", n_clicks=0, style={
                "position":"absolute","top":"8px","right":"10px",
                "background":"none","border":"none","fontSize":"16px",
                "cursor":"pointer","color":"#94A3B8"
            }),
        ], style={
            "backgroundColor":"white","borderRadius":"12px","padding":"20px",
            "boxShadow":"0 8px 30px rgba(0,0,0,0.18)","position":"relative",
            "width":"220px"
        })
    ], id="modal-formato", style={
        "display":"none","position":"fixed","top":"70px","right":"24px","zIndex":"9999"
    }),

    sidebar,
    html.Div([
        html.Div([
            html.Div([
                html.Div(id="page-title", style={
                    "fontSize":"18px","fontWeight":"800","color":C_TEXT
                }),
                html.Div("ADY1104 · Visualización de Datos · Duoc UC 2026", style={
                    "fontSize":"11px","color":C_MUTED
                }),
            ]),
            html.Div([
                html.Button("Auto", id="theme-toggle-btn", n_clicks=0, className="sv-theme-toggle"),
                html.Button(
                    "⬇ Descargar resumen",
                    id="btn-descargar-infografia",
                    n_clicks=0,
                    style={
                        "fontSize":"11px","fontWeight":"600","color":"#E2E8F0",
                        "backgroundColor":"#7C3AED","border":"none","borderRadius":"8px",
                        "padding":"7px 14px","cursor":"pointer","marginLeft":"10px",
                        "letterSpacing":"0.2px","transition":"opacity 0.15s",
                    }
                ),
            ], style={"display":"flex","alignItems":"center"}),
        ], className="sv-header", style={
            "padding":"20px 24px 16px",
            "borderBottom":f"1px solid {C_BORDER}",
            "marginBottom":"20px",
            "display":"flex","justifyContent":"space-between","alignItems":"flex-start"
        }),

        html.Div(id="kpi-row", className="sv-kpi-row", style={"padding":"0 24px","marginBottom":"20px"}),
        html.Div(id="tab-content", className="sv-tab-content", style={"padding":"0 24px 32px"}),

    ], className="sv-main", style={"marginLeft":"200px","backgroundColor":BG_MAIN,"minHeight":"100vh","fontFamily":FONT}),

], style={"backgroundColor":BG_MAIN,"fontFamily":FONT})


# ── TEMA ─────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(n_clicks, mode) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        var order = ["system", "light", "dark"];
        var idx = order.indexOf(mode);
        return order[(idx + 1) % order.length];
    }
    """,
    Output("theme-mode", "data"),
    Input("theme-toggle-btn", "n_clicks"),
    State("theme-mode", "data"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(mode) {
        function resolve(m) {
            if (m === "light" || m === "dark") { return m; }
            return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
        }
        var resolved = resolve(mode);
        document.documentElement.setAttribute("data-theme", resolved);
        var labels = {system: "Auto", light: "Claro", dark: "Oscuro"};
        return [resolved, labels[mode] || "Auto"];
    }
    """,
    Output("theme-resolved", "data"),
    Output("theme-toggle-btn", "children"),
    Input("theme-mode", "data"),
)


# ── CALLBACKS ────────────────────────────────────────────────────

@app.callback(
    Output("sidebar-body","className"),
    Output("mobile-menu-btn","children"),
    Input("mobile-menu-btn","n_clicks"),
    Input("nav-tab","value"),
    prevent_initial_call=True,
)
def toggle_mobile_menu(n_clicks, tab):
    if ctx.triggered_id == "nav-tab":
        return "sv-sidebar-body", "Menú"
    if n_clicks % 2 == 1:
        return "sv-sidebar-body sv-open", "Cerrar"
    return "sv-sidebar-body", "Menú"


@app.callback(
    Output("page-title","children"),
    Input("nav-tab","value"),
)
def update_title(tab):
    titles = {
        "tab-catalogo": "Composición del Catálogo",
        "tab-geo":       "Geografía e Idiomas",
        "tab-pop":       "Popularidad y Valoración",
    }
    return titles.get(tab, "StreamView Analytics")


@app.callback(
    Output("kpi-row","children"),
    Input("filtro-tipo","value"),
    Input("filtro-years","value"),
    Input("filtro-genero", "value"),
    Input("filtro-pais", "value"),
    Input("filtro-idioma", "value"),
    Input("cruces", "data"),
)
def update_kpis(tipo, years, genero=None, pais=None, idioma=None, cruces=None):
    df = filtrar(tipo, years, genero, pais, idioma, cruces)
    total    = len(df)
    pel      = len(df[df["type_label"]=="Película"])
    ser      = len(df[df["type_label"]=="Serie"])
    paises   = int(df["country"].dropna().str.split(", ").explode().nunique())
    idiomas  = int(df["language"].nunique())
    pop_avg  = df["popularity"].mean()
    val_avg  = df["vote_average"].mean()

    inicio, fin = years or (YEAR_MIN, YEAR_MAX)
    base_periodo = int(TITULOS_POR_ANIO.loc[inicio:fin].sum())
    porcentaje = total / base_periodo if base_periodo else 0
    cobertura = f"{porcentaje:.1%} de {base_periodo:,} títulos del período"
    kpis = [
        ("Total Contenidos",   f"{total:,}",       cobertura,        C_HIGHLIGHT),
        ("Películas",          f"{pel:,}",          f"{pel/total:.0%} del total" if total else "Sin títulos", C_MOVIE),
        ("Series",             f"{ser:,}",          f"{ser/total:.0%} del total" if total else "Sin títulos", C_SHOW),
        ("Países",             f"{paises}",          "países productores",       C_ACCENT4),
        ("Idiomas",            f"{idiomas}",         "idiomas disponibles",      C_ACCENT3),
        ("Popularidad Prom.",  f"{pop_avg:.1f}" if pd.notna(pop_avg) else "—",    "índice TMDB relativo",     C_TEXT),
        ("Calificación Prom.", f"{val_avg:.2f}/10" if pd.notna(val_avg) else "—", "escala 0–10",              C_TEXT),
    ]

    cols = []
    for label, val, sub, color in kpis:
        cols.append(
            dbc.Col(kpi_card(label, val, sub, color), xs=6, sm=4, md=3, style={"marginBottom":"0"})
        )
    activos = dict(cruces or [])
    filtros = [
        ("Tipo", tipo or "Todos"),
        ("País", " o ".join(seleccion(pais)) or "Todos"),
        ("Período", f"{inicio}–{fin}"),
        ("Género", " o ".join(seleccion(genero)) or "Todos"),
        ("Idioma", " o ".join(LANG_MAP.get(v, v) for v in seleccion(idioma)) or "Todos"),
    ]
    for campo, valor in activos.items():
        filtros.append(("Clic en país" if campo == "country" else "Clic en género", valor))
    filtrado = (tipo not in ("Todos", None) or (inicio, fin) != (YEAR_MIN, YEAR_MAX)
                or genero or pais or idioma or activos)
    resumen = [html.Strong("Segmento analizado · " + ("Filtrado" if filtrado else "Catálogo completo"))]
    resumen.extend(html.Span(f"{nombre}: {valor}") for nombre, valor in filtros)
    if df["country"].str.contains(",", regex=False, na=False).any():
        resumen.append(html.Span("Cada coproducción cuenta una vez en el total y una vez por país participante."))
    return html.Div([
        html.Div(resumen, style={"display": "flex", "flexWrap": "wrap", "gap": "4px 14px",
                                 "color": C_TEXT, "fontSize": "12px", "marginBottom": "12px"}),
        dbc.Row(cols, className="g-2"),
    ])


# ── CALLBACK: DESCARGA INFOGRAFÍA ────────────────────────────────
@app.callback(
    Output("modal-formato", "style"),
    Input("btn-descargar-infografia", "n_clicks"),
    Input("btn-cerrar-modal", "n_clicks"),
    Input("btn-dl-png", "n_clicks"),
    Input("btn-dl-pdf", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(*_):
    abierto = {"display":"block","position":"fixed","top":"70px","right":"24px","zIndex":"9999"}
    cerrado = {"display":"none","position":"fixed","top":"70px","right":"24px","zIndex":"9999"}
    if ctx.triggered_id == "btn-descargar-infografia":
        return abierto
    return cerrado

@app.callback(
    Output("download-infografia", "data"),
    Input("btn-dl-png", "n_clicks"),
    State("filtro-tipo", "value"),
    State("filtro-years", "value"),
    State("filtro-top", "value"),
    State("filtro-min-votos", "value"),
    State("filtro-genero", "value"),
    State("filtro-pais", "value"),
    State("filtro-idioma", "value"),
    State("cruces", "data"),
    prevent_initial_call=True,
)
def descargar_png(n_clicks, tipo, year_range, top_n, min_votos, genero=None, pais=None, idioma=None, cruces=None):
    png_b64 = generar_infografia_png(
        tipo, year_range, top_n, min_votos, genero, pais, idioma, cruces
    )
    png_bytes = base64.b64decode(png_b64)
    tipo_label = tipo.lower().replace("é","e").replace("í","i") if tipo and tipo != "Todos" else "catalogo"
    y0 = int(year_range[0]) if year_range else YEAR_MIN
    y1 = int(year_range[1]) if year_range else YEAR_MAX
    return dcc.send_bytes(png_bytes, f"streamview_resumen_{tipo_label}_{y0}_{y1}.png")

@app.callback(
    Output("download-infografia-pdf", "data"),
    Input("btn-dl-pdf", "n_clicks"),
    State("filtro-tipo", "value"),
    State("filtro-years", "value"),
    State("filtro-top", "value"),
    State("filtro-min-votos", "value"),
    State("filtro-genero", "value"),
    State("filtro-pais", "value"),
    State("filtro-idioma", "value"),
    State("cruces", "data"),
    prevent_initial_call=True,
)
def descargar_pdf(n_clicks, tipo, year_range, top_n, min_votos, genero=None, pais=None, idioma=None, cruces=None):
    from PIL import Image
    png_b64 = generar_infografia_png(
        tipo, year_range, top_n, min_votos, genero, pais, idioma, cruces
    )
    png_bytes = base64.b64decode(png_b64)
    img = Image.open(io.BytesIO(png_bytes))
    pdf_buf = io.BytesIO()
    img.save(pdf_buf, format="PDF", resolution=130)
    pdf_buf.seek(0)
    tipo_label = tipo.lower().replace("é","e").replace("í","i") if tipo and tipo != "Todos" else "catalogo"
    y0 = int(year_range[0]) if year_range else YEAR_MIN
    y1 = int(year_range[1]) if year_range else YEAR_MAX
    return dcc.send_bytes(pdf_buf.read(), f"streamview_resumen_{tipo_label}_{y0}_{y1}.pdf")


@app.callback(
    Output("filtro-genero", "value"), Output("filtro-pais", "value"),
    Output("filtro-idioma", "value"), Output("filtro-tipo", "value"),
    Output("filtro-years", "value"), Output("filtro-top", "value"),
    Output("filtro-min-votos", "value"), Output("filtro-min-titulos-genero", "value"),
    Input("restablecer-filtros", "n_clicks"), prevent_initial_call=True,
)
def restablecer_filtros(_):
    return [], [], [], "Todos", [YEAR_MIN, YEAR_MAX], 10, 30, 30


@app.callback(
    Output("cruces", "data"),
    Input({"rol": "cruce", "campo": ALL}, "clickData"),
    Input("deshacer-cruce", "n_clicks"), Input("restablecer-filtros", "n_clicks"),
    State("cruces", "data"), prevent_initial_call=True,
)
def actualizar_cruces(clics, deshacer, restablecer, historial):
    origen = ctx.triggered_id
    historial = historial or []
    if origen == "restablecer-filtros":
        return []
    if origen == "deshacer-cruce":
        return historial[:-1]
    if not isinstance(origen, dict):
        return no_update
    # Las gráficas se montan y desmontan al cambiar de vista.
    for entrada, clic in zip(ctx.inputs_list[0], clics):
        if entrada["id"] == origen and clic and clic.get("points"):
            campo = origen["campo"]
            valor = clic["points"][0].get("y")
            if valor in INDICES[campo] and dict(historial).get(campo) != valor:
                return historial + [[campo, valor]]
    return no_update


@app.callback(
    Output("deshacer-cruce", "disabled"),
    Input("cruces", "data"),
)
def mostrar_cruces(historial):
    return not bool(historial)


@app.callback(
    Output("store-cuadrante", "data"),
    Input({"rol": "selector-cuadrante", "vista": ALL}, "value"),
    Input("restablecer-filtros", "n_clicks"), State("store-cuadrante", "data"),
    prevent_initial_call=True,
)
def actualizar_cuadrante(valores, restablecer, anterior):
    if ctx.triggered_id == "restablecer-filtros":
        return "Todos"
    if valores and valores[0] and valores[0] != anterior:
        return valores[0]
    return no_update


# ── CALLBACK PRINCIPAL: TABS ─────────────────────────────────────
@app.callback(
    Output("tab-content","children"),
    Input("nav-tab","value"),
    Input("filtro-tipo","value"),
    Input("filtro-years","value"),
    Input("filtro-top","value"),
    Input("theme-resolved","data"),
    Input("filtro-min-votos","value"),
    Input("filtro-min-titulos-genero","value"),
    Input("store-cuadrante","data"),
    Input("filtro-genero", "value"),
    Input("filtro-pais", "value"),
    Input("filtro-idioma", "value"),
    Input("cruces", "data"),
)
def render_tab(tab, tipo, years, top_n, theme, min_votos, min_titulos_genero, cuadrante_sel, genero=None, pais=None, idioma=None, cruces=None):
    df = filtrar(tipo, years, genero, pais, idioma, cruces)

    if df.empty:
        return card(html.P("No hay títulos con esta combinación de filtros. Prueba quitar uno o ampliar los años."))

    tc = CHART_THEMES.get(theme, CHART_THEMES["dark"])
    BG_CHART, C_TEXT, C_MUTED, C_BORDER = tc["bg"], tc["text"], tc["muted"], tc["border"]

    # ══ CATÁLOGO ══════════════════════════════════════════════════
    if tab == "tab-catalogo":

        counts = df["type_label"].value_counts().reset_index()
        counts.columns = ["Tipo","Cantidad"]
        counts["Pct"] = (counts["Cantidad"]/counts["Cantidad"].sum()*100).round(1)
        counts["Label"] = counts.apply(lambda r: f"{r['Cantidad']:,} ({r['Pct']}%)", axis=1)

        fig_tipo = go.Figure()
        colors_tipo = [C_MOVIE if t=="Película" else C_SHOW for t in counts["Tipo"]]
        fig_tipo.add_trace(go.Bar(
            x=counts["Cantidad"], y=counts["Tipo"],
            orientation="h",
            marker=dict(color=colors_tipo, opacity=0.9, line=dict(color="rgba(0,0,0,0)", width=0)),
            text=counts["Label"], textposition="outside",
            textfont=dict(color=C_TEXT, size=11),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        ))
        lay = base_layout("¿Cómo se distribuye el catálogo?", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay["xaxis"]["title"] = "Cantidad de títulos"
        lay["yaxis"]["title"] = ""
        lay["xaxis"]["showgrid"] = False
        fig_tipo.update_layout(**lay, xaxis_range=[0, counts["Cantidad"].max()*1.22])

        rows_g = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_g.append({"Género":g,"Tipo":row["type_label"]})
        gdf = pd.DataFrame(rows_g, columns=['Género', 'Tipo'])
        top_g = gdf["Género"].value_counts().head(top_n).reset_index()
        top_g.columns = ["Género","Cantidad"]
        top_g = top_g.sort_values("Cantidad")

        fig_gen = go.Figure(go.Bar(
            x=top_g["Cantidad"], y=top_g["Género"],
            orientation="h",
            marker=dict(
                color=top_g["Cantidad"],
                colorscale=[[0, "rgba(124,58,237,0.35)"],[1, C_ACCENT1]],
                showscale=False, line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_g["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay2 = base_layout(f"Top {top_n} géneros — ¿Cuáles dominan el catálogo?", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay2["xaxis"]["title"] = "Cantidad de contenidos"
        lay2["yaxis"]["title"] = ""
        lay2["xaxis"]["showgrid"] = False
        lay2["margin"] = dict(l=8, r=60, t=44, b=8)
        lay2["title"]["text"] += " · Clic para filtrar"
        fig_gen.update_layout(**lay2, xaxis_range=[0, top_g["Cantidad"].max() * 1.18] if not top_g.empty else None)

        rc = df.groupby(["rating_bin","type_label"], observed=True).size().reset_index(name="Cantidad")
        fig_rat = go.Figure()
        for tipo_val, color in [("Película", C_MOVIE), ("Serie", C_SHOW)]:
            d = rc[rc["type_label"]==tipo_val]
            fig_rat.add_trace(go.Bar(
                name=tipo_val, x=d["rating_bin"].astype(str), y=d["Cantidad"],
                marker=dict(color=color, opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
                hovertemplate=f"{tipo_val} — %{{x}}: %{{y:,}}<extra></extra>",
            ))
        lay3 = base_layout("Distribución de calificaciones (0–10)", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay3["xaxis"]["title"] = "Rango de calificación"
        lay3["yaxis"]["title"] = "Cantidad de contenidos"
        lay3["barmode"] = "group"
        fig_rat.update_layout(**lay3)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_tipo, config={"displayModeBar":False,"responsive":True},
                             style={"height":"220px"}), padding="16px"), xs=12, lg=4),
                dbc.Col(card(dcc.Graph(figure=fig_rat,  config={"displayModeBar":False,"responsive":True},
                             style={"height":"220px"}), padding="16px"), xs=12, lg=8),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(id={"rol": "cruce", "campo": "genres"}, figure=fig_gen, config={"displayModeBar":False,"responsive":True},
                             style={"height":"320px"}), padding="16px"), width=12),
            ]),
        ])

    # ══ GEOGRAFÍA ═════════════════════════════════════════════════
    elif tab == "tab-geo":

        pais_clic = dict(cruces or []).get("country")
        destacados = set(seleccion(pais))
        if pais_clic:
            destacados.add(pais_clic)
        azul = "#205AB9" if theme == "light" else "#75A8FF"
        gris = "#737E8B" if theme == "light" else "#A8B2BE"
        unidad = {"Serie": "series", "Película": "películas"}.get(tipo, "títulos")
        foco = pais_clic or " o ".join(seleccion(pais))
        if len(foco) > 35:
            foco = f"{len(seleccion(pais))} países seleccionados"
        def marca_pais(valores):
            return dict(color=[azul if p in destacados else gris for p in valores],
                        line=dict(width=[2 if p == pais_clic else 0 for p in valores], color=C_TEXT))

        paises_s = df["country"].dropna().str.split(", ").explode().str.strip()

        rows_cg = []
        for _, row in df.dropna(subset=["country", "genres"]).iterrows():
            pop = row["popularity"]
            if pd.isna(pop):
                continue
            for c in row["country"].split(", "):
                c = c.strip()
                if c:
                    for g in row["genres"].split(", "):
                        g = g.strip()
                        if g:
                            rows_cg.append({"País": c, "Género": g, "popularity": pop})

        fig_gen_pais = go.Figure()
        if rows_cg:
            cgdf = pd.DataFrame(rows_cg)
            pais_count = paises_s.value_counts()
            paises_validos = pais_count[pais_count >= 5].index.tolist()
            cgdf = cgdf[cgdf["País"].isin(paises_validos)]

            if len(cgdf) > 0:
                pop_pais = (
                    cgdf.groupby("País")["popularity"]
                    .mean().reset_index()
                    .rename(columns={"popularity": "PopProm"})
                )
                gpc = (
                    cgdf.groupby(["País", "Género"])["popularity"]
                    .mean().reset_index()
                    .sort_values(["País", "popularity", "Género"], ascending=[True, False, True])
                )
                genero_dom = (
                    gpc.groupby("País").first().reset_index()[["País", "Género"]]
                )
                df_plot = pop_pais.merge(genero_dom, on="País")
                df_plot = df_plot.sort_values("PopProm", ascending=False).head(top_n)
                df_plot = df_plot.sort_values("PopProm")

                fig_gen_pais.add_trace(go.Bar(
                    x=df_plot["PopProm"], y=df_plot["País"], orientation="h",
                    marker=marca_pais(df_plot["País"]),
                    text=[f"{g}  ·  {v:.1f}" for g, v in zip(df_plot["Género"], df_plot["PopProm"])],
                    textposition="outside", textfont=dict(color=C_TEXT, size=9),
                    hovertemplate=(
                        "<b>%{y}</b><br>Popularidad promedio: %{x:.1f}<br>"
                        "Género dominante: %{customdata}<extra></extra>"
                    ),
                    customdata=df_plot["Género"],
                ))
                lay_gp = base_layout(f"Popularidad promedio por país — top {top_n}",
                                     bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
                lay_gp["xaxis"]["title"] = "Popularidad promedio (todos los títulos del país)"
                lay_gp["yaxis"]["title"] = ""
                lay_gp["xaxis"]["showgrid"] = True
                lay_gp["margin"] = dict(l=8, r=100, t=48, b=8)
                lay_gp["showlegend"] = False
                max_pop = df_plot["PopProm"].max() if len(df_plot) > 0 else 1
                fig_gen_pais.update_layout(**lay_gp, xaxis_range=[0, max_pop * 1.28])
            else:
                lay_gp = base_layout("Popularidad promedio por país",
                                     bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
                fig_gen_pais.update_layout(**lay_gp)
                fig_gen_pais.add_annotation(
                    text="Sin datos suficientes (mín. 5 títulos por país)",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(color=C_MUTED, size=13)
                )
        else:
            lay_gp = base_layout("Popularidad promedio por país",
                                 bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
            fig_gen_pais.update_layout(**lay_gp)
            fig_gen_pais.add_annotation(
                text="Sin datos disponibles",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=C_MUTED, size=13)
            )

        top_p = paises_s.value_counts().head(top_n).reset_index()
        top_p.columns = ["País", "Cantidad"]
        top_p = top_p.sort_values("Cantidad")

        fig_paises = go.Figure(go.Bar(
            x=top_p["Cantidad"], y=top_p["País"], orientation="h",
            marker=marca_pais(top_p["País"]),
            customdata=["País del clic" if p == pais_clic else "País seleccionado" if p in destacados
                        else "País coproductor" if destacados else "País productor" for p in top_p["País"]],
            text=[f"{n:,} coprod." if destacados and p not in destacados else f"{n:,}"
                  for p, n in zip(top_p["País"], top_p["Cantidad"])],
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y} · %{customdata}<br>%{x:,} títulos del segmento<extra></extra>",
        ))
        lay_p = base_layout(f"Top {top_n} países productores",
                            bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_p["xaxis"]["title"] = "Cantidad de títulos"
        lay_p["yaxis"]["title"] = ""
        lay_p["xaxis"]["showgrid"] = False
        lay_p["margin"] = dict(l=8, r=80, t=44, b=8)
        lay_p["title"]["text"] += " · Clic para filtrar"
        fig_paises.update_layout(**lay_p, xaxis_range=[0, top_p["Cantidad"].max() * 1.18] if not top_p.empty else None)

        lang_s = df["lang_label"].value_counts().head(top_n).reset_index()
        lang_s.columns = ["Idioma", "Cantidad"]
        lang_s = lang_s.sort_values("Cantidad")

        idiomas_activos = {LANG_MAP.get(v, v) for v in seleccion(idioma)}
        fig_lang = go.Figure(go.Bar(
            x=lang_s["Cantidad"], y=lang_s["Idioma"], orientation="h",
            marker=dict(color=[azul if etiqueta in idiomas_activos else gris for etiqueta in lang_s["Idioma"]]),
            text=lang_s["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay_l = base_layout(f"Top {top_n} idiomas disponibles",
                            bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        if foco:
            lay_l["title"]["text"] = f"Idiomas de {len(df):,} {unidad} · {foco} (top {top_n})"
        lay_l["xaxis"]["title"] = "Cantidad de contenidos"
        lay_l["yaxis"]["title"] = ""
        lay_l["xaxis"]["showgrid"] = False
        lay_l["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_lang.update_layout(**lay_l, xaxis_range=[0, lang_s["Cantidad"].max() * 1.18] if not lang_s.empty else None)

        nota = html.Div(
            ("Países seleccionados en azul · Coproductores adicionales en gris. " if destacados else "")
            + ("El borde indica el país elegido con un clic. " if pais_clic else "")
            + "Una coproducción cuenta en cada país participante. "
            "El género más popular se determina por popularidad promedio (mín. 5 títulos por país).",
            style={"fontSize": "10px", "color": C_MUTED, "marginTop": "10px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_gen_pais,
                             config={"displayModeBar": False, "responsive": True},
                             style={"height": "420px"}), padding="16px"), width=12),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(id={"rol": "cruce", "campo": "country"}, figure=fig_paises,
                             config={"displayModeBar": False, "responsive": True},
                             style={"height": "300px"}), padding="16px"), xs=12, lg=6),
                dbc.Col(card(dcc.Graph(figure=fig_lang,
                             config={"displayModeBar": False, "responsive": True},
                             style={"height": "300px"}), padding="16px"), xs=12, lg=6),
            ], className="g-3"),
            nota,
        ])

    # ══ POPULARIDAD ═══════════════════════════════════════════════
    elif tab == "tab-pop":

        min_votos = int(min_votos) if min_votos else 0
        min_gen   = int(min_titulos_genero) if min_titulos_genero else 30

        # ── Bump chart (existente) ───────────────────────────────
        rows_g_pop = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_g_pop.append({"Género": g, "popularity": row["popularity"]})
        gpdf = pd.DataFrame(rows_g_pop)

        insight_banner = None
        fig_bump = go.Figure()

        if len(gpdf) > 0 and gpdf["Género"].nunique() >= 2:
            vol = gpdf.groupby("Género").size().rename("Cantidad")
            pop = gpdf.groupby("Género")["popularity"].mean().rename("PopProm")
            rank_df = pd.concat([vol, pop], axis=1).reset_index()
            rank_df["rank_vol"] = rank_df["Cantidad"].rank(ascending=False, method="min").astype(int)
            rank_df["rank_pop"] = rank_df["PopProm"].rank(ascending=False, method="min").astype(int)
            n_generos = len(rank_df)

            n_mostrar = max(3, min(top_n, 6))
            plot_df = rank_df.nsmallest(n_mostrar, "rank_vol").sort_values("rank_vol")
            lider = plot_df.iloc[0]

            for _, r in plot_df.iterrows():
                es_lider = r["Género"] == lider["Género"]
                color_line = C_HIGHLIGHT if es_lider else C_MUTED
                width_line = 4 if es_lider else 1.5
                op = 0.95 if es_lider else 0.45
                if es_lider:
                    texto = [f"#{int(r['rank_vol'])} {r['Género']}", f"#{int(r['rank_pop'])}"]
                    pos_texto = ["middle left", "middle right"]
                else:
                    texto = [r["Género"], ""]
                    pos_texto = ["middle left", "middle right"]
                fig_bump.add_trace(go.Scatter(
                    x=["Ranking por volumen", "Ranking por popularidad"],
                    y=[r["rank_vol"], r["rank_pop"]],
                    mode="lines+markers+text",
                    line=dict(color=color_line, width=width_line),
                    marker=dict(color=color_line, size=7),
                    text=texto, textposition=pos_texto,
                    textfont=dict(color=color_line, size=12 if es_lider else 9.5),
                    opacity=op,
                    hovertemplate=f"<b>{r['Género']}</b><br>Puesto por volumen: #{int(r['rank_vol'])}<br>"
                                  f"Puesto por popularidad: #{int(r['rank_pop'])} de {n_generos}<extra></extra>",
                    showlegend=False,
                ))

            lay_bump = base_layout("El género que domina el catálogo, ¿domina también el interés del público?",
                                    bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
            lay_bump["yaxis"]["autorange"] = "reversed"
            lay_bump["yaxis"]["title"] = "Puesto en el ranking (1 = líder)"
            lay_bump["xaxis"]["showgrid"] = False
            lay_bump["margin"] = dict(l=150, r=70, t=44, b=8)
            lay_bump["xaxis"]["range"] = [-0.35, 1.15]
            fig_bump.update_layout(**lay_bump)

            salto = int(lider["rank_pop"] - lider["rank_vol"])
            if salto > 0:
                texto_banner = (f"{lider['Género']} lidera el catálogo en volumen (puesto #1) pero cae al puesto "
                                 f"#{int(lider['rank_pop'])} de {n_generos} en popularidad promedio.")
            else:
                texto_banner = (f"{lider['Género']} lidera tanto en volumen como en popularidad — "
                                 f"en este subconjunto filtrado, ambos rankings coinciden.")
            insight_banner = html.Div(texto_banner, style={
                "fontSize":"13px","fontWeight":"700","color":C_HIGHLIGHT,
                "padding":"10px 16px","marginBottom":"12px",
                "backgroundColor":"rgba(245,166,35,0.10)","borderRadius":"10px",
                "border":"1px solid rgba(245,166,35,0.35)",
            })

        bloque_hallazgo = []
        if insight_banner is not None:
            bloque_hallazgo = [
                insight_banner,
                dbc.Row([
                    dbc.Col(card(dcc.Graph(figure=fig_bump, config={"displayModeBar": False, "responsive": True},
                                 style={"height": "340px"}), padding="16px"), width=12),
                ], className="g-3 mb-3"),
            ]

        # ────────────────────────────────────────────────────────────
        # 1. SCATTER POPULARIDAD vs VALORACIÓN (mejorado)
        # ────────────────────────────────────────────────────────────

        # Elegibles: popularidad > 0, vote_average > 0, vote_count >= min_votos
        df_elig = df[
            (df["popularity"] > 0) &
            (df["vote_average"] > 0) &
            (df["vote_count"] > 0) &
            (df["vote_count"] >= min_votos)
        ].copy()

        n_excluidos_pop = int((df["popularity"] <= 0).sum())
        n_total_elig = len(df_elig)

        bloques_scatter = []

        if n_total_elig == 0:
            bloques_scatter.append(
                card(html.Div("Sin títulos elegibles con los filtros actuales.",
                              style={"color": C_MUTED, "padding": "40px", "textAlign": "center"}),
                     padding="16px")
            )
        else:
            # Medianas calculadas sobre el subconjunto elegible completo (no cambian con cuadrante)
            med_pop = float(df_elig["popularity"].median())
            med_val = float(df_elig["vote_average"].median())

            # Cuadrante: >= mediana → "alto" (regla consistente: igual a mediana → alto)
            df_elig = df_elig.copy()
            df_elig["pop_alta"] = df_elig["popularity"] >= med_pop
            df_elig["val_alta"] = df_elig["vote_average"] >= med_val

            def nombre_cuadrante(row):
                if row["pop_alta"] and row["val_alta"]:
                    return "1 · Destacar"
                elif not row["pop_alta"] and row["val_alta"]:
                    return "2 · Descubrimiento"
                elif row["pop_alta"] and not row["val_alta"]:
                    return "3 · Revisar expectativas"
                else:
                    return "4 · Menor prioridad"

            df_elig["cuadrante"] = df_elig.apply(nombre_cuadrante, axis=1)

            # Verificar que los cuatro cuadrantes tienen datos
            cuadrantes_con_datos = df_elig["cuadrante"].nunique()
            if cuadrantes_con_datos < 4:
                aviso_cuad = html.Div(
                    f"ℹ Con los filtros actuales solo se forman {cuadrantes_con_datos} cuadrante(s). "
                    "Ajusta el mínimo de votos o el rango de años.",
                    style={"fontSize":"10px","color":C_MUTED,"marginBottom":"8px"}
                )
            else:
                aviso_cuad = None

            # Filtrar para resaltar cuadrante seleccionado
            if cuadrante_sel and cuadrante_sel != "Todos":
                df_resaltado = df_elig[df_elig["cuadrante"] == cuadrante_sel]
                df_atenuado  = df_elig[df_elig["cuadrante"] != cuadrante_sel]
            else:
                df_resaltado = df_elig
                df_atenuado  = pd.DataFrame(columns=df_elig.columns)

            colores_accion = {clave: oscuro if theme == "dark" else claro
                              for clave, (nombre, claro, oscuro) in ACCIONES.items()}
            simbolos = {"Película": "circle", "Serie": "diamond"}

            fig_sc = go.Figure()

            # Puntos atenuados (gris, todos los formatos juntos)
            if len(df_atenuado) > 0:
                fig_sc.add_trace(go.Scattergl(
                    x=df_atenuado["popularity"],
                    y=df_atenuado["vote_average"],
                    mode="markers",
                    name="Otros cuadrantes",
                    marker=dict(color=df_atenuado["cuadrante"].map(colores_accion),
                                symbol=df_atenuado["type_label"].map(simbolos), size=5, opacity=0.18,
                                line=dict(color="rgba(0,0,0,0)")),
                    hoverinfo="skip",
                    showlegend=False,
                ))

            # Puntos resaltados por formato
            for tipo_val, simbolo in simbolos.items():
                d = df_resaltado[df_resaltado["type_label"] == tipo_val]
                if len(d) == 0:
                    continue
                genres_str = d["genres"].fillna("—")
                fig_sc.add_trace(go.Scattergl(
                    x=d["popularity"],
                    y=d["vote_average"],
                    mode="markers",
                    name=f"{tipo_val} ({'círculo' if simbolo == 'circle' else 'rombo'})",
                    marker=dict(color=d["cuadrante"].map(colores_accion), symbol=simbolo, size=6, opacity=0.9,
                                line=dict(color="rgba(0,0,0,0)")),
                    customdata=np.column_stack([
                        d["type_label"].values,
                        genres_str.values,
                        d["release_year"].fillna("—").astype(str).values,
                        d["vote_count"].fillna(0).astype(int).values,
                        d["cuadrante"].values,
                    ]),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Formato: %{customdata[0]}<br>"
                        "Géneros: %{customdata[1]}<br>"
                        "Estreno: %{customdata[2]}<br>"
                        "Popularidad: %{x:.1f}<br>"
                        "Calificación: %{y:.2f}/10<br>"
                        "Votos: %{customdata[3]:,}<br>"
                        "Cuadrante: %{customdata[4]}<extra></extra>"
                    ),
                    text=d["title"].values,
                ))

            # Líneas de mediana
            fig_sc.add_hline(
                y=med_val, line_dash="dot", line_color=C_MUTED, opacity=0.6,
                annotation_text=f"Med. cal. {med_val:.2f}",
                annotation_font=dict(size=9, color=C_MUTED),
            )
            fig_sc.add_vline(
                x=med_pop, line_dash="dot", line_color=C_MUTED, opacity=0.6,
                annotation_text=f"Med. pop. {med_pop:.1f}",
                annotation_font=dict(size=9, color=C_MUTED),
            )

            lay_sc = base_layout(
                f"Popularidad vs. Valoración — {n_total_elig:,} títulos elegibles "
                f"(mín. {min_votos} votos) · medianas relativas al subconjunto",
                bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER
            )
            lay_sc["xaxis"]["title"] = "Popularidad TMDB (escala logarítmica)"
            lay_sc["xaxis"]["type"] = "log"
            lay_sc["yaxis"]["title"] = "Calificación (0–10)"
            lay_sc["yaxis"]["range"] = [0, 10.5]
            lay_sc["showlegend"] = True
            lay_sc["margin"] = dict(l=8, r=16, t=52, b=8)
            fig_sc.update_layout(**lay_sc)

            # Etiquetas de cuadrantes en las esquinas
            cuad_labels = [
                (0.98, 0.98, "Alta pop · Alta cal\nCandidatos para destacar",  "right", "top"),
                (0.02, 0.98, "Baja pop · Alta cal\nExplorar descubrimiento", "left",  "top"),
                (0.98, 0.02, "Alta pop · Baja cal\nRevisar expectativas",       "right", "bottom"),
                (0.02, 0.02, "Baja pop · Baja cal\nMenor prioridad inicial",    "left",  "bottom"),
            ]
            for xr, yr, txt, xa, ya in cuad_labels:
                fig_sc.add_annotation(
                    xref="paper", yref="paper", x=xr, y=yr, text=txt,
                    showarrow=False, xanchor=xa, yanchor=ya,
                    font=dict(size=8, color=C_MUTED),
                    bgcolor="rgba(0,0,0,0)",
                )

            # Selector de cuadrante
            opciones_cuad = [
                {"label": "Todos", "value": "Todos"},
                {"label": "1 · Destacar",             "value": "1 · Destacar"},
                {"label": "2 · Descubrimiento",       "value": "2 · Descubrimiento"},
                {"label": "3 · Revisar expectativas", "value": "3 · Revisar expectativas"},
                {"label": "4 · Menor prioridad",      "value": "4 · Menor prioridad"},
            ]

            selector_cuad = html.Div([
                html.Span("Cuadrante: ", style={"fontSize":"11px","color":C_MUTED,"marginRight":"8px"}),
                dcc.Dropdown(
                    id={"rol": "selector-cuadrante", "vista": "pop"},
                    options=opciones_cuad,
                    value=cuadrante_sel if cuadrante_sel else "Todos",
                    clearable=False,
                    style={"width":"220px","fontSize":"11px","display":"inline-block","verticalAlign":"middle"},
                ),
            ], style={"display":"flex","alignItems":"center","marginBottom":"8px"})

            # Tabla vinculada
            if cuadrante_sel and cuadrante_sel != "Todos":
                df_tabla = df_resaltado.copy()
                n_cuad = len(df_tabla)
            else:
                df_tabla = df_elig.copy()
                n_cuad = n_total_elig

            df_tabla_show = df_tabla[[
                "title","type_label","genres","release_year",
                "popularity","vote_average","vote_count","cuadrante"
            ]].copy()
            df_tabla_show.columns = [
                "Título","Formato","Géneros","Estreno",
                "Popularidad","Calificación","Votos","Cuadrante"
            ]
            df_tabla_show["Popularidad"] = df_tabla_show["Popularidad"].round(1)
            df_tabla_show["Calificación"] = df_tabla_show["Calificación"].round(2)
            df_tabla_show["Votos"] = df_tabla_show["Votos"].fillna(0).astype(int)
            df_tabla_show["Estreno"] = df_tabla_show["Estreno"].fillna("—").astype(str).str.replace(".0","",regex=False)
            df_tabla_show["Géneros"] = df_tabla_show["Géneros"].fillna("—")
            df_tabla_show = df_tabla_show.sort_values("Popularidad", ascending=False)

            tabla_style_header = {
                "backgroundColor": BG_CHART,
                "color": C_TEXT,
                "fontWeight": "600",
                "fontSize": "11px",
                "border": f"1px solid {C_BORDER}",
                "fontFamily": FONT,
            }
            tabla_style_cell = {
                "backgroundColor": BG_CHART,
                "color": C_TEXT,
                "fontSize": "11px",
                "border": f"1px solid {C_BORDER}",
                "fontFamily": FONT,
                "textOverflow": "ellipsis",
                "overflow": "hidden",
                "maxWidth": "200px",
            }

            conteo_label = html.Div(
                f"Mostrando {n_cuad:,} de {n_total_elig:,} títulos elegibles"
                + (f" · cuadrante: {cuadrante_sel}" if cuadrante_sel != "Todos" else ""),
                style={"fontSize":"10px","color":C_MUTED,"marginBottom":"6px","marginTop":"12px"}
            )

            tabla_comp = dash_table.DataTable(
                data=df_tabla_show.to_dict("records"),
                columns=[{"name": c, "id": c, "deletable": False} for c in df_tabla_show.columns],
                page_size=10,
                sort_action="native",
                filter_action="none",
                style_header=tabla_style_header,
                style_cell=tabla_style_cell,
                style_data_conditional=[
                    {"if": {"row_index": "odd"}, "backgroundColor": BG_CHART},
                ],
                style_table={"overflowX": "auto"},
            )

            nota_sc = html.Div([
                html.Span(
                    f"⚠ Popularidad TMDB ≠ reproducciones ni audiencia. "
                    f"Calificaciones reflejan quienes votaron, no todos los espectadores. "
                    f"Dataset muestreado (1.000 registros/año/formato). "
                    + (f"{n_excluidos_pop} títulos excluidos por popularidad ≤ 0. " if n_excluidos_pop > 0 else "")
                    + "Los cuadrantes orientan la selección para pruebas de marketing, no predicen el resultado de una campaña."
                )
            ], style={"fontSize":"10px","color":C_MUTED,"marginTop":"8px","lineHeight":"1.5"})

            scatter_bloque = [
                card([
                    selector_cuad,
                    aviso_cuad or html.Span(),
                    html.Div([
                        html.Span([html.Span("● ", style={"color": colores_accion[clave]}), datos[0]])
                        for clave, datos in ACCIONES.items()
                    ],
                       style={"display": "flex", "flexWrap": "wrap", "gap": "4px 14px",
                              "fontSize": "12px", "color": C_TEXT}),
                    html.P(NOTA_MEDIANAS, style={"fontSize": "12px", "color": C_TEXT, "margin": "6px 0"}),
                    dcc.Graph(figure=fig_sc, config={"displayModeBar": False, "responsive": True},
                              style={"height": "420px"}),
                    conteo_label,
                    tabla_comp,
                    nota_sc,
                ], padding="16px")
            ]
            bloques_scatter = scatter_bloque

        # ────────────────────────────────────────────────────────────
        # 2. MATRIZ DE OPORTUNIDADES POR GÉNERO
        # ────────────────────────────────────────────────────────────

        VALID_GENRES_EXCL = {"Unknown", "unknown", ""}

        def construir_genero_df(sub_df):
            """
            Por cada género, cuenta títulos únicos (por uid) y calcula mediana de popularidad.
            Excluye géneros vacíos y 'Unknown'. Filtra por min_gen.
            Retorna DataFrame con columnas: Género, N_titulos, Med_pop.
            """
            rows = []
            seen = {}  # género -> set de uid
            for _, row in sub_df.dropna(subset=["genres"]).iterrows():
                uid = row["uid"]
                for g in row["genres"].split(", "):
                    g = g.strip()
                    if not g or g in VALID_GENRES_EXCL:
                        continue
                    if g not in seen:
                        seen[g] = []
                    seen[g].append(row["popularity"])

            rows = []
            for g, pops in seen.items():
                rows.append({
                    "Género": g,
                    "N_titulos": len(pops),
                    "Med_pop": float(np.median(pops)),
                })
            gdf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Género","N_titulos","Med_pop"])
            if len(gdf) > 0:
                gdf = gdf[gdf["N_titulos"] >= min_gen]
            return gdf

        # Títulos sin género válido
        sin_genero = df[
            df["genres"].isna() |
            df["genres"].str.strip().isin(VALID_GENRES_EXCL)
        ]
        n_sin_genero = len(sin_genero)

        # Determinar qué formatos mostrar
        tipos_mostrar = []
        if tipo == "Todos":
            tipos_mostrar = [("Película", C_MOVIE), ("Serie", C_SHOW)]
        elif tipo == "Película":
            tipos_mostrar = [("Película", C_MOVIE)]
        else:
            tipos_mostrar = [("Serie", C_SHOW)]

        # Calcular rangos globales para escalas compartidas
        all_gdfs = {}
        for tipo_val, _ in tipos_mostrar:
            sub = df[df["type_label"] == tipo_val]
            gdf_t = construir_genero_df(sub)
            all_gdfs[tipo_val] = gdf_t

        non_empty = [g for g in all_gdfs.values() if len(g) > 0]
        if non_empty:
            x_max_global = max(g["N_titulos"].max() for g in non_empty) * 1.15
            y_min_global = min(g["Med_pop"].min() for g in non_empty) * 0.9
            y_max_global = max(g["Med_pop"].max() for g in non_empty) * 1.12
        else:
            x_max_global = y_min_global = y_max_global = 1

        cols_matriz = []
        for tipo_val, color in tipos_mostrar:
            gdf_t = all_gdfs[tipo_val]

            fig_mat = go.Figure()

            if len(gdf_t) == 0:
                lay_m = base_layout(
                    f"Matriz de oportunidades — {tipo_val}s",
                    bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER
                )
                fig_mat.update_layout(**lay_m)
                fig_mat.add_annotation(
                    text=f"Sin géneros con ≥ {min_gen} títulos en el filtro actual.",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(color=C_MUTED, size=12)
                )
                cols_matriz.append(
                    dbc.Col(card(dcc.Graph(figure=fig_mat, config={"displayModeBar":False,"responsive":True},
                                          style={"height":"380px"}), padding="16px"),
                            xs=12, lg=12 if len(tipos_mostrar)==1 else 6)
                )
                continue

            # Medianas para las líneas divisorias (relativas a este formato)
            med_n   = float(gdf_t["N_titulos"].median())
            med_pop = float(gdf_t["Med_pop"].median())

            # Verificar si las medianas permiten 4 zonas útiles
            n_zonas = gdf_t.apply(
                lambda r: (int(r["N_titulos"] >= med_n), int(r["Med_pop"] >= med_pop)), axis=1
            ).apply(tuple).nunique()

            aviso_zonas = None
            if n_zonas < 4:
                aviso_zonas = html.Div(
                    f"ℹ Con los filtros actuales solo se forman {n_zonas} zona(s) distintas.",
                    style={"fontSize":"9px","color":C_MUTED,"marginBottom":"4px"}
                )

            fig_mat.add_trace(go.Scatter(
                x=gdf_t["N_titulos"],
                y=gdf_t["Med_pop"],
                mode="markers+text",
                text=gdf_t["Género"],
                textposition="top center",
                textfont=dict(size=8, color=C_TEXT),
                marker=dict(
                    color=gdf_t["Med_pop"],
                    colorscale=[[0, f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.35)"],
                                [1, color]],
                    size=10,
                    showscale=False,
                    line=dict(color=color, width=1),
                ),
                customdata=np.column_stack([
                    gdf_t["N_titulos"].values,
                    gdf_t["Med_pop"].round(1).values,
                    np.full(len(gdf_t), tipo_val),
                ]),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Formato: %{customdata[2]}<br>"
                    "Títulos: %{customdata[0]:,}<br>"
                    "Popularidad mediana: %{customdata[1]}<extra></extra>"
                ),
            ))

            # Líneas divisorias
            fig_mat.add_vline(
                x=med_n, line_dash="dot", line_color=C_MUTED, opacity=0.5,
                annotation_text=f"Med. títulos {int(med_n)}",
                annotation_font=dict(size=8, color=C_MUTED),
            )
            fig_mat.add_hline(
                y=med_pop, line_dash="dot", line_color=C_MUTED, opacity=0.5,
                annotation_text=f"Med. pop. {med_pop:.1f}",
                annotation_font=dict(size=8, color=C_MUTED),
            )

            lay_m = base_layout(
                f"Matriz de oportunidades — {tipo_val}s (mín. {min_gen} títulos/género)",
                bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER
            )
            lay_m["xaxis"]["title"] = "Títulos en el catálogo (un título puede estar en varios géneros)"
            lay_m["yaxis"]["title"] = "Popularidad mediana TMDB"
            lay_m["xaxis"]["range"] = [0, x_max_global]
            lay_m["yaxis"]["range"] = [y_min_global, y_max_global]
            lay_m["margin"] = dict(l=8, r=16, t=52, b=8)
            fig_mat.update_layout(**lay_m)

            # Etiquetas de zonas
            zona_labels = [
                (0.98, 0.98, "Muchos títulos\nAlta popularidad",  "right", "top"),
                (0.02, 0.98, "Pocos títulos\nAlta popularidad",   "left",  "top"),
                (0.98, 0.02, "Muchos títulos\nBaja popularidad",  "right", "bottom"),
                (0.02, 0.02, "Pocos títulos\nBaja popularidad",   "left",  "bottom"),
            ]
            for xr, yr, txt, xa, ya in zona_labels:
                fig_mat.add_annotation(
                    xref="paper", yref="paper", x=xr, y=yr, text=txt,
                    showarrow=False, xanchor=xa, yanchor=ya,
                    font=dict(size=8, color=C_MUTED),
                )

            cols_matriz.append(
                dbc.Col([
                    aviso_zonas or html.Span(),
                    card(dcc.Graph(figure=fig_mat, config={"displayModeBar":False,"responsive":True},
                                  style={"height":"380px"}), padding="16px"),
                ], xs=12, lg=12 if len(tipos_mostrar)==1 else 6)
            )

        nota_matriz = html.Div([
            html.Span(
                f"ℹ 'Alto' y 'bajo' son relativos al subconjunto seleccionado, no al mercado general. "
                f"Pocos títulos + alta popularidad es una señal para explorar, no evidencia de demanda insatisfecha. "
                f"Un título puede estar en varios géneros; los conteos por género no son sumables como títulos únicos. "
                + (f"{n_sin_genero:,} título(s) sin género válido excluidos. " if n_sin_genero > 0 else "")
                + "Popularidad TMDB no equivale a reproducciones, audiencia ni conversión. "
                "Estos gráficos orientan la selección de contenidos para pruebas de marketing, no predicen el resultado de una campaña."
            )
        ], style={"fontSize":"10px","color":C_MUTED,"marginTop":"8px","lineHeight":"1.5"})

        # ── Gráficos existentes (top populares y géneros por popularidad) ──

        top_pop_movies = df[(df["vote_average"]>0) & (df["type_label"]=="Película")].nlargest(top_n,"popularity")[
            ["title","popularity","vote_average","type_label"]
        ]
        top_pop_shows = df[(df["vote_average"]>0) & (df["type_label"]=="Serie")].nlargest(top_n,"popularity")[
            ["title","popularity","vote_average","type_label"]
        ]
        top_pop = pd.concat([top_pop_movies, top_pop_shows], ignore_index=True)
        fig_pop = go.Figure()
        for tipo_val, color in [("Película",C_MOVIE),("Serie",C_SHOW)]:
            d = top_pop[top_pop["type_label"]==tipo_val].sort_values("popularity")
            if len(d)==0: continue
            fig_pop.add_trace(go.Bar(
                x=d["popularity"], y=d["title"].str[:22],
                orientation="h", name=tipo_val,
                marker=dict(color=color, opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
                hovertemplate="%{y}<br>Popularidad: %{x:.1f}<extra></extra>",
            ))
        lay_pop = base_layout(f"Top {top_n} más populares por formato", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_pop["xaxis"]["title"] = "Índice de popularidad"
        lay_pop["yaxis"]["title"] = ""
        lay_pop["barmode"] = "stack"
        lay_pop["xaxis"]["showgrid"] = False
        fig_pop.update_layout(**lay_pop)

        rows_pg = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_pg.append({"Género":g,"popularity":row["popularity"]})
        pgdf = pd.DataFrame(rows_pg, columns=['Género', 'popularity'])
        top_pg = pgdf.groupby("Género")["popularity"].mean().nlargest(top_n).reset_index()
        top_pg.columns = ["Género","Popularidad promedio"]
        top_pg = top_pg.sort_values("Popularidad promedio")

        fig_pg = go.Figure(go.Bar(
            x=top_pg["Popularidad promedio"], y=top_pg["Género"],
            orientation="h",
            marker=dict(
                color=top_pg["Popularidad promedio"],
                colorscale=[[0,"rgba(16,185,129,0.3)"],[1,C_ACCENT4]],
                showscale=False, line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_pg["Popularidad promedio"].round(1),
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ))
        lay_pg = base_layout(f"Top {top_n} géneros por popularidad promedio", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_pg["xaxis"]["title"] = "Popularidad promedio (TMDB)"
        lay_pg["yaxis"]["title"] = ""
        lay_pg["xaxis"]["showgrid"] = False
        lay_pg["margin"] = dict(l=8, r=50, t=44, b=8)
        fig_pg.update_layout(**lay_pg, xaxis_range=[0, top_pg["Popularidad promedio"].max() * 1.18] if not top_pg.empty else None)

        rows_cp = []
        for _, row in df.dropna(subset=["country"]).iterrows():
            for c in row["country"].split(", "):
                rows_cp.append({"País": c.strip(), "vote_average": row["vote_average"]})
        cpdf = pd.DataFrame(rows_cp, columns=['País', 'vote_average'])
        country_stats = cpdf.groupby("País")["vote_average"].agg(["mean","count"]).reset_index()
        country_stats = country_stats[country_stats["count"] >= 30]
        top_cp = country_stats.nlargest(top_n, "mean")[["País","mean"]].copy()
        top_cp.columns = ["País","Calificación promedio"]
        top_cp = top_cp.sort_values("Calificación promedio")

        fig_cp = go.Figure(go.Bar(
            x=top_cp["Calificación promedio"], y=top_cp["País"],
            orientation="h",
            marker=dict(
                color=top_cp["Calificación promedio"],
                colorscale=[[0, "rgba(6,182,212,0.3)"],[1, C_ACCENT2]],
                showscale=False, line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_cp["Calificación promedio"].round(2),
            textposition="outside", textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        ))
        lay_cp = base_layout(f"Top {top_n} países con mejor calificación promedio (mín. 30 títulos)", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_cp["xaxis"]["title"] = "Calificación promedio (0–10)"
        lay_cp["yaxis"]["title"] = ""
        lay_cp["xaxis"]["showgrid"] = False
        lay_cp["margin"] = dict(l=8, r=50, t=44, b=8)
        fig_cp.update_layout(**lay_cp, xaxis_range=[0, 10.5])

        nota_cp = html.Div(
            "ℹ vote_average en escala 0–10 (TMDB). Solo países con al menos 30 contenidos en el filtro actual.",
            style={"fontSize":"10px","color":C_MUTED,"marginTop":"10px"}
        )

        # ── Título de sección para matriz ──
        titulo_matriz = html.Div([
            html.Span("Matriz de oportunidades por género", style={
                "fontSize":"14px","fontWeight":"700","color":C_TEXT
            }),
            html.Span(
                "  ·  Eje X: títulos en catálogo · Eje Y: popularidad mediana · "
                "Líneas = medianas del subconjunto · No aplica el filtro Top N",
                style={"fontSize":"10px","color":C_MUTED}
            ),
        ], style={"marginBottom":"8px","marginTop":"8px"})

        titulo_scatter = html.Div([
            html.Span("Popularidad vs. Valoración", style={
                "fontSize":"14px","fontWeight":"700","color":C_TEXT
            }),
            html.Span(
                "  ·  Escala log en X · Todos los títulos elegibles · No aplica el filtro Top N",
                style={"fontSize":"10px","color":C_MUTED}
            ),
        ], style={"marginBottom":"8px","marginTop":"8px"})

        return html.Div([
            *bloque_hallazgo,

            # Scatter mejorado
            titulo_scatter,
            dbc.Row([
                dbc.Col(bloques_scatter, width=12),
            ], className="g-3 mb-3"),

            # Matriz de oportunidades
            titulo_matriz,
            dbc.Row(cols_matriz, className="g-3 mb-2"),
            nota_matriz,

            # Gráficos existentes
            html.Div(style={"marginTop":"16px"}),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_pop, config={"displayModeBar":False,"responsive":True},
                             style={"height":"320px"}), padding="16px"), xs=12, lg=5),
                dbc.Col(card(dcc.Graph(figure=fig_pg, config={"displayModeBar":False,"responsive":True},
                             style={"height":"320px"}), padding="16px"), xs=12, lg=7),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_cp, config={"displayModeBar":False,"responsive":True},
                             style={"height":"280px"}), padding="16px"), width=12),
            ], className="mt-3"),
            nota_cp,
        ])

    return html.Div()


if __name__ == "__main__":
    print("=" * 55)
    print("  StreamView Analytics — Dashboard EP1 (Dark)")
    print("  Abrir en: http://127.0.0.1:8050")
    print("=" * 55)
    app.run(debug=False, host="127.0.0.1", port=8050)