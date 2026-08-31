"""
=============================================================================
STREAMVIEW ANALYTICS — EP1 | DASHBOARD INTERACTIVO
Visual Analytics para la Toma de Decisiones Estratégicas
ADY1104 — Visualización de Datos | Duoc UC 2026
=============================================================================
Ejecutar: python app.py   →   http://127.0.0.1:8050
=============================================================================
"""

import os, warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PALETA — simple, consistente, sin exceso de color
# ─────────────────────────────────────────────────────────────────
C_MOVIE   = "#E50914"   # rojo  → películas
C_SHOW    = "#1565C0"   # azul  → series
C_NEUTRAL = "#37474F"   # gris oscuro → texto/KPIs neutros
C_BG      = "#F5F5F5"   # fondo general
C_CARD    = "#FFFFFF"   # fondo tarjeta
C_BORDER  = "#E0E0E0"   # borde sutil
C_MUTED   = "#78909C"   # texto secundario

FONT = "Inter, system-ui, sans-serif"

LANG_MAP = {
    "en":"Inglés","fr":"Francés","ja":"Japonés","ko":"Coreano","es":"Español",
    "zh":"Chino","it":"Italiano","hi":"Hindi","de":"Alemán","ru":"Ruso",
    "tl":"Filipino","ar":"Árabe","pt":"Portugués","nl":"Holandés","tr":"Turco"
}

# ─────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

movies = pd.read_csv(os.path.join(DATA_DIR, "netflix_movies_detailed_up_to_2025.csv"))
shows  = pd.read_csv(os.path.join(DATA_DIR, "netflix_tv_shows_detailed_up_to_2025.csv"))

for df, label in [(movies, "Película"), (shows, "Serie")]:
    df["date_added"]  = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"]  = df["date_added"].dt.year
    df["type_label"]  = label
    df["lang_label"]  = df["language"].map(LANG_MAP).fillna(df["language"])
    # Calificación en rangos legibles (vote_average / rating son iguales aquí)
    df["rating_bin"]  = pd.cut(
        df["vote_average"],
        bins=[0, 4, 5, 6, 7, 8, 10],
        labels=["< 4","4–5","5–6","6–7","7–8","8–10"],
        right=True
    )

COMMON = ["show_id","type","type_label","title","country","year_added",
          "release_year","genres","language","lang_label",
          "popularity","vote_count","vote_average","rating_bin"]
combined = pd.concat([movies[COMMON], shows[COMMON]], ignore_index=True)

# Valores para filtros — Python int nativo
YEAR_MIN = int(combined["year_added"].min())
YEAR_MAX = int(combined["year_added"].max())
YEARS    = list(range(YEAR_MIN, YEAR_MAX + 1))

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def filtrar(tipo="Todos", year_range=None):
    df = combined.copy()
    if tipo != "Todos":
        df = df[df["type_label"] == tipo]
    if year_range:
        df = df[
            (df["year_added"] >= year_range[0]) &
            (df["year_added"] <= year_range[1])
        ]
    return df

def layout_grafico():
    return dict(
        plot_bgcolor=C_CARD,
        paper_bgcolor=C_CARD,
        font=dict(family=FONT, size=12, color=C_NEUTRAL),
        margin=dict(l=10, r=20, t=40, b=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11)
        ),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

def kpi(titulo, valor, subtitulo="", color=C_NEUTRAL, grande=False):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.P(titulo, style={
                    "color": C_MUTED, "fontSize": "11px",
                    "fontWeight": "600", "textTransform": "uppercase",
                    "letterSpacing": "0.5px", "marginBottom": "4px"
                }),
                html.H2(valor, style={
                    "color": color,
                    "fontWeight": "800",
                    "fontSize": "28px" if grande else "22px",
                    "marginBottom": "2px", "lineHeight": "1"
                }),
                html.P(subtitulo, style={
                    "color": C_MUTED, "fontSize": "10px", "marginBottom": "0"
                }) if subtitulo else html.Div(),
            ], style={"padding": "14px 16px"})
        ], style={
            "borderRadius": "8px", "border": f"1px solid {C_BORDER}",
            "backgroundColor": C_CARD, "height": "100%"
        }),
        style={"marginBottom": "0"}
    )

# ─────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap"
    ],
    title="StreamView Analytics — EP1"
)

# ── HEADER ──────────────────────────────────────────────────────
header = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Span("🎬 ", style={"fontSize": "20px"}),
                html.Span("StreamView Analytics", style={
                    "fontWeight": "800", "fontSize": "18px",
                    "color": "white", "letterSpacing": "-0.3px"
                }),
                html.Span("  EP1 — Visual Analytics", style={
                    "color": "rgba(255,255,255,0.55)",
                    "fontSize": "13px", "marginLeft": "8px"
                }),
            ], width="auto"),
            dbc.Col([
                html.Span("ADY1104 · Visualización de Datos · Duoc UC 2026", style={
                    "color": "rgba(255,255,255,0.45)", "fontSize": "12px"
                })
            ], className="text-end"),
        ], align="center")
    ], fluid=True)
], style={
    "backgroundColor": C_NEUTRAL, "padding": "12px 0",
    "marginBottom": "0", "fontFamily": FONT
})

# ── FILTROS (compactos, barra lateral delgada) ───────────────────
sidebar = html.Div([
    html.P("FILTROS", style={
        "fontSize": "10px", "fontWeight": "700", "color": C_MUTED,
        "letterSpacing": "1px", "marginBottom": "12px", "marginTop": "4px"
    }),

    html.Label("Tipo de contenido", style={"fontSize": "12px", "fontWeight": "600", "color": C_NEUTRAL}),
    dcc.RadioItems(
        id="filtro-tipo",
        options=[{"label": f"  {t}", "value": t} for t in ["Todos", "Película", "Serie"]],
        value="Todos",
        labelStyle={"display": "block", "marginBottom": "5px", "fontSize": "13px"},
        inputStyle={"marginRight": "6px"},
        style={"marginBottom": "20px"}
    ),

    html.Label("Años de incorporación", style={"fontSize": "12px", "fontWeight": "600", "color": C_NEUTRAL}),
    dcc.RangeSlider(
        id="filtro-years",
        min=YEAR_MIN, max=YEAR_MAX,
        value=[YEAR_MIN, YEAR_MAX],
        marks={2010: "2010", 2015: "2015", 2020: "2020", 2025: "2025"},
        step=1,
        tooltip={"placement": "bottom", "always_visible": False},
        style={"marginBottom": "20px"}
    ),

    html.Label("Ítems en rankings", style={"fontSize": "12px", "fontWeight": "600", "color": C_NEUTRAL}),
    dcc.Slider(
        id="filtro-top", min=5, max=20, step=5, value=10,
        marks={5: "5", 10: "10", 15: "15", 20: "20"},
        style={"marginBottom": "24px"}
    ),

    html.Hr(style={"borderColor": C_BORDER, "margin": "0 0 12px 0"}),
    html.P(
        "⚠ Popularidad = índice relativo TMDB. No equivale a reproducciones.",
        style={"fontSize": "10px", "color": C_MUTED, "lineHeight": "1.4"}
    ),
    html.P(
        "ℹ Dataset: 1.000 registros por año (2010–2025).",
        style={"fontSize": "10px", "color": C_MUTED, "lineHeight": "1.4"}
    ),
], style={
    "backgroundColor": C_CARD,
    "border": f"1px solid {C_BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "fontFamily": FONT,
    "position": "sticky",
    "top": "16px"
})

# ── TABS ────────────────────────────────────────────────────────
tabs = dbc.Tabs([
    dbc.Tab(label="Catálogo",                tab_id="tab-catalogo",  label_style={"fontSize": "13px"}),
    dbc.Tab(label="Geografía e Idiomas",     tab_id="tab-geo",       label_style={"fontSize": "13px"}),
    dbc.Tab(label="Popularidad y Valoración",tab_id="tab-pop",       label_style={"fontSize": "13px"}),
    dbc.Tab(label="Evolución Temporal",      tab_id="tab-temporal",  label_style={"fontSize": "13px"}),
], id="tabs", active_tab="tab-catalogo",
   style={"marginBottom": "16px", "fontFamily": FONT})

# ── LAYOUT ──────────────────────────────────────────────────────
app.layout = html.Div([
    header,
    dbc.Container([
        dbc.Row([
            # Sidebar
            dbc.Col(sidebar, width=2, style={"paddingTop": "20px"}),

            # Contenido principal
            dbc.Col([
                # Fila KPIs
                html.Div(id="kpi-row", style={"marginTop": "20px", "marginBottom": "16px"}),
                # Tabs + contenido
                tabs,
                html.Div(id="tab-content"),
            ], width=10),
        ])
    ], fluid=True, style={"maxWidth": "1400px"}),
], style={"backgroundColor": C_BG, "minHeight": "100vh", "fontFamily": FONT})


# ─────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Input("filtro-tipo", "value"),
    Input("filtro-years", "value"),
)
def update_kpis(tipo, years):
    df = filtrar(tipo, years)
    total     = len(df)
    peliculas = len(df[df["type_label"] == "Película"])
    series_n  = len(df[df["type_label"] == "Serie"])
    paises    = int(df["country"].dropna().str.split(", ").explode().nunique())
    idiomas   = int(df["language"].nunique())
    pop_avg   = df["popularity"].mean()
    val_avg   = df["vote_average"].mean()

    return dbc.Row([
        kpi("Total contenidos", f"{total:,}",    "catálogo completo", C_NEUTRAL, grande=True),
        kpi("Películas",        f"{peliculas:,}", f"{peliculas/total*100:.0f}% del catálogo", C_MOVIE),
        kpi("Series",           f"{series_n:,}",  f"{series_n/total*100:.0f}% del catálogo",  C_SHOW),
        kpi("Países",           f"{paises}",      "países productores", C_NEUTRAL),
        kpi("Idiomas",          f"{idiomas}",     "idiomas disponibles", C_NEUTRAL),
        kpi("Popularidad prom.",f"{pop_avg:.1f}", "índice TMDB relativo", C_NEUTRAL),
        kpi("Calificación prom.",f"{val_avg:.2f}/10","escala 0–10", C_NEUTRAL),
    ], className="g-2")


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    Input("filtro-tipo", "value"),
    Input("filtro-years", "value"),
    Input("filtro-top", "value"),
)
def render_tab(tab, tipo, years, top_n):
    df = filtrar(tipo, years)

    # ══ TAB CATÁLOGO ══════════════════════════════════════════════
    if tab == "tab-catalogo":

        # 1. Proporción películas/series
        counts = df["type_label"].value_counts().reset_index()
        counts.columns = ["Tipo", "Cantidad"]
        counts["Pct"] = (counts["Cantidad"] / counts["Cantidad"].sum() * 100).round(1)
        counts["Label"] = counts.apply(lambda r: f"{r['Cantidad']:,}  ({r['Pct']}%)", axis=1)

        fig_tipo = px.bar(
            counts, x="Cantidad", y="Tipo", orientation="h",
            color="Tipo",
            color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
            text="Label",
            title="¿Cómo se distribuye el catálogo?",
        )
        fig_tipo.update_traces(textposition="outside", textfont_size=12)
        fig_tipo.update_layout(**layout_grafico(),
            showlegend=False,
            xaxis=dict(title="Cantidad de títulos", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title=""),
            xaxis_range=[0, counts["Cantidad"].max() * 1.22],
        )

        # 2. Top géneros (película y serie combinados o por tipo)
        rows_g = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_g.append({"Género": g, "Tipo": row["type_label"]})
        gdf = pd.DataFrame(rows_g)
        top_g = gdf["Género"].value_counts().head(top_n).reset_index()
        top_g.columns = ["Género", "Cantidad"]

        fig_generos = px.bar(
            top_g.sort_values("Cantidad"),
            x="Cantidad", y="Género", orientation="h",
            color_discrete_sequence=[C_SHOW if tipo == "Serie" else C_MOVIE if tipo == "Película" else C_NEUTRAL],
            text="Cantidad",
            title=f"Top {top_n} géneros — ¿Cuáles dominan el catálogo?",
        )
        fig_generos.update_traces(textposition="outside", textfont_size=11)
        fig_generos.update_layout(**layout_grafico(),
            showlegend=False,
            xaxis=dict(title="Cantidad de contenidos", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title=""),
            xaxis_range=[0, top_g["Cantidad"].max() * 1.18],
        )

        # 3. Distribución calificación en rangos
        rating_counts = df.groupby(["rating_bin", "type_label"]).size().reset_index(name="Cantidad")
        fig_rating = px.bar(
            rating_counts, x="rating_bin", y="Cantidad", color="type_label",
            barmode="group",
            color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
            title="¿Cómo se distribuyen las calificaciones?  (escala 0–10)",
            labels={"rating_bin": "Rango de calificación", "type_label": "Tipo", "Cantidad": "N° contenidos"}
        )
        fig_rating.update_layout(**layout_grafico(),
            xaxis=dict(title="Rango de calificación", showgrid=False),
            yaxis=dict(title="Cantidad de contenidos", showgrid=True, gridcolor=C_BORDER),
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_tipo,    config={"displayModeBar": False}), width=4),
                dbc.Col(dcc.Graph(figure=fig_rating,  config={"displayModeBar": False}), width=8),
            ], className="mb-3 g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_generos, config={"displayModeBar": False}), width=12),
            ]),
        ])

    # ══ TAB GEOGRAFÍA ═════════════════════════════════════════════
    elif tab == "tab-geo":

        # Países
        paises_s = df["country"].dropna().str.split(", ").explode()
        top_p = paises_s.value_counts().head(top_n).reset_index()
        top_p.columns = ["País", "Cantidad"]

        fig_paises = px.bar(
            top_p.sort_values("Cantidad"),
            x="Cantidad", y="País", orientation="h",
            color="Cantidad",
            color_continuous_scale=[[0, "#BBDEFB"], [1, C_MOVIE]],
            text="Cantidad",
            title=f"Top {top_n} países productores — ¿Dónde se produce el contenido?",
            labels={"Cantidad": "Títulos"}
        )
        fig_paises.update_traces(textposition="outside", textfont_size=11)
        fig_paises.update_layout(**layout_grafico(),
            coloraxis_showscale=False,
            xaxis=dict(title="Cantidad de títulos", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title=""),
            xaxis_range=[0, top_p["Cantidad"].max() * 1.18],
        )

        # Idiomas
        lang_s = df["lang_label"].value_counts().head(top_n).reset_index()
        lang_s.columns = ["Idioma", "Cantidad"]

        fig_lang = px.bar(
            lang_s.sort_values("Cantidad"),
            x="Cantidad", y="Idioma", orientation="h",
            color_discrete_sequence=[C_SHOW],
            text="Cantidad",
            title=f"Top {top_n} idiomas — ¿En qué idiomas está disponible el catálogo?",
        )
        fig_lang.update_traces(textposition="outside", textfont_size=11)
        fig_lang.update_layout(**layout_grafico(),
            showlegend=False,
            xaxis=dict(title="Cantidad de contenidos", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title=""),
            xaxis_range=[0, lang_s["Cantidad"].max() * 1.18],
        )

        nota = html.P(
            "ℹ Un contenido con coproducción se contabiliza en cada país participante.",
            style={"fontSize": "11px", "color": C_MUTED, "marginTop": "4px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_paises, config={"displayModeBar": False}), width=7),
                dbc.Col(dcc.Graph(figure=fig_lang,   config={"displayModeBar": False}), width=5),
            ], className="g-3"),
            nota,
        ])

    # ══ TAB POPULARIDAD Y VALORACIÓN ══════════════════════════════
    elif tab == "tab-pop":

        # Scatter popularidad vs calificación
        p95   = df["popularity"].quantile(0.95)
        df_sc = df[(df["popularity"] <= p95) & (df["vote_average"] > 0)].copy()
        sample = df_sc.sample(n=min(4000, len(df_sc)), random_state=42)

        fig_sc = px.scatter(
            sample, x="popularity", y="vote_average",
            color="type_label",
            color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
            size="vote_count", size_max=14, opacity=0.5,
            hover_name="title",
            hover_data={
                "popularity": ":.1f",
                "vote_average": ":.2f",
                "vote_count": True,
                "type_label": False
            },
            labels={
                "popularity":    "Popularidad (TMDB)",
                "vote_average":  "Calificación promedio (0–10)",
                "type_label":    "Tipo",
                "vote_count":    "N° votos"
            },
            title="¿Los contenidos más populares son los mejor valorados?",
        )
        med_pop = float(df_sc["popularity"].median())
        med_val = float(df_sc["vote_average"].median())
        fig_sc.add_hline(y=med_val, line_dash="dot", line_color=C_MUTED,
                         opacity=0.6, annotation_text=f"Mediana: {med_val:.1f}",
                         annotation_font_size=10)
        fig_sc.add_vline(x=med_pop, line_dash="dot", line_color=C_MUTED,
                         opacity=0.6, annotation_text=f"Med.: {med_pop:.0f}",
                         annotation_font_size=10)
        fig_sc.update_layout(**layout_grafico(),
            xaxis=dict(title="Popularidad (TMDB)", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title="Calificación (0–10)", showgrid=True, gridcolor=C_BORDER, range=[0, 10.5]),
        )

        # Top contenidos por popularidad
        top_pop = df[df["vote_average"] > 0].nlargest(top_n, "popularity")[
            ["title","popularity","vote_average","type_label"]
        ]
        fig_pop = px.bar(
            top_pop.sort_values("popularity"),
            x="popularity", y="title",
            orientation="h",
            color="type_label",
            color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
            hover_data={"vote_average": ":.2f"},
            title=f"Top {top_n} contenidos más populares",
            labels={"popularity": "Popularidad", "title": "", "type_label": "Tipo"},
        )
        fig_pop.update_layout(**layout_grafico(),
            xaxis=dict(title="Índice de popularidad (TMDB)", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title=""),
        )

        # Popularidad promedio por género
        rows_pg = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_pg.append({"Género": g, "popularity": row["popularity"]})
        pgdf = pd.DataFrame(rows_pg)
        top_pg = pgdf.groupby("Género")["popularity"].mean().nlargest(top_n).reset_index()
        top_pg.columns = ["Género", "Popularidad promedio"]

        fig_pg = px.bar(
            top_pg.sort_values("Popularidad promedio"),
            x="Popularidad promedio", y="Género", orientation="h",
            color_discrete_sequence=[C_NEUTRAL],
            text=top_pg.sort_values("Popularidad promedio")["Popularidad promedio"].round(1),
            title=f"Top {top_n} géneros por popularidad promedio",
            labels={"Popularidad promedio": "Popularidad promedio (TMDB)"}
        )
        fig_pg.update_traces(textposition="outside", textfont_size=10)
        fig_pg.update_layout(**layout_grafico(),
            showlegend=False,
            xaxis=dict(title="Popularidad promedio (TMDB)", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title=""),
        )

        nota_pop = html.P(
            "⚠ El índice de popularidad es un valor relativo de TMDB y no representa cantidad de reproducciones. "
            "El scatter excluye el 5% de valores extremos de popularidad para mayor legibilidad.",
            style={"fontSize": "11px", "color": C_MUTED, "marginTop": "4px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_sc,  config={"displayModeBar": False}), width=7),
                dbc.Col(dcc.Graph(figure=fig_pop, config={"displayModeBar": False}), width=5),
            ], className="mb-3 g-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_pg,  config={"displayModeBar": False}), width=12),
            ]),
            nota_pop,
        ])

    # ══ TAB EVOLUCIÓN TEMPORAL ════════════════════════════════════
    elif tab == "tab-temporal":

        # Incorporación por año
        evol = df.groupby(["year_added", "type_label"]).size().reset_index(name="Cantidad")
        fig_ev = px.line(
            evol, x="year_added", y="Cantidad", color="type_label",
            markers=True,
            color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
            title="¿Cómo creció el catálogo año a año?",
            labels={
                "year_added": "Año de incorporación a la plataforma",
                "Cantidad":   "Contenidos incorporados",
                "type_label": "Tipo"
            }
        )
        fig_ev.update_traces(line_width=2.5, marker_size=7)
        fig_ev.update_layout(**layout_grafico(),
            xaxis=dict(title="Año de incorporación", showgrid=True,
                       gridcolor=C_BORDER, dtick=1),
            yaxis=dict(title="Contenidos incorporados", showgrid=True, gridcolor=C_BORDER),
        )

        # Estrenos por año de producción
        evol_rel = df[df["release_year"] >= 1990].groupby(
            ["release_year", "type_label"]
        ).size().reset_index(name="Cantidad")

        fig_rel = px.line(
            evol_rel, x="release_year", y="Cantidad", color="type_label",
            color_discrete_map={"Película": C_MOVIE, "Serie": C_SHOW},
            title="¿Cuándo fueron producidos los contenidos? (desde 1990)",
            labels={
                "release_year": "Año de estreno",
                "Cantidad":     "Cantidad de contenidos",
                "type_label":   "Tipo"
            }
        )
        fig_rel.update_traces(line_width=2)
        fig_rel.update_layout(**layout_grafico(),
            xaxis=dict(title="Año de estreno", showgrid=True, gridcolor=C_BORDER),
            yaxis=dict(title="Cantidad de contenidos", showgrid=True, gridcolor=C_BORDER),
        )

        nota_t = html.P(
            "ℹ El dataset contiene exactamente 1.000 registros por año de incorporación por categoría "
            "→ la distribución temporal uniforme refleja un muestreo balanceado, no el ritmo real de incorporación.",
            style={"fontSize": "11px", "color": C_MUTED, "marginTop": "4px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_ev,  config={"displayModeBar": False}), width=6),
                dbc.Col(dcc.Graph(figure=fig_rel, config={"displayModeBar": False}), width=6),
            ], className="g-3"),
            nota_t,
        ])

    return html.Div()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  StreamView Analytics — Dashboard EP1")
    print("  Abrir en: http://127.0.0.1:8050")
    print("=" * 55)
    app.run(debug=False, host="127.0.0.1", port=8050)