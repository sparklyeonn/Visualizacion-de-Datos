"""
STREAMVIEW ANALYTICS — EP1 | DASHBOARD INTERACTIVO
Diseño oscuro profesional | ADY1104 Duoc UC 2026
"""

import os, warnings
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# ── PALETA OSCURA ────────────────────────────────────────────────
BG_MAIN    = "#0F1117"   # fondo general
BG_CARD    = "#1A1D27"   # tarjetas
BG_SIDEBAR = "#13151F"   # sidebar
BG_CHART   = "#1A1D27"   # fondo gráficos

C_ACCENT1  = "#7C3AED"   # violeta principal
C_ACCENT2  = "#06B6D4"   # cyan
C_ACCENT3  = "#F472B6"   # rosa/magenta
C_ACCENT4  = "#10B981"   # verde esmeralda
C_MOVIE    = "#7C3AED"   # películas → violeta
C_SHOW     = "#06B6D4"   # series → cyan

C_TEXT     = "#E2E8F0"   # texto principal
C_MUTED    = "#64748B"   # texto secundario
C_BORDER   = "#2D3148"   # bordes

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

YEAR_MIN = int(combined["year_added"].min())
YEAR_MAX = int(combined["year_added"].max())

# ── HELPERS ──────────────────────────────────────────────────────
def filtrar(tipo="Todos", year_range=None):
    df = combined.copy()
    if tipo != "Todos":
        df = df[df["type_label"] == tipo]
    if year_range:
        df = df[(df["year_added"] >= year_range[0]) & (df["year_added"] <= year_range[1])]
    return df

def base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=13, color=C_TEXT, family=FONT), x=0, pad=dict(l=4)),
        plot_bgcolor=BG_CHART,
        paper_bgcolor=BG_CHART,
        font=dict(family=FONT, size=11, color=C_TEXT),
        margin=dict(l=8, r=16, t=44, b=8),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=10, color=C_MUTED),
            bgcolor="rgba(0,0,0,0)"
        ),
        hoverlabel=dict(bgcolor=BG_CARD, font_size=12, font_color=C_TEXT,
                        bordercolor=C_BORDER),
        xaxis=dict(showgrid=True, gridcolor=C_BORDER, gridwidth=1,
                   zeroline=False, tickfont=dict(color=C_MUTED, size=10),
                   title_font=dict(color=C_MUTED, size=10)),
        yaxis=dict(showgrid=True, gridcolor=C_BORDER, gridwidth=1,
                   zeroline=False, tickfont=dict(color=C_MUTED, size=10),
                   title_font=dict(color=C_MUTED, size=10)),
    )

def card(children, padding="20px", height=None):
    style = {
        "backgroundColor": BG_CARD,
        "borderRadius": "12px",
        "border": f"1px solid {C_BORDER}",
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

# ── ESTILOS CSS GLOBALES ─────────────────────────────────────────
external_css = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
]

app = Dash(__name__, external_stylesheets=external_css, title="StreamView Analytics")

# ── SIDEBAR ──────────────────────────────────────────────────────
sidebar = html.Div([
    # Logo / título
    html.Div([
        html.Div("▶", style={
            "fontSize":"22px","color":C_ACCENT1,
            "lineHeight":"1","marginBottom":"2px"
        }),
        html.Div("StreamView", style={
            "fontSize":"15px","fontWeight":"800","color":C_TEXT,"letterSpacing":"-0.3px"
        }),
        html.Div("Analytics EP1", style={
            "fontSize":"10px","color":C_MUTED,"letterSpacing":"0.5px"
        }),
    ], style={"marginBottom":"32px"}),

    # Sección filtros
    html.Div("FILTROS", style={
        "fontSize":"9px","fontWeight":"700","color":C_MUTED,
        "letterSpacing":"1.5px","marginBottom":"14px"
    }),

    html.Div("Tipo de contenido", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT,"marginBottom":"8px"
    }),
    dcc.RadioItems(
        id="filtro-tipo",
        options=[{"label": f"  {t}", "value": t} for t in ["Todos","Película","Serie"]],
        value="Todos",
        labelStyle={"display":"block","marginBottom":"8px","fontSize":"12px","color":C_MUTED,"cursor":"pointer"},
        inputStyle={"marginRight":"8px","accentColor":C_ACCENT1},
        style={"marginBottom":"24px"}
    ),

    html.Div("Años de incorporación", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT,"marginBottom":"10px"
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
        "fontSize":"11px","fontWeight":"600","color":C_TEXT,"marginBottom":"10px"
    }),
    dcc.Slider(
        id="filtro-top", min=5, max=20, step=5, value=10,
        marks={5:"5", 10:"10", 15:"15", 20:"20"},
    ),
    html.Div(style={"marginBottom":"32px"}),

    # Sección vistas
    html.Div("VISTAS", style={
        "fontSize":"9px","fontWeight":"700","color":C_MUTED,
        "letterSpacing":"1.5px","marginBottom":"14px"
    }),
    dcc.RadioItems(
        id="nav-tab",
        options=[
            {"label":"  📊  Catálogo",            "value":"tab-catalogo"},
            {"label":"  🌍  Geografía & Idiomas",  "value":"tab-geo"},
            {"label":"  ⭐  Popularidad",           "value":"tab-pop"},
            {"label":"  📅  Evolución Temporal",   "value":"tab-temporal"},
        ],
        value="tab-catalogo",
        labelStyle={
            "display":"block","marginBottom":"6px","fontSize":"12px",
            "color":C_MUTED,"cursor":"pointer","padding":"6px 10px",
            "borderRadius":"6px"
        },
        inputStyle={"display":"none"},
        style={"marginBottom":"24px"}
    ),

    # Notas
    html.Div(style={"borderTop":f"1px solid {C_BORDER}","marginBottom":"16px"}),
    html.Div("⚠ Popularidad = índice relativo TMDB.", style={
        "fontSize":"10px","color":C_MUTED,"lineHeight":"1.5","marginBottom":"6px"
    }),
    html.Div("ℹ 1.000 registros/año por categoría.", style={
        "fontSize":"10px","color":C_MUTED,"lineHeight":"1.5"
    }),
], style={
    "backgroundColor": BG_SIDEBAR,
    "width":"200px","minHeight":"100vh",
    "padding":"24px 16px",
    "position":"fixed","top":"0","left":"0",
    "borderRight":f"1px solid {C_BORDER}",
    "fontFamily":FONT,
    "overflowY":"auto",
})

# ── LAYOUT PRINCIPAL ─────────────────────────────────────────────
app.layout = html.Div([
    sidebar,
    # Contenido (margen izquierdo = ancho sidebar)
    html.Div([
        # Header
        html.Div([
            html.Div(id="page-title", style={
                "fontSize":"18px","fontWeight":"800","color":C_TEXT
            }),
            html.Div("ADY1104 · Visualización de Datos · Duoc UC 2026", style={
                "fontSize":"11px","color":C_MUTED
            }),
        ], style={
            "padding":"20px 24px 16px",
            "borderBottom":f"1px solid {C_BORDER}",
            "marginBottom":"20px"
        }),

        # KPIs
        html.Div(id="kpi-row", style={"padding":"0 24px","marginBottom":"20px"}),

        # Contenido dinámico
        html.Div(id="tab-content", style={"padding":"0 24px 32px"}),

    ], style={"marginLeft":"200px","backgroundColor":BG_MAIN,"minHeight":"100vh","fontFamily":FONT}),

], style={"backgroundColor":BG_MAIN,"fontFamily":FONT})


# ── CALLBACKS ────────────────────────────────────────────────────

@app.callback(
    Output("page-title","children"),
    Input("nav-tab","value"),
)
def update_title(tab):
    titles = {
        "tab-catalogo": "📊 Composición del Catálogo",
        "tab-geo":       "🌍 Geografía e Idiomas",
        "tab-pop":       "⭐ Popularidad y Valoración",
        "tab-temporal":  "📅 Evolución Temporal",
    }
    return titles.get(tab, "StreamView Analytics")


@app.callback(
    Output("kpi-row","children"),
    Input("filtro-tipo","value"),
    Input("filtro-years","value"),
)
def update_kpis(tipo, years):
    df = filtrar(tipo, years)
    total    = len(df)
    pel      = len(df[df["type_label"]=="Película"])
    ser      = len(df[df["type_label"]=="Serie"])
    paises   = int(df["country"].dropna().str.split(", ").explode().nunique())
    idiomas  = int(df["language"].nunique())
    pop_avg  = df["popularity"].mean()
    val_avg  = df["vote_average"].mean()

    kpis = [
        ("Total Contenidos",   f"{total:,}",       "catálogo completo",        C_ACCENT1),
        ("Películas",          f"{pel:,}",          f"{pel/total*100:.0f}% del total", C_MOVIE),
        ("Series",             f"{ser:,}",          f"{ser/total*100:.0f}% del total", C_SHOW),
        ("Países",             f"{paises}",          "países productores",       C_ACCENT4),
        ("Idiomas",            f"{idiomas}",         "idiomas disponibles",      C_ACCENT3),
        ("Popularidad Prom.",  f"{pop_avg:.1f}",    "índice TMDB relativo",     C_ACCENT2),
        ("Calificación Prom.", f"{val_avg:.2f}/10", "escala 0–10",              C_ACCENT4),
    ]

    cols = []
    for label, val, sub, color in kpis:
        cols.append(
            dbc.Col(kpi_card(label, val, sub, color), style={"marginBottom":"0"})
        )
    return dbc.Row(cols, className="g-2")


@app.callback(
    Output("tab-content","children"),
    Input("nav-tab","value"),
    Input("filtro-tipo","value"),
    Input("filtro-years","value"),
    Input("filtro-top","value"),
)
def render_tab(tab, tipo, years, top_n):
    df = filtrar(tipo, years)

    # ══ CATÁLOGO ══════════════════════════════════════════════════
    if tab == "tab-catalogo":

        # Películas vs Series
        counts = df["type_label"].value_counts().reset_index()
        counts.columns = ["Tipo","Cantidad"]
        counts["Pct"] = (counts["Cantidad"]/counts["Cantidad"].sum()*100).round(1)
        counts["Label"] = counts.apply(lambda r: f"{r['Cantidad']:,} ({r['Pct']}%)", axis=1)

        fig_tipo = go.Figure()
        colors_tipo = [C_MOVIE if t=="Película" else C_SHOW for t in counts["Tipo"]]
        fig_tipo.add_trace(go.Bar(
            x=counts["Cantidad"], y=counts["Tipo"],
            orientation="h",
            marker=dict(
                color=colors_tipo,
                opacity=0.9,
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            text=counts["Label"],
            textposition="outside",
            textfont=dict(color=C_TEXT, size=11),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        ))
        lay = base_layout("¿Cómo se distribuye el catálogo?")
        lay["xaxis"]["title"] = "Cantidad de títulos"
        lay["yaxis"]["title"] = ""
        lay["xaxis"]["showgrid"] = False
        fig_tipo.update_layout(**lay, xaxis_range=[0, counts["Cantidad"].max()*1.22])

        # Géneros
        rows_g = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_g.append({"Género":g,"Tipo":row["type_label"]})
        gdf = pd.DataFrame(rows_g)
        top_g = gdf["Género"].value_counts().head(top_n).reset_index()
        top_g.columns = ["Género","Cantidad"]
        top_g = top_g.sort_values("Cantidad")

        fig_gen = go.Figure(go.Bar(
            x=top_g["Cantidad"], y=top_g["Género"],
            orientation="h",
            marker=dict(
                color=top_g["Cantidad"],
                colorscale=[[0, C_ACCENT1+"55"],[1, C_ACCENT1]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_g["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay2 = base_layout(f"Top {top_n} géneros — ¿Cuáles dominan el catálogo?")
        lay2["xaxis"]["title"] = "Cantidad de contenidos"
        lay2["yaxis"]["title"] = ""
        lay2["xaxis"]["showgrid"] = False
        lay2["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_gen.update_layout(**lay2, xaxis_range=[0, top_g["Cantidad"].max()*1.18])

        # Calificaciones en rangos
        rc = df.groupby(["rating_bin","type_label"], observed=True).size().reset_index(name="Cantidad")
        fig_rat = go.Figure()
        for tipo_val, color in [("Película", C_MOVIE), ("Serie", C_SHOW)]:
            d = rc[rc["type_label"]==tipo_val]
            fig_rat.add_trace(go.Bar(
                name=tipo_val, x=d["rating_bin"].astype(str), y=d["Cantidad"],
                marker=dict(color=color, opacity=0.85, line=dict(color="rgba(0,0,0,0)")),
                hovertemplate=f"{tipo_val} — %{{x}}: %{{y:,}}<extra></extra>",
            ))
        lay3 = base_layout("Distribución de calificaciones (0–10)")
        lay3["xaxis"]["title"] = "Rango de calificación"
        lay3["yaxis"]["title"] = "Cantidad de contenidos"
        lay3["barmode"] = "group"
        fig_rat.update_layout(**lay3)

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_tipo, config={"displayModeBar":False},
                             style={"height":"220px"}), padding="16px"), width=4),
                dbc.Col(card(dcc.Graph(figure=fig_rat,  config={"displayModeBar":False},
                             style={"height":"220px"}), padding="16px"), width=8),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_gen, config={"displayModeBar":False},
                             style={"height":"320px"}), padding="16px"), width=12),
            ]),
        ])

    # ══ GEOGRAFÍA ═════════════════════════════════════════════════
    elif tab == "tab-geo":

        paises_s = df["country"].dropna().str.split(", ").explode()
        top_p = paises_s.value_counts().head(top_n).reset_index()
        top_p.columns = ["País","Cantidad"]
        top_p = top_p.sort_values("Cantidad")

        fig_paises = go.Figure(go.Bar(
            x=top_p["Cantidad"], y=top_p["País"], orientation="h",
            marker=dict(
                color=top_p["Cantidad"],
                colorscale=[[0, C_ACCENT2+"44"],[1, C_ACCENT2]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_p["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} títulos<extra></extra>",
        ))
        lay_p = base_layout(f"Top {top_n} países productores")
        lay_p["xaxis"]["title"] = "Cantidad de títulos"
        lay_p["yaxis"]["title"] = ""
        lay_p["xaxis"]["showgrid"] = False
        lay_p["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_paises.update_layout(**lay_p, xaxis_range=[0, top_p["Cantidad"].max()*1.18])

        lang_s = df["lang_label"].value_counts().head(top_n).reset_index()
        lang_s.columns = ["Idioma","Cantidad"]
        lang_s = lang_s.sort_values("Cantidad")

        fig_lang = go.Figure(go.Bar(
            x=lang_s["Cantidad"], y=lang_s["Idioma"], orientation="h",
            marker=dict(
                color=lang_s["Cantidad"],
                colorscale=[[0, C_ACCENT3+"44"],[1, C_ACCENT3]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=lang_s["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay_l = base_layout(f"Top {top_n} idiomas disponibles")
        lay_l["xaxis"]["title"] = "Cantidad de contenidos"
        lay_l["yaxis"]["title"] = ""
        lay_l["xaxis"]["showgrid"] = False
        lay_l["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_lang.update_layout(**lay_l, xaxis_range=[0, lang_s["Cantidad"].max()*1.18])

        nota = html.Div(
            "ℹ Un contenido con coproducción se contabiliza en cada país participante.",
            style={"fontSize":"10px","color":C_MUTED,"marginTop":"10px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_paises, config={"displayModeBar":False},
                             style={"height":"360px"}), padding="16px"), width=7),
                dbc.Col(card(dcc.Graph(figure=fig_lang,   config={"displayModeBar":False},
                             style={"height":"360px"}), padding="16px"), width=5),
            ], className="g-3"),
            nota,
        ])

    # ══ POPULARIDAD ═══════════════════════════════════════════════
    elif tab == "tab-pop":

        p95   = df["popularity"].quantile(0.95)
        df_sc = df[(df["popularity"] <= p95) & (df["vote_average"] > 0)].copy()
        sample = df_sc.sample(n=min(3500, len(df_sc)), random_state=42)

        fig_sc = go.Figure()
        for tipo_val, color in [("Película",C_MOVIE),("Serie",C_SHOW)]:
            d = sample[sample["type_label"]==tipo_val]
            size_norm = (d["vote_count"].fillna(0) / d["vote_count"].max() * 14 + 4).clip(4,14)
            fig_sc.add_trace(go.Scatter(
                x=d["popularity"], y=d["vote_average"],
                mode="markers",
                name=tipo_val,
                marker=dict(color=color, size=size_norm, opacity=0.5,
                            line=dict(color="rgba(0,0,0,0)")),
                text=d["title"],
                customdata=d[["vote_count","type_label"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Popularidad: %{x:.1f}<br>"
                    "Calificación: %{y:.2f}/10<br>"
                    "Votos: %{customdata[0]:,}<extra></extra>"
                ),
            ))
        med_pop = float(df_sc["popularity"].median())
        med_val = float(df_sc["vote_average"].median())
        fig_sc.add_hline(y=med_val, line_dash="dot", line_color=C_MUTED, opacity=0.5,
                         annotation_text=f"Med. cal. {med_val:.1f}",
                         annotation_font=dict(size=9, color=C_MUTED))
        fig_sc.add_vline(x=med_pop, line_dash="dot", line_color=C_MUTED, opacity=0.5,
                         annotation_text=f"Med. pop. {med_pop:.0f}",
                         annotation_font=dict(size=9, color=C_MUTED))
        lay_sc = base_layout("¿Los más populares son los mejor valorados?")
        lay_sc["xaxis"]["title"] = "Popularidad (TMDB)"
        lay_sc["yaxis"]["title"] = "Calificación (0–10)"
        lay_sc["yaxis"]["range"] = [0, 10.5]
        fig_sc.update_layout(**lay_sc)

        # Top popularidad
        top_pop = df[df["vote_average"]>0].nlargest(top_n,"popularity")[
            ["title","popularity","vote_average","type_label"]
        ]
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
        lay_pop = base_layout(f"Top {top_n} más populares")
        lay_pop["xaxis"]["title"] = "Índice de popularidad"
        lay_pop["yaxis"]["title"] = ""
        lay_pop["barmode"] = "stack"
        lay_pop["xaxis"]["showgrid"] = False
        fig_pop.update_layout(**lay_pop)

        # Popularidad por género
        rows_pg = []
        for _, row in df.dropna(subset=["genres"]).iterrows():
            for g in row["genres"].split(", "):
                rows_pg.append({"Género":g,"popularity":row["popularity"]})
        pgdf = pd.DataFrame(rows_pg)
        top_pg = pgdf.groupby("Género")["popularity"].mean().nlargest(top_n).reset_index()
        top_pg.columns = ["Género","Popularidad promedio"]
        top_pg = top_pg.sort_values("Popularidad promedio")

        fig_pg = go.Figure(go.Bar(
            x=top_pg["Popularidad promedio"], y=top_pg["Género"],
            orientation="h",
            marker=dict(
                color=top_pg["Popularidad promedio"],
                colorscale=[[0,C_ACCENT4+"44"],[1,C_ACCENT4]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_pg["Popularidad promedio"].round(1),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ))
        lay_pg = base_layout(f"Top {top_n} géneros por popularidad promedio")
        lay_pg["xaxis"]["title"] = "Popularidad promedio (TMDB)"
        lay_pg["yaxis"]["title"] = ""
        lay_pg["xaxis"]["showgrid"] = False
        lay_pg["margin"] = dict(l=8, r=50, t=44, b=8)
        fig_pg.update_layout(**lay_pg,
                             xaxis_range=[0, top_pg["Popularidad promedio"].max()*1.18])

        nota_pop = html.Div(
            "⚠ Popularidad = índice relativo TMDB. No equivale a reproducciones. "
            "Scatter excluye el 5% extremo de popularidad.",
            style={"fontSize":"10px","color":C_MUTED,"marginTop":"10px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_sc, config={"displayModeBar":False},
                             style={"height":"320px"}), padding="16px"), width=7),
                dbc.Col(card(dcc.Graph(figure=fig_pop, config={"displayModeBar":False},
                             style={"height":"320px"}), padding="16px"), width=5),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_pg, config={"displayModeBar":False},
                             style={"height":"280px"}), padding="16px"), width=12),
            ]),
            nota_pop,
        ])

    # ══ EVOLUCIÓN TEMPORAL ════════════════════════════════════════
    elif tab == "tab-temporal":

        evol = df.groupby(["year_added","type_label"]).size().reset_index(name="Cantidad")

        fig_ev = go.Figure()
        for tipo_val, color in [("Película",C_MOVIE),("Serie",C_SHOW)]:
            d = evol[evol["type_label"]==tipo_val].sort_values("year_added")
            # Área rellena
            fig_ev.add_trace(go.Scatter(
                x=d["year_added"], y=d["Cantidad"],
                name=tipo_val, mode="lines+markers",
                line=dict(color=color, width=2.5, shape="spline"),
                marker=dict(size=6, color=color),
                fill="tozeroy",
                hovertemplate=f"{tipo_val} %{{x}}: %{{y:,}} contenidos<extra></extra>",
            ))
        # Usar fill con alpha manual
        fig_ev.data[0].fillcolor = "rgba(124,58,237,0.08)"
        if len(fig_ev.data) > 1:
            fig_ev.data[1].fillcolor = "rgba(6,182,212,0.08)"

        lay_ev = base_layout("Incorporación de contenidos al catálogo por año")
        lay_ev["xaxis"]["title"] = "Año de incorporación"
        lay_ev["yaxis"]["title"] = "Contenidos incorporados"
        lay_ev["xaxis"]["dtick"] = 2
        fig_ev.update_layout(**lay_ev)

        # Estrenos por año de producción
        evol_rel = df[df["release_year"]>=1990].groupby(
            ["release_year","type_label"]
        ).size().reset_index(name="Cantidad")

        fig_rel = go.Figure()
        for tipo_val, color in [("Película",C_MOVIE),("Serie",C_SHOW)]:
            d = evol_rel[evol_rel["type_label"]==tipo_val].sort_values("release_year")
            fig_rel.add_trace(go.Scatter(
                x=d["release_year"], y=d["Cantidad"],
                name=tipo_val, mode="lines",
                line=dict(color=color, width=2, shape="spline"),
                fill="tozeroy",
                hovertemplate=f"{tipo_val} %{{x}}: %{{y:,}}<extra></extra>",
            ))
        fig_rel.data[0].fillcolor = "rgba(124,58,237,0.08)"
        if len(fig_rel.data) > 1:
            fig_rel.data[1].fillcolor = "rgba(6,182,212,0.08)"

        lay_rel = base_layout("Año de producción de los contenidos (desde 1990)")
        lay_rel["xaxis"]["title"] = "Año de estreno"
        lay_rel["yaxis"]["title"] = "Cantidad de contenidos"
        fig_rel.update_layout(**lay_rel)

        nota_t = html.Div(
            "ℹ Dataset muestreado: 1.000 registros/año por categoría → "
            "la distribución temporal uniforme es por diseño, no refleja el ritmo real de incorporación.",
            style={"fontSize":"10px","color":C_MUTED,"marginTop":"10px"}
        )

        return html.Div([
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_ev,  config={"displayModeBar":False},
                             style={"height":"300px"}), padding="16px"), width=6),
                dbc.Col(card(dcc.Graph(figure=fig_rel, config={"displayModeBar":False},
                             style={"height":"300px"}), padding="16px"), width=6),
            ], className="g-3"),
            nota_t,
        ])

    return html.Div()


if __name__ == "__main__":
    print("=" * 55)
    print("  StreamView Analytics — Dashboard EP1 (Dark)")
    print("  Abrir en: http://127.0.0.1:8050")
    print("=" * 55)
    app.run(debug=False, host="127.0.0.1", port=8050)