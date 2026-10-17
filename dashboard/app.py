"""
STREAMVIEW ANALYTICS | DASHBOARD INTERACTIVO
Diseño Visualización de Datos | ADY1104 Duoc UC 2026
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
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

warnings.filterwarnings("ignore")

# COLORES DE INTERFAZ (chrome) 
BG_MAIN              = "var(--bg-page)"
BG_CARD              = "var(--bg-card)"
BG_SIDEBAR           = "var(--bg-sidebar)"
C_TEXT               = "var(--text-primary)"
C_TEXT_SIDEBAR       = "var(--text-on-sidebar)"
C_MUTED              = "var(--text-muted)"
C_MUTED_SIDEBAR      = "var(--text-muted-on-sidebar)"
C_BORDER             = "var(--border-color)"
C_ACCENT_PRIMARY     = "var(--accent-primary)"

# COLORES DE GRÁFICOS (Plotly) 
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

FONT = "Inter, system-ui, sans-serif"

LANG_MAP = {
    "en":"Inglés",
    "fr":"Francés",
    "ja":"Japonés",
    "ko":"Coreano",
    "es":"Español",
    "zh":"Chino",
    "it":"Italiano",
    "hi":"Hindi",
    "de":"Alemán",
    "ru":"Ruso",
    "tl":"Filipino",
    "ar":"Árabe",
    "pt":"Portugués",
    "nl":"Holandés",
    "tr":"Turco"
}

#  CARGA 
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
    if tipo and tipo != "Todos":
        df = df[df["type_label"] == tipo]
    if year_range:
        y0 = int(year_range[0])
        y1 = int(year_range[1])
        df = df[
            (pd.to_numeric(df["year_added"], errors="coerce") >= y0) &
            (pd.to_numeric(df["year_added"], errors="coerce") <= y1)
        ]
    return df

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
def generar_infografia_png(tipo, year_range):
    """
    Genera una infografía resumen en PNG como bytes base64.
    Respeta los filtros activos (tipo de contenido + rango de años).
    """
    df = filtrar(tipo, year_range)

    y0 = int(year_range[0]) if year_range else YEAR_MIN
    y1 = int(year_range[1]) if year_range else YEAR_MAX

    # ── Paleta fija de la infografía (siempre oscura, independiente del tema del dashboard)
    BG       = "#0F1117"
    BG_CARD_ = "#1A1D27"
    ACCENT   = "#7C3AED"   # violeta = películas
    CYAN     = "#06B6D4"   # cyan = series
    GOLD     = "#F5A623"   # dorado = hallazgo
    WHITE    = "#E2E8F0"
    GRAY     = "#64748B"
    BORDER   = "#2D3148"

    # ── KPIs base
    total   = len(df)
    n_pel   = len(df[df["type_label"] == "Película"])
    n_ser   = len(df[df["type_label"] == "Serie"])
    pct_pel = n_pel / total * 100 if total > 0 else 0
    pct_ser = n_ser / total * 100 if total > 0 else 0
    n_paises = int(df["country"].dropna().str.split(", ").explode().nunique())
    n_idiomas = int(df["language"].nunique())
    pop_avg  = df["popularity"].mean() if total > 0 else 0
    val_avg  = df["vote_average"].mean() if total > 0 else 0

    # País #1
    paises_exploded = df["country"].dropna().str.split(", ").explode()
    pais_top = paises_exploded.value_counts().index[0] if len(paises_exploded) > 0 else "N/D"

    # Año pico
    evol = df.groupby("year_added").size()
    anio_pico = int(evol.idxmax()) if len(evol) > 0 else y0

    # Top 5 géneros
    rows_g = []
    for _, row in df.dropna(subset=["genres"]).iterrows():
        for g in row["genres"].split(", "):
            rows_g.append(g)
    from collections import Counter
    generos_cnt = Counter(rows_g)
    top5_gen = [(g, c) for g, c in generos_cnt.most_common(5)]

    # Top 5 países
    top5_paises = paises_exploded.value_counts().head(5)

    # Evolución temporal (todos los años del rango)
    evol_full = df.groupby(["year_added", "type_label"]).size().reset_index(name="Cantidad")

    # ── Figura principal
    fig = plt.figure(figsize=(14, 9), facecolor=BG)

    # Título principal
    fig.text(0.5, 0.965, "StreamView Analytics", ha="center", va="top",
             fontsize=22, fontweight="bold", color=WHITE, fontfamily="DejaVu Sans")

    # Subtítulo con contexto de filtro
    tipo_label = tipo if tipo and tipo != "Todos" else "Todo el catálogo"
    fig.text(0.5, 0.935, f"Resumen del catálogo · {tipo_label} · {y0}–{y1}",
             ha="center", va="top", fontsize=11, color=GRAY, fontfamily="DejaVu Sans")

    # Línea divisora bajo título
    ax_line = fig.add_axes([0.04, 0.915, 0.92, 0.002])
    ax_line.set_facecolor(BORDER)
    ax_line.set_xticks([]); ax_line.set_yticks([])
    for sp in ax_line.spines.values(): sp.set_visible(False)

    # ── FILA KPIs (4 tarjetas)
    kpi_data = [
        (f"{total:,}",    "Total títulos",       GOLD),
        (f"{pct_pel:.0f}% / {pct_ser:.0f}%", "Películas / Series", ACCENT),
        (f"{n_paises}",   "Países productores",  CYAN),
        (f"{anio_pico}",  "Año pico",            "#10B981"),
    ]
    kpi_xs = [0.04, 0.27, 0.50, 0.73]
    kpi_w  = 0.21
    kpi_y  = 0.78
    kpi_h  = 0.115

    for (val, lab, col), x in zip(kpi_data, kpi_xs):
        ax_k = fig.add_axes([x, kpi_y, kpi_w, kpi_h])
        ax_k.set_facecolor(BG_CARD_)
        ax_k.set_xlim(0, 1); ax_k.set_ylim(0, 1)
        ax_k.set_xticks([]); ax_k.set_yticks([])
        for sp in ax_k.spines.values():
            sp.set_edgecolor(BORDER); sp.set_linewidth(1)
        ax_k.text(0.5, 0.62, val, ha="center", va="center",
                  fontsize=18, fontweight="bold", color=col, fontfamily="DejaVu Sans")
        ax_k.text(0.5, 0.18, lab, ha="center", va="center",
                  fontsize=9, color=GRAY, fontfamily="DejaVu Sans")

    # ── FILA INFERIOR: 3 paneles
    # Panel 1: Top 5 géneros
    ax1 = fig.add_axes([0.04, 0.13, 0.28, 0.56])
    ax1.set_facecolor(BG_CARD_)
    for sp in ax1.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(1)

    if top5_gen:
        gen_names = [g for g, _ in reversed(top5_gen)]
        gen_vals  = [c for _, c in reversed(top5_gen)]
        max_val   = max(gen_vals) if gen_vals else 1
        colors_bar = [GOLD if i == len(gen_vals)-1 else ACCENT + "AA" for i in range(len(gen_vals))]
        bars = ax1.barh(gen_names, gen_vals, color=colors_bar, height=0.55, edgecolor="none")
        for bar, val in zip(bars, gen_vals):
            ax1.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:,}", va="center", fontsize=8.5, color=WHITE, fontfamily="DejaVu Sans")
        ax1.set_xlim(0, max_val * 1.25)
        ax1.tick_params(colors=GRAY, labelsize=8.5)
        ax1.xaxis.set_visible(False)
        ax1.set_facecolor(BG_CARD_)
        for sp in ["top","right","bottom"]: ax1.spines[sp].set_visible(False)
        ax1.spines["left"].set_edgecolor(BORDER)
        ax1.tick_params(axis="y", colors=WHITE)

    ax1.set_title("Top 5 géneros", fontsize=10, fontweight="bold",
                  color=WHITE, pad=8, fontfamily="DejaVu Sans", loc="left")

    # Panel 2: Evolución temporal
    ax2 = fig.add_axes([0.375, 0.13, 0.30, 0.56])
    ax2.set_facecolor(BG_CARD_)
    for sp in ax2.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(1)

    if len(evol_full) > 0:
        for tipo_val, col in [("Película", ACCENT), ("Serie", CYAN)]:
            d = evol_full[evol_full["type_label"] == tipo_val].sort_values("year_added")
            if len(d) > 0:
                ax2.plot(d["year_added"], d["Cantidad"], color=col,
                         linewidth=2, marker="o", markersize=4, label=tipo_val)
        # Marcar año pico
        ax2.axvline(anio_pico, color=GOLD, linestyle="--", linewidth=1, alpha=0.7)
        ax2.text(anio_pico + 0.15, ax2.get_ylim()[1] * 0.92 if ax2.get_ylim()[1] > 0 else 1,
                 f"Pico {anio_pico}", fontsize=7.5, color=GOLD, fontfamily="DejaVu Sans")
        ax2.legend(fontsize=8, labelcolor=GRAY, facecolor=BG_CARD_, edgecolor=BORDER,
                   framealpha=0.7)
        ax2.tick_params(colors=GRAY, labelsize=8)
        ax2.set_facecolor(BG_CARD_)
        ax2.spines["bottom"].set_edgecolor(BORDER)
        ax2.spines["left"].set_edgecolor(BORDER)
        for sp in ["top","right"]: ax2.spines[sp].set_visible(False)
        ax2.tick_params(axis="x", colors=GRAY)
        ax2.tick_params(axis="y", colors=GRAY)
        ax2.yaxis.set_tick_params(labelcolor=GRAY)

    ax2.set_title("Incorporaciones por año", fontsize=10, fontweight="bold",
                  color=WHITE, pad=8, fontfamily="DejaVu Sans", loc="left")

    # Panel 3: Top 5 países
    ax3 = fig.add_axes([0.715, 0.13, 0.265, 0.56])
    ax3.set_facecolor(BG_CARD_)
    for sp in ax3.spines.values(): sp.set_edgecolor(BORDER); sp.set_linewidth(1)

    if len(top5_paises) > 0:
        p_names = list(reversed(top5_paises.index.tolist()))
        p_vals  = list(reversed(top5_paises.values.tolist()))
        max_p   = max(p_vals) if p_vals else 1
        colors_p = [GOLD if i == len(p_vals)-1 else CYAN + "AA" for i in range(len(p_vals))]
        bars3 = ax3.barh(p_names, p_vals, color=colors_p, height=0.55, edgecolor="none")
        for bar, val in zip(bars3, p_vals):
            ax3.text(bar.get_width() + max_p * 0.02, bar.get_y() + bar.get_height()/2,
                     f"{val:,}", va="center", fontsize=8.5, color=WHITE, fontfamily="DejaVu Sans")
        ax3.set_xlim(0, max_p * 1.28)
        ax3.xaxis.set_visible(False)
        ax3.set_facecolor(BG_CARD_)
        for sp in ["top","right","bottom"]: ax3.spines[sp].set_visible(False)
        ax3.spines["left"].set_edgecolor(BORDER)
        ax3.tick_params(axis="y", colors=WHITE, labelsize=8.5)

    ax3.set_title("Top 5 países", fontsize=10, fontweight="bold",
                  color=WHITE, pad=8, fontfamily="DejaVu Sans", loc="left")

    # ── Pie de página
    fig.text(0.5, 0.025,
             f"Fuente: TMDB · Dataset StreamView Analytics · ADY1104 Visualización de Datos · Duoc UC 2026",
             ha="center", va="bottom", fontsize=8, color=GRAY, fontfamily="DejaVu Sans")

    # ── Exportar a PNG en memoria
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── ESTILOS CSS GLOBALES ─────────────────────────────────────────
external_css = [
    dbc.themes.BOOTSTRAP,
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
]

app = Dash(__name__,
           external_stylesheets=external_css,
           title="StreamView Analytics")
server = app.server

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

sidebar_body = html.Div([
    # Sección filtros
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

    html.Div("Años de incorporación", style={
        "fontSize":"11px","fontWeight":"600","color":C_TEXT_SIDEBAR,"marginBottom":"10px"
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
    html.Div(style={"marginBottom":"32px"}),

    # Sección vistas
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

    # Notas
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
    "width":"200px","minHeight":"100vh",
    "padding":"24px 16px",
    "position":"fixed","top":"0","left":"0",
    "borderRight":"1px solid var(--border-color-sidebar)",
    "fontFamily":FONT,
    "overflowY":"auto",
})

# ── LAYOUT PRINCIPAL ─────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="theme-mode", storage_type="local", data="system"),
    dcc.Store(id="theme-resolved", data="dark"),

    # Componente de descarga — necesario para que dcc.Download funcione
    dcc.Download(id="download-infografia"),

    sidebar,
    html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div(id="page-title", style={
                    "fontSize":"18px","fontWeight":"800","color":C_TEXT
                }),
                html.Div("ADY1104 · Visualización de Datos · Duoc UC 2026", style={
                    "fontSize":"11px","color":C_MUTED
                }),
            ]),
            # Botones de acción: tema + descarga infografía
            html.Div([
                html.Button("Auto", id="theme-toggle-btn", n_clicks=0, className="sv-theme-toggle"),
                html.Button(
                    "⬇ Descargar resumen",
                    id="btn-descargar-infografia",
                    n_clicks=0,
                    style={
                        "fontSize":"11px",
                        "fontWeight":"600",
                        "color":"#E2E8F0",
                        "backgroundColor":"#7C3AED",
                        "border":"none",
                        "borderRadius":"8px",
                        "padding":"7px 14px",
                        "cursor":"pointer",
                        "marginLeft":"10px",
                        "letterSpacing":"0.2px",
                        "transition":"opacity 0.15s",
                    }
                ),
            ], style={"display":"flex","alignItems":"center"}),
        ], className="sv-header", style={
            "padding":"20px 24px 16px",
            "borderBottom":f"1px solid {C_BORDER}",
            "marginBottom":"20px",
            "display":"flex","justifyContent":"space-between","alignItems":"flex-start"
        }),

        # KPIs
        html.Div(id="kpi-row", className="sv-kpi-row", style={"padding":"0 24px","marginBottom":"20px"}),

        # Contenido dinámico
        html.Div(id="tab-content", className="sv-tab-content", style={"padding":"0 24px 32px"}),

    ], className="sv-main", style={"marginLeft":"200px","backgroundColor":BG_MAIN,"minHeight":"100vh","fontFamily":FONT}),

], style={"backgroundColor":BG_MAIN,"fontFamily":FONT})


# ── TEMA (claro / oscuro / automático) ──────────────────────────
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
        ("Total Contenidos",   f"{total:,}",       "catálogo completo",        C_HIGHLIGHT),
        ("Películas",          f"{pel:,}",          f"{pel/total*100:.0f}% del total", C_MOVIE),
        ("Series",             f"{ser:,}",          f"{ser/total*100:.0f}% del total", C_SHOW),
        ("Países",             f"{paises}",          "países productores",       C_ACCENT4),
        ("Idiomas",            f"{idiomas}",         "idiomas disponibles",      C_ACCENT3),
        ("Popularidad Prom.",  f"{pop_avg:.1f}",    "índice TMDB relativo",     C_ACCENT5),
        ("Calificación Prom.", f"{val_avg:.2f}/10", "escala 0–10",              C_ACCENT6),
    ]

    cols = []
    for label, val, sub, color in kpis:
        cols.append(
            dbc.Col(kpi_card(label, val, sub, color), xs=6, sm=4, md=3, style={"marginBottom":"0"})
        )
    return dbc.Row(cols, className="g-2")


# ── CALLBACK: DESCARGA INFOGRAFÍA ────────────────────────────────
@app.callback(
    Output("download-infografia", "data"),
    Input("btn-descargar-infografia", "n_clicks"),
    State("filtro-tipo", "value"),
    State("filtro-years", "value"),
    prevent_initial_call=True,
)
def descargar_infografia(n_clicks, tipo, year_range):
    """
    Al hacer click en el botón, genera la infografía con los filtros
    activos en ese momento y la envía como PNG al browser.
    """
    png_b64 = generar_infografia_png(tipo, year_range)
    png_bytes = base64.b64decode(png_b64)

    tipo_label = tipo.lower().replace("é","e").replace("í","i") if tipo and tipo != "Todos" else "catalogo"
    y0 = int(year_range[0]) if year_range else YEAR_MIN
    y1 = int(year_range[1]) if year_range else YEAR_MAX
    filename = f"streamview_resumen_{tipo_label}_{y0}_{y1}.png"

    return dcc.send_bytes(png_bytes, filename)


@app.callback(
    Output("tab-content","children"),
    Input("nav-tab","value"),
    Input("filtro-tipo","value"),
    Input("filtro-years","value"),
    Input("filtro-top","value"),
    Input("theme-resolved","data"),
)
def render_tab(tab, tipo, years, top_n, theme):
    df = filtrar(tipo, years)

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
        lay = base_layout("¿Cómo se distribuye el catálogo?", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay["xaxis"]["title"] = "Cantidad de títulos"
        lay["yaxis"]["title"] = ""
        lay["xaxis"]["showgrid"] = False
        fig_tipo.update_layout(**lay, xaxis_range=[0, counts["Cantidad"].max()*1.22])

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
                colorscale=[[0, "rgba(124,58,237,0.35)"],[1, C_ACCENT1]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_g["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay2 = base_layout(f"Top {top_n} géneros — ¿Cuáles dominan el catálogo?", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay2["xaxis"]["title"] = "Cantidad de contenidos"
        lay2["yaxis"]["title"] = ""
        lay2["xaxis"]["showgrid"] = False
        lay2["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_gen.update_layout(**lay2, xaxis_range=[0, top_g["Cantidad"].max()*1.18])

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
                dbc.Col(card(dcc.Graph(figure=fig_gen, config={"displayModeBar":False,"responsive":True},
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
                colorscale=[[0, "rgba(6,182,212,0.3)"],[1, C_ACCENT2]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_p["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} títulos<extra></extra>",
        ))
        lay_p = base_layout(f"Top {top_n} países productores", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
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
                colorscale=[[0, "rgba(244,114,182,0.3)"],[1, C_ACCENT3]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=lang_s["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay_l = base_layout(f"Top {top_n} idiomas disponibles", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
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
                dbc.Col(card(dcc.Graph(figure=fig_paises, config={"displayModeBar":False,"responsive":True},
                             style={"height":"360px"}), padding="16px"), xs=12, lg=7),
                dbc.Col(card(dcc.Graph(figure=fig_lang,   config={"displayModeBar":False,"responsive":True},
                             style={"height":"360px"}), padding="16px"), xs=12, lg=5),
            ], className="g-3"),
            nota,
        ])

    # ══ POPULARIDAD ═══════════════════════════════════════════════
    elif tab == "tab-pop":

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
                    text=texto,
                    textposition=pos_texto,
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
        lay_sc = base_layout("¿Los más populares son los mejor valorados?", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_sc["xaxis"]["title"] = "Popularidad (TMDB)"
        lay_sc["yaxis"]["title"] = "Calificación (0–10)"
        lay_sc["yaxis"]["range"] = [0, 10.5]
        fig_sc.update_layout(**lay_sc)

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
        pgdf = pd.DataFrame(rows_pg)
        top_pg = pgdf.groupby("Género")["popularity"].mean().nlargest(top_n).reset_index()
        top_pg.columns = ["Género","Popularidad promedio"]
        top_pg = top_pg.sort_values("Popularidad promedio")

        fig_pg = go.Figure(go.Bar(
            x=top_pg["Popularidad promedio"], y=top_pg["Género"],
            orientation="h",
            marker=dict(
                color=top_pg["Popularidad promedio"],
                colorscale=[[0,"rgba(16,185,129,0.3)"],[1,C_ACCENT4]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_pg["Popularidad promedio"].round(1),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        ))
        lay_pg = base_layout(f"Top {top_n} géneros por popularidad promedio", bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
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

        rows_cp = []
        for _, row in df.dropna(subset=["country"]).iterrows():
            for c in row["country"].split(", "):
                rows_cp.append({"País": c.strip(), "vote_average": row["vote_average"]})
        cpdf = pd.DataFrame(rows_cp)
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
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_cp["Calificación promedio"].round(2),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
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

        bloque_hallazgo = []
        if insight_banner is not None:
            bloque_hallazgo = [
                insight_banner,
                dbc.Row([
                    dbc.Col(card(dcc.Graph(figure=fig_bump, config={"displayModeBar": False, "responsive": True},
                                 style={"height": "340px"}), padding="16px"), width=12),
                ], className="g-3 mb-3"),
            ]

        return html.Div([
            *bloque_hallazgo,
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_sc, config={"displayModeBar":False,"responsive":True},
                             style={"height":"320px"}), padding="16px"), xs=12, lg=7),
                dbc.Col(card(dcc.Graph(figure=fig_pop, config={"displayModeBar":False,"responsive":True},
                             style={"height":"320px"}), padding="16px"), xs=12, lg=5),
            ], className="g-3 mb-3"),
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_pg, config={"displayModeBar":False,"responsive":True},
                             style={"height":"280px"}), padding="16px"), width=12),
            ], className="mb-3"),
            nota_pop,
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