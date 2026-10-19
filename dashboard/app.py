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
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_bootstrap_components as dbc

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
def generar_infografia_png(tipo, year_range, top_n=5):
    """
    Infografía dashboard: fondo #F8F9FA, tarjetas con borde sutil,
    rankings visuales con círculos numerados del 1 al 5.
    Layout: header | 2 KPIs | 2×2 rankings | footer
    Formato: 10×14 pulgadas a 130 dpi.
    """
    from collections import Counter
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches_local

    TOP = min(int(top_n) if top_n else 5, 5)

    df = filtrar(tipo, year_range)
    y0 = int(year_range[0]) if year_range else YEAR_MIN
    y1 = int(year_range[1]) if year_range else YEAR_MAX

    # ── Paleta
    BG       = "#F8F9FA"
    BG_CARD  = "#FFFFFF"
    ACCENT   = "#7C3AED"
    TEAL     = "#06B6D4"
    GREEN    = "#10B981"
    BLUE     = "#3B82F6"
    AMBER    = "#F5A623"
    TEXT     = "#0F172A"
    TEXT2    = "#334155"
    MUTED    = "#94A3B8"
    DIVIDER  = "#E2E8F0"
    BORDER   = "#D1D5DB"

    tipo_label = tipo if tipo and tipo != "Todos" else "Catálogo completo"

    # ── Datos
    total   = len(df)
    n_pel   = int(len(df[df["type_label"] == "Película"]))
    n_ser   = int(len(df[df["type_label"] == "Serie"]))
    val_avg = df["vote_average"].mean() if total > 0 else 0

    if tipo == "Película":
        kpi_sub = f"{n_pel:,} películas"
    elif tipo == "Serie":
        kpi_sub = f"{n_ser:,} series"
    else:
        kpi_sub = f"{n_pel:,} películas  ·  {n_ser:,} series"

    # Géneros por catálogo
    rows_g = []
    for _, row in df.dropna(subset=["genres"]).iterrows():
        for g in row["genres"].split(", "):
            rows_g.append(g)
    generos_cnt = Counter(rows_g)
    top5_gen_cat = generos_cnt.most_common(TOP)

    # Géneros por popularidad
    rows_gp = []
    for _, row in df.dropna(subset=["genres"]).iterrows():
        for g in row["genres"].split(", "):
            rows_gp.append({"Género": g, "pop": row["popularity"]})
    gpdf = pd.DataFrame(rows_gp) if rows_gp else pd.DataFrame(columns=["Género","pop"])
    if len(gpdf) > 0:
        _gp = gpdf.groupby("Género")["pop"].mean().nlargest(TOP).reset_index()
        top5_gen_pop = list(zip(_gp["Género"], _gp["pop"]))
    else:
        top5_gen_pop = []

    # Países por producción
    paises_s = df["country"].dropna().str.split(", ").explode().str.strip()
    top5_paises_prod = paises_s.value_counts().head(TOP)
    top5_paises_prod_list = list(zip(top5_paises_prod.index, top5_paises_prod.values))

    # Países por popularidad
    rows_cp = []
    for _, row in df.dropna(subset=["country"]).iterrows():
        for c in row["country"].split(", "):
            rows_cp.append({"País": c.strip(), "pop": row["popularity"]})
    cpdf = pd.DataFrame(rows_cp) if rows_cp else pd.DataFrame(columns=["País","pop"])
    if len(cpdf) > 0:
        cp_stats = cpdf.groupby("País")["pop"].agg(["mean","count"]).reset_index()
        cp_stats = cp_stats[cp_stats["count"] >= 5]
        _cp = cp_stats.nlargest(TOP, "mean")
        top5_paises_pop = list(zip(_cp["País"], _cp["mean"]))
    else:
        top5_paises_pop = []

    # ── helpers ──────────────────────────────────────────────────────

    def card_ax(ax, bg=BG_CARD, border=BORDER):
        """Tarjeta con fondo blanco y borde sutil."""
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(border)
            sp.set_linewidth(0.8)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def kpi_card(ax, value_str, label_str, sub_str, color):
        """Tarjeta KPI: barra de color izquierda + número grande."""
        card_ax(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        # Barra de acento izquierda
        rect = plt.Rectangle((0, 0), 0.038, 1, transform=ax.transAxes,
                              facecolor=color, clip_on=False, zorder=3)
        ax.add_patch(rect)
        ax.text(0.10, 0.63, value_str,
                ha="left", va="center", fontsize=32, fontweight="bold",
                color=color, fontfamily="DejaVu Sans", transform=ax.transAxes)
        ax.text(0.10, 0.30, label_str,
                ha="left", va="center", fontsize=8.5, fontweight="bold",
                color=TEXT2, fontfamily="DejaVu Sans", transform=ax.transAxes)
        if sub_str:
            ax.text(0.10, 0.12, sub_str,
                    ha="left", va="center", fontsize=7.5,
                    color=MUTED, fontfamily="DejaVu Sans", transform=ax.transAxes)

    # Colores de badge por posición (1=más saturado → 5=más suave)
    BADGE_ALPHA = [1.0, 0.82, 0.64, 0.50, 0.38]

    def ranking_card(ax, items, color, title, fmt_fn=None):
        """
        Lista de ranking 1→5 con círculo numerado + nombre + valor.
        items: [(nombre, valor), ...] ya ordenados de mayor a menor.
        """
        card_ax(ax)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

        # Título de sección
        ax.text(0.5, 0.955, title,
                ha="center", va="top", fontsize=9, fontweight="bold",
                color=TEXT2, fontfamily="DejaVu Sans", transform=ax.transAxes)
        # Línea bajo el título
        ax.axhline(0.915, color=DIVIDER, linewidth=0.8, xmin=0.04, xmax=0.96)

        if not items:
            ax.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                    fontsize=9, color=MUTED, transform=ax.transAxes)
            return

        n     = len(items)
        # Espacio disponible entre 0.89 y 0.04
        y_top = 0.875
        step  = (y_top - 0.04) / max(n, 1)

        for i, (nombre, valor) in enumerate(items):
            rank  = i + 1
            yc    = y_top - i * step - step * 0.42  # centro de la fila
            alpha = BADGE_ALPHA[i] if i < len(BADGE_ALPHA) else 0.35

            # ── Círculo badge
            cx, cy = 0.095, yc
            r = 0.048
            circ = plt.Circle((cx, cy), r, transform=ax.transAxes,
                               facecolor=plt.matplotlib.colors.to_rgba(color, alpha),
                               edgecolor=color, linewidth=1.0, zorder=3,
                               clip_on=False)
            ax.add_patch(circ)
            ax.text(cx, cy, str(rank),
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if alpha > 0.55 else color,
                    fontfamily="DejaVu Sans", transform=ax.transAxes, zorder=4)

            # ── Nombre
            ax.text(0.175, yc + 0.025, nombre,
                    ha="left", va="center",
                    fontsize=9 if rank == 1 else 8.5,
                    fontweight="bold" if rank == 1 else "normal",
                    color=TEXT if rank == 1 else TEXT2,
                    fontfamily="DejaVu Sans", transform=ax.transAxes)

            # ── Valor (alineado a la derecha)
            txt = fmt_fn(valor) if fmt_fn else f"{valor:.1f}"
            ax.text(0.965, yc + 0.025, txt,
                    ha="right", va="center", fontsize=8.5,
                    color=color if rank == 1 else MUTED,
                    fontfamily="DejaVu Sans", transform=ax.transAxes,
                    fontweight="bold" if rank == 1 else "normal")

            # ── Línea separadora sutil (excepto última)
            if i < n - 1:
                ax.axhline(yc - step * 0.42, color=DIVIDER,
                           linewidth=0.5, xmin=0.04, xmax=0.96)

    # ── Canvas: 5 filas × 2 cols ────────────────────────────────────
    # Row 0: header (con padding)   height_ratio 0.80
    # Row 1: 2 KPIs                 height_ratio 0.95
    # Row 2: géneros pop | géneros cat   1.80
    # Row 3: países prod | países pop    1.80
    # Row 4: footer                 0.26
    fig = plt.figure(figsize=(10, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        5, 2,
        figure=fig,
        left=0.04, right=0.96,
        top=0.935, bottom=0.04,   # top<1 → margen superior real
        hspace=0.38, wspace=0.10,
        height_ratios=[0.80, 0.95, 1.80, 1.80, 0.26]
    )

    # ── ROW 0: Cabecera (con margen superior gracias a top=0.935) ────
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_facecolor(BG); ax_h.axis("off")
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)

    # Lado izquierdo
    ax_h.text(0.0, 0.82, "STREAMVIEW ANALYTICS",
              ha="left", va="top", fontsize=21, fontweight="bold",
              color=ACCENT, fontfamily="DejaVu Sans", transform=ax_h.transAxes)
    ax_h.text(0.0, 0.28, "Resumen del catálogo",
              ha="left", va="top", fontsize=10, color=TEXT2,
              fontfamily="DejaVu Sans", transform=ax_h.transAxes)

    # Lado derecho: filtro (pequeño) + rango de años (grande)
    ax_h.text(1.0, 0.82, tipo_label,
              ha="right", va="top", fontsize=9.5, color=MUTED,
              fontfamily="DejaVu Sans", transform=ax_h.transAxes)
    ax_h.text(1.0, 0.38, f"{y0} – {y1}",
              ha="right", va="top", fontsize=20, fontweight="bold",
              color=TEXT2, fontfamily="DejaVu Sans", transform=ax_h.transAxes)

    ax_h.axhline(0.0, color=DIVIDER, linewidth=1.5, xmin=0, xmax=1)

    # ── ROW 1: KPIs ──────────────────────────────────────────────────
    ax_k1 = fig.add_subplot(gs[1, 0])
    ax_k2 = fig.add_subplot(gs[1, 1])
    kpi_card(ax_k1, f"{total:,}", "TÍTULOS EN EL CATÁLOGO", kpi_sub, ACCENT)
    kpi_card(ax_k2, f"{val_avg:.1f}", "CALIFICACIÓN PROMEDIO", "sobre 10 puntos", GREEN)

    # ── ROW 2: Géneros (2×1) ─────────────────────────────────────────
    ax_gp = fig.add_subplot(gs[2, 0])
    ax_gc = fig.add_subplot(gs[2, 1])

    ranking_card(ax_gp, top5_gen_pop, ACCENT,
                 "TOP GÉNEROS · POPULARIDAD",
                 fmt_fn=lambda v: f"{v:.1f}")
    ranking_card(ax_gc, top5_gen_cat, TEAL,
                 "TOP GÉNEROS · CATÁLOGO",
                 fmt_fn=lambda v: f"{int(v):,}")

    # ── ROW 3: Países (2×1) ──────────────────────────────────────────
    ax_pp   = fig.add_subplot(gs[3, 0])
    ax_ppop = fig.add_subplot(gs[3, 1])

    ranking_card(ax_pp, top5_paises_prod_list, BLUE,
                 "TOP PAÍSES · PRODUCCIÓN",
                 fmt_fn=lambda v: f"{int(v):,}")
    ranking_card(ax_ppop, top5_paises_pop, AMBER,
                 "TOP PAÍSES · POPULARIDAD",
                 fmt_fn=lambda v: f"{v:.1f}")

    # ── ROW 4: Footer ────────────────────────────────────────────────
    ax_foot = fig.add_subplot(gs[4, :])
    ax_foot.set_facecolor(BG); ax_foot.axis("off")
    ax_foot.set_xlim(0, 1); ax_foot.set_ylim(0, 1)
    ax_foot.axhline(0.92, color=DIVIDER, linewidth=0.8, xmin=0, xmax=1)
    ax_foot.text(0.5, 0.35,
                 "Fuente: TMDB  ·  ADY1104 Visualización de Datos  ·  Duoc UC 2026",
                 ha="center", va="center", fontsize=7, color=MUTED,
                 fontfamily="DejaVu Sans", transform=ax_foot.transAxes)

    # ── Exportar
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
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
    State("filtro-top", "value"),
    prevent_initial_call=True,
)
def descargar_infografia(n_clicks, tipo, year_range, top_n):
    """
    Al hacer click en el botón, genera la infografía con los filtros
    activos en ese momento y la envía como PNG al browser.
    """
    png_b64 = generar_infografia_png(tipo, year_range, top_n)
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

        # ── Datos compartidos
        paises_s = df["country"].dropna().str.split(", ").explode().str.strip()

        # ── 1. Género más popular por país (gráfico principal) ──────
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

        # Popularidad promedio por país, coloreada por género dominante
        fig_gen_pais = go.Figure()
        if rows_cg:
            cgdf = pd.DataFrame(rows_cg)
            # Sólo países con ≥5 títulos
            pais_count = paises_s.value_counts()
            paises_validos = pais_count[pais_count >= 5].index.tolist()
            cgdf = cgdf[cgdf["País"].isin(paises_validos)]

            if len(cgdf) > 0:
                # Popularidad promedio del país (todos sus títulos)
                pop_pais = (
                    cgdf.groupby("País")["popularity"]
                    .mean()
                    .reset_index()
                    .rename(columns={"popularity": "PopProm"})
                )

                # Género dominante por país (mayor pop media; desempate alfabético)
                gpc = (
                    cgdf.groupby(["País", "Género"])["popularity"]
                    .mean()
                    .reset_index()
                    .sort_values(["País", "popularity", "Género"],
                                 ascending=[True, False, True])
                )
                genero_dom = (
                    gpc.groupby("País")
                    .first()
                    .reset_index()[["País", "Género"]]
                )

                # Unir: popularidad total + género dominante
                df_plot = pop_pais.merge(genero_dom, on="País")

                # Ordenar de mayor a menor popularidad, tomar top_n
                df_plot = df_plot.sort_values("PopProm", ascending=False).head(top_n)
                df_plot = df_plot.sort_values("PopProm")  # ascendente para barras horizontales

                fig_gen_pais.add_trace(go.Bar(
                    x=df_plot["PopProm"],
                    y=df_plot["País"],
                    orientation="h",
                    marker=dict(
                        color=df_plot["PopProm"],
                        colorscale=[[0, "rgba(245,166,35,0.35)"], [1, C_HIGHLIGHT]],
                        showscale=False,
                        line=dict(color="rgba(0,0,0,0)"),
                    ),
                    text=[f"{g}  ·  {v:.1f}"
                          for g, v in zip(df_plot["Género"], df_plot["PopProm"])],
                    textposition="outside",
                    textfont=dict(color=C_TEXT, size=9),
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Popularidad promedio: %{x:.1f}<br>"
                        "Género dominante: %{customdata}<extra></extra>"
                    ),
                    customdata=df_plot["Género"],
                ))

                lay_gp = base_layout(
                    f"Popularidad promedio por país — top {top_n}",
                    bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER
                )
                lay_gp["xaxis"]["title"] = "Popularidad promedio (todos los títulos del país)"
                lay_gp["yaxis"]["title"] = ""
                lay_gp["xaxis"]["showgrid"] = True
                lay_gp["margin"] = dict(l=8, r=100, t=48, b=8)
                lay_gp["showlegend"] = False
                max_pop = df_plot["PopProm"].max() if len(df_plot) > 0 else 1
                fig_gen_pais.update_layout(**lay_gp,
                                           xaxis_range=[0, max_pop * 1.28])
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

        # ── 2. Top N países productores ──────────────────────────────
        top_p = paises_s.value_counts().head(top_n).reset_index()
        top_p.columns = ["País", "Cantidad"]
        top_p = top_p.sort_values("Cantidad")

        fig_paises = go.Figure(go.Bar(
            x=top_p["Cantidad"], y=top_p["País"], orientation="h",
            marker=dict(
                color=top_p["Cantidad"],
                colorscale=[[0, "rgba(6,182,212,0.3)"], [1, C_ACCENT2]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=top_p["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} títulos<extra></extra>",
        ))
        lay_p = base_layout(f"Top {top_n} países productores",
                            bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_p["xaxis"]["title"] = "Cantidad de títulos"
        lay_p["yaxis"]["title"] = ""
        lay_p["xaxis"]["showgrid"] = False
        lay_p["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_paises.update_layout(**lay_p,
                                 xaxis_range=[0, top_p["Cantidad"].max() * 1.18])

        # ── 4. Top N idiomas (existente) ─────────────────────────────
        lang_s = df["lang_label"].value_counts().head(top_n).reset_index()
        lang_s.columns = ["Idioma", "Cantidad"]
        lang_s = lang_s.sort_values("Cantidad")

        fig_lang = go.Figure(go.Bar(
            x=lang_s["Cantidad"], y=lang_s["Idioma"], orientation="h",
            marker=dict(
                color=lang_s["Cantidad"],
                colorscale=[[0, "rgba(244,114,182,0.3)"], [1, C_ACCENT3]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0)")
            ),
            text=lang_s["Cantidad"].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color=C_TEXT, size=10),
            hovertemplate="%{y}: %{x:,} contenidos<extra></extra>",
        ))
        lay_l = base_layout(f"Top {top_n} idiomas disponibles",
                            bg=BG_CHART, text=C_TEXT, muted=C_MUTED, border=C_BORDER)
        lay_l["xaxis"]["title"] = "Cantidad de contenidos"
        lay_l["yaxis"]["title"] = ""
        lay_l["xaxis"]["showgrid"] = False
        lay_l["margin"] = dict(l=8, r=60, t=44, b=8)
        fig_lang.update_layout(**lay_l,
                               xaxis_range=[0, lang_s["Cantidad"].max() * 1.18])

        nota = html.Div(
            "ℹ Un contenido con coproducción se contabiliza en cada país participante.  "
            "El género más popular se determina por popularidad promedio (mín. 5 títulos por país).",
            style={"fontSize": "10px", "color": C_MUTED, "marginTop": "10px"}
        )

        return html.Div([
            # Gráfico principal: popularidad promedio por país (full width)
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_gen_pais,
                             config={"displayModeBar": False, "responsive": True},
                             style={"height": "420px"}), padding="16px"), width=12),
            ], className="g-3 mb-3"),
            # Fila secundaria: países productores + idiomas
            dbc.Row([
                dbc.Col(card(dcc.Graph(figure=fig_paises,
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