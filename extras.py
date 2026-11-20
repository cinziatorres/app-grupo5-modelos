"""Páginas añadidas a la app integrada (Grupo 5).

- Inicio y objetivos: objetivos específicos cuantificados con resultado en vivo.
- Datos y análisis descriptivo: estadísticos por variable, estandarización, correlación y PCA.
- Tablero BI integrador: réplica de las 4 páginas del reporte Power BI (PBIX).
- Comparar escenarios: evaluación prescriptiva de varios escenarios de precio y servicio.
- Fundamentación: algoritmos, regularización y referencias (se agrega a «Método y fuentes»).
Todo reutiliza el mismo motor (motor.py), los mismos modelos y los mismos artefactos.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import motor as M
from modelos import ROOT, fc, NAMES, COMBO, predict

INK, BASE, SCEN = "#123b5d", "#3978b3", "#cf6225"
MODEL_COLORS = {"Realidad": "#5B6573", fc.MODEL_NAIVE: "#E66101", fc.MODEL_RIDGE: "#7B3FA0",
                fc.MODEL_SARIMAX: "#1F78B4", fc.MODEL_DHR: "#33A02C", COMBO: "#B8860B"}
ALL_MODELS = NAMES + [COMBO]
STATUS_COLORS = {"Agotamiento crítico": "#D9534F", "Reordenar urgente": "#F0AD4E",
                 "Óptimo": "#5CB85C", "Sobre-stock": "#0275D8"}
STATUS_ORDER = list(STATUS_COLORS)
LAYOUT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
              margin=dict(l=40, r=30, t=60, b=90), font=dict(color="#172f46"),
              legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
NUMERIC_VARS = ["units_sold", "price_unit", "promotion_flag", "stock_available", "delivery_days",
                "is_holiday_peak", "is_holiday_week", "is_summer", "is_winter", "lag_1", "lag_2",
                "rolling_mean_4", "rolling_std_4", "momentum", "target_next_week"]
VAR_LABELS = {"units_sold": "Unidades vendidas", "price_unit": "Precio unitario", "promotion_flag": "Promoción (0/1)",
              "stock_available": "Stock disponible", "delivery_days": "Días de entrega",
              "is_holiday_peak": "Pico festivo (0/1)", "is_holiday_week": "Semana festiva (0/1)",
              "is_summer": "Verano (0/1)", "is_winter": "Invierno (0/1)", "lag_1": "Rezago 1 semana",
              "lag_2": "Rezago 2 semanas", "rolling_mean_4": "Media móvil 4 sem.",
              "rolling_std_4": "Desv. móvil 4 sem.", "momentum": "Momentum",
              "target_next_week": "Demanda semana siguiente"}


def plot(fig, **kw):
    st.plotly_chart(fig, theme=None, width="stretch", **kw)


def vline(fig, x, text=None, dash="dot"):
    if hasattr(x, "to_pydatetime"):
        x = x.to_pydatetime()
    fig.add_vline(x=x, line_dash=dash, line_color="#64748B")
    if text:
        fig.add_annotation(x=x, y=1, yref="paper", text=text, showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11, color="#475569"))


# ============================================================
# Datos compartidos (en caché)
# ============================================================
@st.cache_data(show_spinner="Cargando la base…")
def raw() -> pd.DataFrame:
    df = M.load_data()
    df["sku"] = df["sku"].astype(str)
    for c in NUMERIC_VARS:
        if c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def registry() -> pd.DataFrame:
    return M.load_registry()


@st.cache_data(show_spinner=False)
def priority_skus() -> list[str]:
    """10 SKU priorizados del informe: ≥104 semanas comunes y mayor ingreso histórico."""
    s = fc.sku_summary(raw())
    return fc.select_top_skus(s, 10)["sku"].tolist()


@st.cache_data(show_spinner=False)
def metrics() -> pd.DataFrame:
    return pd.read_csv(ROOT / "artifacts/metrics.csv")


@st.cache_data(show_spinner=False)
def elasticity() -> pd.DataFrame:
    e = pd.read_csv(ROOT / "artifacts/elasticity.csv")
    p = pd.read_csv(ROOT / "artifacts/price_scenario.csv")[["sku", "elasticity_used", "scenario_elasticity_low",
                                                              "scenario_elasticity_high", "epsilon_corrected_free",
                                                              "corrected_ci_low", "corrected_ci_high"]]
    e = e.merge(p, on="sku", how="left")
    md = raw()
    md = md[(md.units_sold > 0) & (md.price_unit > 0)]
    g = md.groupby("sku").agg(unidades_totales=("units_sold", "sum"), precio_promedio=("price_unit", "mean")).reset_index()
    e = e.merge(g, on="sku", how="left")
    e["significativo_5pct"] = np.where(e["p_cluster"] < 0.05, "Sí", "No")
    e["clasificacion"] = np.where(e["elasticity"].abs() > 1, "Demanda elástica", "Demanda inelástica")
    return e


@st.cache_data(show_spinner="Calculando el forecast de los cuatro modelos…")
def all_models_forecast(horizon: int = 12) -> pd.DataFrame:
    """Forecast de los 4 modelos por SKU desde el corte global (equivalente a BI_Forecast)."""
    data = raw()
    man = M.load_manifest()
    as_of = pd.Timestamp(man["as_of"])
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sku in registry().sku:
            meta = M.load_meta(sku)
            hist = fc.aggregate_sku(data, sku)
            gap = int((as_of - hist.week.iloc[-1]).days // 7)
            fut = fc.future_scenario(hist, horizon + gap, source_df=data)
            fut["price_unit"] = float(hist.price_unit.tail(4).mean())
            fut["is_holiday_week"] = 0
            for name in NAMES + [COMBO]:
                if name not in meta["params"]:
                    continue
                try:
                    y = M.predict_named(sku, name, hist, fut, meta)
                except Exception:
                    continue
                rows.append(pd.DataFrame({"sku": sku, "week": fut.week.iloc[gap:].to_numpy(),
                                          "model": name, "value": y[gap:], "is_winner": name == meta["winner"]}))
    return pd.concat(rows, ignore_index=True)


@st.cache_data(show_spinner=False)
def history(sku: str) -> pd.DataFrame:
    return fc.aggregate_sku(raw(), sku)


@st.cache_data(show_spinner="Calculando reposición base de los SKU…")
def default_inventory(service: float = 0.95) -> pd.DataFrame:
    """Parámetros hacia adelante de los 30 SKU con los supuestos de la base
    (L y σL de delivery_days, R = 1, stock de la última semana), sin cambio de precio."""
    rows = []
    data = raw()
    for sku in registry().sku:
        c = M.default_scenario(sku, service=service)
        r = M.calculate(c, M.forecast_base(c, data))
        b = r["inventory"].set_index("scenario").loc["Base"]
        rows.append({"sku": sku, "modelo_campeon": r["info"]["forecast_model"],
                     "forecast_12w": float(r["weekly"].base_demand.sum()),
                     "L_sem": c.lead_weeks, "sigma_L_sem": c.lead_sd, "sigma_sem": float(b.sigma_week),
                     "SS": float(b.safety_stock), "ROP": float(b.reorder_point), "S": float(b.target_stock),
                     "stock": c.stock, "pedido": float(b.suggested_order),
                     "estado": classify(c.stock, float(b.safety_stock), float(b.reorder_point), float(b.target_stock)),
                     "servicio_simulado": float(b.simulated_cycle_service)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Preparando datos del tablero de inventarios…")
def bi_inventory_rows() -> pd.DataFrame:
    """Réplica del script R de Power BI (Sofía), por fila canal-región-semana."""
    d = raw()[["sku", "week", "year", "units_sold", "stock_available", "delivery_days",
               "rolling_mean_4", "rolling_std_4"]].copy()
    z, R, sL = norm.ppf(0.95), 1, 0.1
    L = d["delivery_days"] / 7
    D = np.where(d["rolling_mean_4"].isna() | (d["rolling_mean_4"] == 0), d["units_sold"], d["rolling_mean_4"])
    sD = d["rolling_std_4"].fillna(0)
    d["Stock_Seguridad"] = np.ceil(z * np.sqrt(L * sD ** 2 + D ** 2 * sL ** 2))
    d["Punto_Reorden_ROP"] = np.ceil(D * L + d["Stock_Seguridad"])
    d["Inventario_Maximo_S"] = np.ceil(D * (L + R) + d["Stock_Seguridad"])
    return d


def classify(stock, ss, rop, smax):
    if stock < ss:
        return "Agotamiento crítico"
    if stock < rop:
        return "Reordenar urgente"
    if stock > smax:
        return "Sobre-stock"
    return "Óptimo"


@st.cache_data(show_spinner="Calculando el control histórico de inventario…")
def bi_control(service: float = 0.95) -> pd.DataFrame:
    """Control histórico por SKU y semana con las MISMAS ecuaciones del motor.
    En cada semana t solo se usa información disponible antes de t:
      d̂_t = promedio de las 4 semanas previas (pronóstico ex ante)
      σ_t = RMSE de los errores de ese pronóstico en las 12 semanas previas
      SS_t = z·√((L+R)·σ_t² + d̂_t²·σL²);  ROP_t = d̂_t·L + z·√(L·σ_t² + d̂_t²·σL²);  S_t = d̂_t·(L+R) + SS_t
    y se compara con el stock disponible (suma canal × región) y con la demanda real de L+R."""
    z, R = float(norm.ppf(service)), 1.0
    data = raw()
    out = []
    for sku in registry().sku:
        h = history(sku)[["week", "units_sold"]].reset_index(drop=True)
        L, sL = M.lead_time_from_data(sku)
        g = data[data.sku == sku]
        stock = g.groupby("week")["stock_available"].sum().reindex(h.week).to_numpy(float)
        y = h.units_sold.to_numpy(float)
        d = pd.Series(y).shift(1).rolling(4).mean().to_numpy()
        e = y - d
        sig = np.sqrt(pd.Series(e ** 2).shift(1).rolling(12, min_periods=8).mean()).to_numpy()
        ss = z * np.sqrt((L + R) * sig ** 2 + d ** 2 * sL ** 2)
        ss_l = z * np.sqrt(L * sig ** 2 + d ** 2 * sL ** 2)
        real = np.r_[y[:-1] + (L + R - 1) * y[1:], np.nan]
        out.append(pd.DataFrame({"sku": sku, "week": h.week, "year": h.week.dt.year, "demanda": y, "pronostico": d,
                                 "sigma": sig, "L": L, "sigma_L": sL, "stock": stock, "SS": ss, "ROP": d * L + ss_l,
                                 "S": d * (L + R) + ss, "demanda_real_LR": real}))
    df = pd.concat(out, ignore_index=True).dropna(subset=["SS"])
    df["estado"] = [classify(*v) for v in zip(df.stock, df.SS, df.ROP, df.S)]
    df["cubierto"] = np.where(df.demanda_real_LR.notna(), (df.demanda_real_LR <= df.S).astype(float), np.nan)
    return df


@st.cache_data(show_spinner=False)
def coverage_comparison() -> pd.DataFrame:
    """Validación histórica: % de semanas en que S cubrió la demanda real de L+R (meta = 95%).
    Compara la lógica anterior del PBIX con la lógica unificada, en las mismas semanas."""
    new = bi_control()
    old = bi_inventory_rows()
    o = old.groupby(["sku", "week"]).agg(S_ant=("Inventario_Maximo_S", "sum"), SS_ant=("Stock_Seguridad", "sum")).reset_index()
    m = new.merge(o, on=["sku", "week"]).dropna(subset=["demanda_real_LR"])
    rows = []
    for sku, g in m.groupby("sku"):
        rows.append({"sku": sku, "semanas": len(g),
                     "cobertura_anterior": float((g.demanda_real_LR <= g.S_ant).mean()),
                     "cobertura_unificada": float((g.demanda_real_LR <= g.S).mean()),
                     "SS_anterior": float(g.SS_ant.mean()), "SS_unificada": float(g.SS.mean())})
    return pd.DataFrame(rows)


def describe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        mode = s.mode()
        rows.append({"Variable": c, "Descripción": VAR_LABELS.get(c, c), "n": int(s.size), "Media": s.mean(),
                     "Mediana": s.median(), "Moda": mode.iloc[0] if len(mode) else np.nan, "Desv. estándar": s.std(),
                     "Mínimo": s.min(), "Máximo": s.max(),
                     "Coef. variación (%)": 100 * s.std() / s.mean() if s.mean() else np.nan})
    return pd.DataFrame(rows)


# ============================================================
# Página: Inicio y objetivos
# ============================================================
def page_inicio():
    st.subheader("Modelamiento predictivo y analítico de ventas por SKU")
    st.caption("Universidad de Lima · Ingeniería Industrial · Modelamiento predictivo de datos · Sección 805 · "
               "Prof. José Antonio Taquia Gutiérrez · Grupo 5: Sofía Gamboa, Valeria Guzmán, Leonardo Ledesma y Cinzia Torres")
    reg = registry()
    el = elasticity()
    inv = default_inventory()
    mt = metrics()
    hold = mt[mt.scope.eq("holdout")]
    naive_hold = hold[hold.model.eq(fc.MODEL_NAIVE)].set_index("sku")["WAPE"]
    long = reg[reg.n_weeks >= 104]
    c = st.columns(5)
    c[0].metric("Registros", f"{len(raw()):,}".replace(",", " "))
    c[1].metric("SKU modelados", f"{len(reg)}")
    c[2].metric("WAPE comprobación (mediana)", f"{reg.holdout_WAPE.median():.1f}%")
    c[3].metric("Elasticidad OLS significativa", f"{int((el.p_cluster < .05).sum())} de {len(el)}")
    cov = coverage_comparison()
    c[4].metric("Cobertura histórica de S", f"{cov.cobertura_unificada.mean():.1%}", help="Meta: 95% de semanas cubiertas")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Problema")
        st.write("En retail, el error de pronóstico se traduce en exceso de inventario, quiebres de stock o ventas "
                 "perdidas; además, cada SKU responde distinto al precio y a las promociones. La solución conecta "
                 "en una sola herramienta cuánto se venderá, qué tan sensible es la demanda al precio, qué "
                 "inventario protege el servicio y cómo priorizar decisiones.")
        st.markdown("#### Objetivo general")
        st.write("Desarrollar una solución analítica integrada por SKU que pronostique la demanda semanal, estime la "
                 "sensibilidad precio-demanda, traduzca la demanda esperada en parámetros de reposición y consolide "
                 "los resultados en un tablero de soporte a decisiones.")
    with right:
        st.markdown("#### Arquitectura")
        st.graphviz_chart("""
digraph G { rankdir=TB; bgcolor="transparent"; nodesep=.35; ranksep=.4;
 node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#123b5d", fillcolor="#EEF3F8"];
 d [label="Base semanal · 30 SKU\\n3 canales · 3 regiones", fillcolor="white"];
 f [label="1 · Forecast\\nNaive · Fourier+Ridge · SARIMAX · DHR"];
 e [label="2 · Respuesta al precio\\nOLS log-log + modelo semanal"];
 i [label="3 · Inventarios\\nrevisión periódica · simulación"];
 bi [label="4 · Tablero BI integrador", fillcolor="#123b5d", fontcolor="white"];
 s [label="Escenarios prescriptivos", fillcolor="#FFF1E8", color="#cf6225"];
 d->f; d->e; f->i [label=" demanda y error"]; e->i [label=" ε"]; f->bi; e->bi; i->bi; bi->s; }
""", width="stretch")

    st.markdown("#### Objetivos específicos, indicadores, meta y resultado")
    st.caption("La columna Resultado se calcula en vivo con los modelos y artefactos de la app.")
    w_long = long.holdout_WAPE
    beat = int((reg.set_index("sku").holdout_WAPE <= naive_hold.reindex(reg.sku).values).sum())
    sig = int((el.p_cluster < .05).sum())
    rows = [
        {"Dimensión": "1 · Forecast",
         "Objetivo específico": "Pronosticar la demanda semanal por SKU eligiendo un modelo campeón con validación temporal (calibración 4 sem., selección H12 y comprobación H12 independientes).",
         "Indicador": "WAPE (%) y MAE de comprobación",
         "Meta": "WAPE de comprobación ≤ 15% en los SKU con ≥ 104 semanas de historia (5 candidatos: 4 modelos + su combinación)",
         "Resultado": f"{int((w_long <= 15).sum())}/{len(long)} SKU cumplen (mediana {w_long.median():.1f}%); el campeón iguala o mejora al Naive en comprobación en {beat}/{len(reg)} SKU"},
        {"Dimensión": "2 · Elasticidad",
         "Objetivo específico": "Estimar la elasticidad precio-demanda por SKU con regresión log-log y controles, con IC 95% agrupado por semana, y evaluar su significancia.",
         "Indicador": "β ln(precio), p-value agrupado, IC 95%, R²",
         "Meta": "30/30 SKU estimados con IC 95%; significancia evaluada al 5%",
         "Resultado": f"{int(el.ci_low.notna().sum())}/30 SKU con IC; R² entre {el.r2.min():.2f} y {el.r2.max():.2f}; {sig} SKU significativos al 5%"},
        {"Dimensión": "3 · Inventarios",
         "Objetivo específico": "Traducir el forecast y su error fuera de muestra en stock de seguridad, punto de reorden, stock objetivo y pedido (revisión periódica), con plazo de entrega tomado de los datos.",
         "Indicador": "Cobertura histórica de S (% de semanas en que S ≥ demanda real de L+R) y SS promedio",
         "Meta": "Cobertura entre 95% y 97% (calibrada al nivel de servicio) con menos SS que la lógica anterior del PBIX",
         "Resultado": f"Cobertura {cov.cobertura_unificada.mean():.1%} (anterior {cov.cobertura_anterior.mean():.1%}); SS {100*(cov.SS_unificada.mean()/cov.SS_anterior.mean()-1):+.1f}% vs anterior"},
        {"Dimensión": "4 · Integración BI",
         "Objetivo específico": "Integrar forecast, elasticidad e inventario en un tablero filtrable por SKU, modelo y periodo, y evaluar escenarios.",
         "Indicador": "SKU integrados y escenarios comparados",
         "Meta": "100% de SKU con las 3 métricas en una vista ejecutiva y ≥ 3 escenarios comparados",
         "Resultado": "30/30 SKU integrados; 4 vistas BI + comparación de escenarios de precio y servicio"},
    ]
    st.table(pd.DataFrame(rows).set_index("Dimensión"))
    st.caption("Fuente de datos: FMCG daily sales data 2022-2024 (Kaggle), agregada a frecuencia semanal.")


# ============================================================
# Página: Datos y análisis descriptivo
# ============================================================
def page_datos():
    st.subheader("Datos y análisis descriptivo")
    df = raw()
    sec = st.segmented_control("Sección", ["Estadística descriptiva", "Correlación", "Estandarización",
                                           "Reducción de dimensiones (PCA)"],
                               default="Estadística descriptiva", label_visibility="collapsed") or "Estadística descriptiva"
    if sec == "Estadística descriptiva":
        a, b, c, d = st.columns(4)
        a.metric("Registros", f"{len(df):,}".replace(",", " "))
        b.metric("SKU", df.sku.nunique())
        c.metric("Canales × regiones", f"{df.channel.nunique()} × {df.region.nunique()}")
        d.metric("Periodo", f"{df.week.min():%d/%m/%Y} – {df.week.max():%d/%m/%Y}")
        scope = st.radio("Alcance", ["Base completa", "10 SKU priorizados", "SKU de la barra lateral"], horizontal=True)
        x = df
        if scope == "10 SKU priorizados":
            x = df[df.sku.isin(priority_skus())]
        elif scope == "SKU de la barra lateral":
            x = df[df.sku == st.session_state.get("sku")]
        t = describe(x, NUMERIC_VARS)
        st.dataframe(t.style.format({k: "{:,.3f}" for k in t.columns if k not in ("Variable", "Descripción", "n")} | {"n": "{:,.0f}"}),
                     hide_index=True, width="stretch")
        st.download_button("Descargar estadística descriptiva (CSV)", t.to_csv(index=False).encode("utf-8"),
                           "estadistica_descriptiva.csv", "text/csv")
        cat = []
        for c_ in ["sku", "channel", "region", "lifecycle_stage"]:
            vc = x[c_].value_counts()
            cat.append({"Variable": c_, "Categorías": x[c_].nunique(), "Moda": vc.index[0], "Frecuencia de la moda": int(vc.iloc[0])})
        st.dataframe(pd.DataFrame(cat), hide_index=True, width="stretch")
        var = st.selectbox("Distribución de la variable", NUMERIC_VARS, format_func=lambda v: f"{VAR_LABELS[v]} ({v})")
        s = pd.to_numeric(x[var], errors="coerce").dropna()
        c1, c2 = st.columns([1.6, 1])
        f = px.histogram(s, nbins=40, color_discrete_sequence=[BASE], title=f"Histograma · {VAR_LABELS[var]}")
        vline(f, s.mean(), f"Media {s.mean():.2f}", "dash")
        f.update_layout(**LAYOUT, showlegend=False, height=380)
        with c1:
            plot(f)
        f = px.box(s, points="outliers", color_discrete_sequence=[BASE], title="Diagrama de caja")
        f.update_layout(**LAYOUT, showlegend=False, height=380)
        with c2:
            plot(f)
        st.caption("Se excluyen 5 columnas «Unnamed» vacías; existe un valor no numérico (`q`) en units_sold de MI-006 "
                   "(primera semana) que se trata como faltante.")
    elif sec == "Correlación":
        cols = ["units_sold", "price_unit", "promotion_flag", "stock_available", "delivery_days", "is_holiday_week",
                "is_summer", "is_winter", "lag_1", "rolling_mean_4", "rolling_std_4", "momentum"]
        m = st.radio("Coeficiente", ["pearson", "spearman"], horizontal=True)
        corr = df[cols].corr(method=m)
        f = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                      title="Matriz de correlación")
        f.update_layout(**{**LAYOUT, "legend": {}}, height=620)
        plot(f)
        st.caption("Diagnóstico descriptivo y de redundancia (p. ej. lag_1 ↔ rolling_mean_4); no implica causalidad.")
    elif sec == "Estandarización":
        st.markdown("Antes de Ridge, Lasso/regresiones penalizadas y PCA se aplica la estandarización z-score "
                    "(`StandardScaler`): **z = (x − μ) / σ**, que deja cada variable con media 0 y desviación 1.")
        cols = ["units_sold", "price_unit", "promotion_flag", "stock_available", "delivery_days", "rolling_mean_4"]
        X = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        Z = (X - X.mean()) / X.std(ddof=0)
        t = pd.DataFrame({"Variable": cols, "Media original": X.mean().values, "Desv. original": X.std(ddof=0).values,
                          "Mín.": X.min().values, "Máx.": X.max().values, "Media z": Z.mean().round(10).values,
                          "Desv. z": Z.std(ddof=0).values, "Mín. z": Z.min().values, "Máx. z": Z.max().values})
        st.dataframe(t.style.format({k: "{:,.3f}" for k in t.columns if k != "Variable"}), hide_index=True, width="stretch")
        v = st.selectbox("Comparar distribución", cols, index=1)
        c1, c2 = st.columns(2)
        for col_, data_, title, color in [(c1, X[v], "escala original", "#94A3B8"), (c2, Z[v], "estandarizada (z)", BASE)]:
            f = px.histogram(data_, nbins=40, title=f"{v} · {title}", color_discrete_sequence=[color])
            f.update_layout(**LAYOUT, showlegend=False, height=320)
            with col_:
                plot(f)
        st.info("Se usa en Fourier + Ridge (StandardScaler dentro del pipeline), en la eliminación de columnas redundantes "
                "del OLS (rango con variables escaladas) y en el PCA.")
    else:
        st.markdown("Matriz **SKU × 10 indicadores** (ingresos, unidades, precio promedio, participación promocional, stock "
                    "promedio, días de entrega, WAPE de comprobación del campeón, elasticidad OLS, stock de seguridad y "
                    "stock objetivo). Se estandariza y se aplica PCA reteniendo el mínimo de componentes que explique el umbral.")
        c1, c2 = st.columns(2)
        thr = c1.slider("Umbral de varianza acumulada", .6, .95, .8, .05)
        group = c2.radio("SKU", ["30 SKU", "10 SKU priorizados"], horizontal=True)
        summ = fc.sku_summary(df).set_index("sku")
        el = elasticity().set_index("sku")
        inv = default_inventory().set_index("sku")
        reg = registry().set_index("sku")
        skus = reg.index.tolist() if group == "30 SKU" else priority_skus()
        Mx = pd.DataFrame([{
            "sku": s, "Ingresos": summ.loc[s, "total_revenue"], "Unidades": summ.loc[s, "total_units"],
            "Precio promedio": df.loc[df.sku == s, "price_unit"].mean(),
            "Participación promocional": df.loc[df.sku == s, "promotion_flag"].mean(),
            "Stock promedio": df.loc[df.sku == s, "stock_available"].mean(),
            "Días de entrega": df.loc[df.sku == s, "delivery_days"].mean(),
            "WAPE campeón": reg.loc[s, "holdout_WAPE"], "Elasticidad": el.loc[s, "elasticity"],
            "Stock de seguridad": inv.loc[s, "SS"], "Stock objetivo": inv.loc[s, "S"]} for s in skus]).set_index("sku")
        Z = StandardScaler().fit_transform(Mx.values)
        pca = PCA().fit(Z)
        evr, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
        k = int(np.argmax(cum >= thr) + 1)
        m1, m2, m3 = st.columns(3)
        m1.metric("Variables originales", Mx.shape[1])
        m2.metric("Componentes retenidos", k)
        m3.metric("Varianza explicada", f"{100 * cum[k - 1]:.2f}%")
        pcs = [f"PC{i + 1}" for i in range(len(evr))]
        f = go.Figure()
        f.add_bar(x=pcs, y=100 * evr, name="Individual", marker_color=INK)
        f.add_scatter(x=pcs, y=100 * cum, name="Acumulada", mode="lines+markers", line=dict(color=SCEN))
        f.add_hline(y=100 * thr, line_dash="dash")
        f.update_layout(**LAYOUT, height=380, yaxis_title="% varianza", title="Varianza explicada por componente")
        plot(f)
        lo = pd.DataFrame(pca.components_[:k].T, index=Mx.columns, columns=pcs[:k])
        sc = pd.DataFrame(pca.transform(Z)[:, :max(k, 2)], index=Mx.index, columns=pcs[:max(k, 2)]).reset_index()
        c1, c2 = st.columns([1, 1.2])
        f = px.imshow(lo, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", title="Cargas")
        f.update_layout(**{**LAYOUT, "legend": {}}, height=440)
        with c1:
            plot(f)
        f = px.scatter(sc, x="PC1", y="PC2", text="sku", title="SKU en PC1–PC2")
        f.update_traces(marker=dict(size=11, color=INK), textposition="top center")
        f.update_layout(**LAYOUT, height=440)
        with c2:
            plot(f)
        st.caption("PCA es exploratorio: resume el perfil de los SKU; no reemplaza los modelos de cada dimensión.")


# ============================================================
# Página: Tablero BI integrador (réplica del PBIX)
# ============================================================
def page_bi():
    st.subheader("Tablero BI integrador")
    st.caption("Réplica web de las 4 páginas del reporte Power BI (forecast, inventarios, elasticidad y mapa ejecutivo). "
               "En Power BI cada visual R recibe el dataset filtrado por los segmentadores; aquí los filtros cumplen ese rol.")
    reg = registry()
    all_skus = reg.sku.sort_values().tolist()
    top = priority_skus()
    grp = st.radio("Conjunto de SKU", ["10 SKU priorizados (como el PBIX)", "30 SKU"], horizontal=True, key="bi_grp")
    pool = top if grp.startswith("10") else all_skus
    t1, t2, t3, t4, t5 = st.tabs(["Forecast", "Inventarios", "Elasticidad", "Mapa ejecutivo", "Descargas"])

    with t1:
        fl = all_models_forecast()
        c1, c2 = st.columns([2, 1])
        sel = c1.multiselect("Segmentador · SKU", pool, default=[], placeholder="Todos los SKU del conjunto", key="bi_f_sku") or pool
        models = c2.multiselect("Segmentador · Modelo", ALL_MODELS, default=ALL_MODELS, key="bi_f_model") or ALL_MODELS
        single = len(sel) == 1
        f = fl[fl.sku.isin(sel) & fl.model.isin(models)].sort_values(["sku", "model", "week"]).copy()
        f["t"] = f.groupby(["sku", "model"]).cumcount() + 1
        parts = []
        for s in sel:
            h = history(s)[["week", "units_sold"]].tail(79).copy()
            h["t"] = np.arange(-len(h) + 1, 1)
            parts.append(h)
        h = pd.concat(parts)
        xcol = "week" if single else "t"
        act = h.groupby(xcol)["units_sold"].sum().reset_index()
        agg = f.groupby([xcol, "model"])["value"].sum().reset_index()
        lx, lv = act[xcol].iloc[-1], act["units_sold"].iloc[-1]
        left, right = st.columns([1.05, 1])
        with left:
            fig = go.Figure()
            fig.add_scatter(x=act[xcol], y=act.units_sold, name="Realidad", line=dict(color=MODEL_COLORS["Realidad"]))
            dashes = {fc.MODEL_NAIVE: "dash", fc.MODEL_RIDGE: "solid", fc.MODEL_SARIMAX: "solid", fc.MODEL_DHR: "dashdot", COMBO: "dot"}
            for m in models:
                d = agg[agg.model == m].sort_values(xcol)
                fig.add_scatter(x=[lx, *d[xcol]], y=[lv, *d.value], name=m, line=dict(color=MODEL_COLORS[m], dash=dashes[m], width=2.5))
            if single:
                vline(fig, lx, "Inicio forecast")
            else:
                fig.add_vline(x=0, line_dash="dot", line_color="#64748B")
            fig.update_layout(**LAYOUT, height=480, yaxis_title="Unidades vendidas",
                              title=f"Comparativa de modelos · {sel[0] if single else f'{len(sel)} SKU (consolidado)'}",
                              xaxis_title="Semana" if single else "Semanas relativas al inicio del forecast")
            plot(fig)
            if not single:
                st.caption("Con varios SKU se suma por semana y modelo, alineando cada SKU por su última semana observada.")
        with right:
            pr = f.groupby(["sku", "model"])["value"].sum().reset_index()
            order = pr.groupby("sku")["value"].mean().sort_values().index.tolist()
            win = reg.set_index("sku").winner
            pr["★"] = np.where(pr.apply(lambda r: win[r.sku] == r.model, axis=1), "★", "")
            fig = px.bar(pr, x="value", y="sku", color="model", barmode="group", orientation="h", text="★",
                         color_discrete_map=MODEL_COLORS, category_orders={"sku": order, "model": ALL_MODELS},
                         labels={"value": "Unidades proyectadas (12 semanas)", "sku": "SKU", "model": "Modelo"},
                         title="Demanda proyectada por SKU (★ = campeón)")
            fig.update_traces(textposition="outside")
            fig.update_layout(**LAYOUT, height=max(480, 26 * len(sel) + 200))
            plot(fig)
        mt = metrics()
        sc = st.radio("WAPE de", ["holdout", "selection"], horizontal=True, key="bi_f_scope",
                      format_func=lambda s: "Comprobación (holdout)" if s == "holdout" else "Selección")
        pv = mt[mt.scope.eq(sc) & mt.sku.isin(sel) & mt.model.isin(models)].pivot(index="sku", columns="model", values="WAPE")
        st.dataframe(pv.style.format("{:.2f}").highlight_min(axis=1, color="#C8E6C9"), width="stretch")

    with t2:
        ctl = bi_control()
        c1, c2 = st.columns([1, 2])
        years = sorted(ctl.year.unique())
        ys = c1.multiselect("Segmentador · Año", years, default=[], placeholder="Todos los años", key="bi_i_year")
        only = c2.toggle("Solo el SKU de la barra lateral", key="bi_i_one")
        d = ctl[ctl.sku == st.session_state.get("sku")] if only else ctl[ctl.sku.isin(pool)]
        if ys:
            d = d[d.year.isin(ys)]
        st.caption("Misma lógica que la página Inventarios, aplicada semana a semana con la información disponible en cada "
                   "momento: d̂ = promedio de las 4 semanas previas, σ = RMSE de ese pronóstico en las 12 semanas previas, "
                   "L y σL = delivery_days del SKU, R = 1 semana, servicio 95%. "
                   + ("Valores del SKU." if only else "Valores sumados de los SKU seleccionados."))
        left, right = st.columns([1.1, 1])
        with left:
            w = d.groupby("week").agg(stock=("stock", "sum"), rop=("ROP", "sum"), ss=("SS", "sum"), smax=("S", "sum"),
                                      n=("sku", "nunique")).reset_index()
            w = w[w.n == d.sku.nunique()]  # solo semanas con todos los SKU seleccionados (suma comparable)
            al = w[w.stock < w.rop]
            fig = go.Figure()
            fig.add_scatter(x=w.week, y=w.ss, fill="tozeroy", fillcolor="rgba(192,57,43,.15)", line=dict(width=0), name="Zona crítica", hoverinfo="skip")
            fig.add_scatter(x=w.week, y=w.smax, name="Stock objetivo (S)", line=dict(color="#2B5C8F", dash="dash"))
            fig.add_scatter(x=w.week, y=w.rop, name="Punto de reorden (ROP)", line=dict(color="#E67E22", width=2))
            fig.add_scatter(x=w.week, y=w.ss, name="Stock de seguridad (SS)", line=dict(color="#C0392B", dash="dot"))
            fig.add_scatter(x=w.week, y=w.stock, name="Stock disponible real", line=dict(color="#27AE60", width=2))
            if not al.empty:
                fig.add_scatter(x=al.week, y=al.stock, mode="markers", name="Alerta: stock < ROP", marker=dict(color="#D9534F", size=8))
            fig.update_layout(**LAYOUT, height=470, hovermode="x unified", yaxis_title="Unidades",
                              title="Control de inventario y alertas de reabastecimiento")
            plot(fig)
            st.caption(f"Semanas con alerta (stock < ROP): {len(al)} de {len(w)}.")
        with right:
            last = d.sort_values("week").groupby("sku").tail(1).copy()
            last["Brecha"] = last.stock - last.ROP
            fig = px.bar(last.sort_values("Brecha"), x="Brecha", y="sku", color="estado", orientation="h",
                         color_discrete_map=STATUS_COLORS, category_orders={"estado": STATUS_ORDER},
                         labels={"Brecha": "Stock disponible − ROP", "sku": "SKU", "estado": "Estado"},
                         title="Estado operativo por SKU (última semana de cada SKU)")
            fig.add_vline(x=0, line_dash="dash")
            fig.update_layout(**LAYOUT, height=max(470, 22 * len(last) + 220))
            plot(fig)
        with st.expander("Validación histórica: ¿S cubrió la demanda real en el 95% de las semanas?", expanded=False):
            cv = coverage_comparison()
            cv = cv[cv.sku.isin(pool)] if not only else cv[cv.sku == st.session_state.get("sku")]
            a1, a2, a3 = st.columns(3)
            a1.metric("Cobertura lógica anterior (PBIX)", f"{cv.cobertura_anterior.mean():.1%}", help="Meta: 95%")
            a2.metric("Cobertura lógica unificada", f"{cv.cobertura_unificada.mean():.1%}", help="Meta: 95%")
            a3.metric("Stock de seguridad promedio", f"{cv.SS_unificada.mean():,.0f} u.",
                      delta(100 * (cv.SS_unificada.mean() / cv.SS_anterior.mean() - 1), 1, "%", "vs anterior"), delta_color="inverse")
            st.dataframe(cv.style.format({"cobertura_anterior": "{:.1%}", "cobertura_unificada": "{:.1%}",
                                          "SS_anterior": "{:,.0f}", "SS_unificada": "{:,.0f}"}), hide_index=True, width="stretch")
            st.caption("Cobertura = % de semanas en que el stock objetivo S fue mayor o igual a la demanda real de las L+R "
                       "semanas siguientes. La lógica anterior sobreprotege (más de 95%) porque suma stocks de seguridad "
                       "por canal-región y usa la media móvil que incluye la propia semana; la unificada se acerca a la meta "
                       "con menos inventario.")

    with t3:
        el = elasticity()
        el = el[el.sku.isin(pool)].copy()
        c1, c2 = st.columns([1, 3])
        var = c1.select_slider("Parámetro · Variación de precio", list(range(-20, 21, 5)), value=0,
                               format_func=lambda v: f"{v:+d}%", key="bi_e_var")
        base_e = c2.radio("Elasticidad para proyectar", ["OLS (como el PBIX)", "Sensibilidad del escenario (app)"],
                          horizontal=True, key="bi_e_src")
        e_used = el.elasticity if base_e.startswith("OLS") else el.elasticity_used
        el["Unidades proyectadas"] = el.unidades_totales * (1 + var / 100) ** e_used
        el["Δ unidades"] = el["Unidades proyectadas"] - el.unidades_totales
        left, right = st.columns([1, 1.1])
        with left:
            e2 = el.sort_values("elasticity")
            fig = go.Figure()
            for sig, col in [("Sí", "#1E40AF"), ("No", "#94A3B8")]:
                q = e2[e2.significativo_5pct == sig]
                fig.add_scatter(x=q.elasticity, y=q.sku, mode="markers", name=f"Significativo: {sig}", marker=dict(size=11, color=col),
                                error_x=dict(type="data", symmetric=False, array=q.ci_high - q.elasticity,
                                             arrayminus=q.elasticity - q.ci_low, color=col))
            fig.add_vline(x=0, line_dash="dash", line_color="#64748B")
            fig.update_layout(**LAYOUT, height=max(480, 20 * len(e2) + 200), xaxis_title="Elasticidad OLS (IC 95% agrupado)",
                              title="Elasticidad precio de la demanda por SKU",
                              yaxis=dict(categoryorder="array", categoryarray=e2.sku.tolist()))
            plot(fig)
        with right:
            fig = px.scatter(el, x="elasticity", y="unidades_totales", size="precio_promedio", color="significativo_5pct",
                             text="sku", size_max=24, color_discrete_map={"Sí": "#1E40AF", "No": "#94A3B8"},
                             labels={"elasticity": "Elasticidad", "unidades_totales": "Unidades totales", "significativo_5pct": "Significativo"},
                             title="Volumen vs. elasticidad (tamaño = precio promedio)")
            fig.update_traces(textposition="top center")
            fig.update_layout(**LAYOUT, height=320)
            plot(fig)
            t = el[["sku", "unidades_totales", "precio_promedio", "elasticity", "p_cluster", "elasticity_used",
                    "clasificacion", "Unidades proyectadas", "Δ unidades"]]
            t.columns = ["SKU", "Unidades", "Precio prom.", "Elasticidad OLS", "p-value", "ε escenario", "Clasificación",
                         "Unidades proyectadas", "Δ unidades"]
            st.dataframe(t.style.format({"Unidades": "{:,.0f}", "Precio prom.": "{:.2f}", "Elasticidad OLS": "{:+.4f}",
                                         "p-value": "{:.3f}", "ε escenario": "{:+.4f}", "Unidades proyectadas": "{:,.0f}",
                                         "Δ unidades": "{:+,.0f}"}), hide_index=True, width="stretch", height=260)
        st.caption("Ningún SKU tiene una elasticidad OLS significativa al 5% con errores agrupados por semana. La "
                   "«sensibilidad del escenario» es la que usa el simulador de la app (condicionada a ε ≤ 0).")

    with t4:
        inv = default_inventory()
        fl = all_models_forecast()
        wf = fl[fl.is_winner]

        el = elasticity()
        c1, c2 = st.columns(2)
        yrs = sorted(wf.week.dt.year.unique())
        ys = c1.multiselect("Segmentador · Año del forecast", yrs, default=[], placeholder="Todo el horizonte", key="bi_x_y")
        ms = c2.multiselect("Segmentador · Modelo ganador", ALL_MODELS, default=[], placeholder="Todos", key="bi_x_m")
        w = wf if not ys else wf[wf.week.dt.year.isin(ys)]
        ex = (w.groupby("sku").value.sum().rename("forecast_12w").reset_index()
              .merge(el[["sku", "elasticity", "p_cluster", "elasticity_used"]], on="sku")
              .merge(inv[["sku", "modelo_campeon", "SS", "ROP", "S", "stock", "estado"]], on="sku")
              .merge(registry()[["sku", "holdout_WAPE"]], on="sku"))
        ex = ex[ex.sku.isin(pool)]
        if ms:
            ex = ex[ex.modelo_campeon.isin(ms)]
        if ex.empty:
            st.warning("No hay SKU con esa combinación de filtros.")
        else:
            k = st.columns(4)
            k[0].metric("Demanda proyectada", f"{ex.forecast_12w.sum():,.0f} u.")
            k[1].metric("WAPE prom. del campeón", f"{ex.holdout_WAPE.mean():.2f}%")
            k[2].metric("SKU elasticidad sig. (5%)", int((ex.p_cluster < .05).sum()))
            k[3].metric("SKU a reordenar", int(ex.estado.isin(["Agotamiento crítico", "Reordenar urgente"]).sum()))
            fig = px.scatter(ex, x="elasticity", y="forecast_12w", size="ROP", color="estado", text="sku", size_max=40,
                             color_discrete_map=STATUS_COLORS, category_orders={"estado": STATUS_ORDER},
                             hover_data={"modelo_campeon": True, "holdout_WAPE": ":.2f", "p_cluster": ":.3f"},
                             labels={"elasticity": "Elasticidad precio-demanda (OLS)", "forecast_12w": "Demanda proyectada acumulada",
                                     "estado": "Estado de inventario"},
                             title="Mapa integrado: demanda, inventario y precio (tamaño = ROP)")
            fig.add_vline(x=0, line_width=1)
            fig.update_traces(textposition="top center")
            fig.update_layout(**LAYOUT, height=520)
            plot(fig)
            ex["Acción sugerida"] = np.select(
                [ex.estado.isin(["Agotamiento crítico", "Reordenar urgente"]), ex.estado.eq("Sobre-stock")],
                ["Emitir orden de reposición", "Frenar compras / evaluar promoción"], default="Mantener política")
            show = ex[["sku", "modelo_campeon", "holdout_WAPE", "forecast_12w", "elasticity", "p_cluster", "elasticity_used",
                       "SS", "ROP", "S", "stock", "estado", "Acción sugerida"]]
            show.columns = ["SKU", "Modelo campeón", "WAPE %", "Forecast 12 sem.", "Elasticidad OLS", "p-value",
                            "ε escenario", "SS", "ROP", "Stock objetivo", "Stock actual", "Estado", "Acción sugerida"]
            st.dataframe(show.style.format({"WAPE %": "{:.2f}", "Forecast 12 sem.": "{:,.0f}", "Elasticidad OLS": "{:+.3f}",
                                            "p-value": "{:.3f}", "ε escenario": "{:+.4f}", "SS": "{:,.0f}", "ROP": "{:,.0f}",
                                            "Stock objetivo": "{:,.0f}", "Stock actual": "{:,.0f}"}), hide_index=True, width="stretch")
            st.caption("Todo con una sola lógica (la del motor): forecast del campeón, σ = RMSE fuera de muestra, L y σL de "
                       "delivery_days, R = 1, servicio 95% y stock de la última semana del Excel.")

    with t5:
        st.write("Tablas que alimentan este tablero (equivalentes a las tablas BI_* del PBIX).")
        hist_all = pd.concat([history(s)[["week", "units_sold"]].assign(sku=s) for s in all_skus])
        bi_fc = pd.concat([hist_all.rename(columns={"units_sold": "value"}).assign(record_type="Actual", model="Actual", is_winner=False),
                           all_models_forecast().assign(record_type="Forecast")])[["sku", "week", "record_type", "model", "value", "is_winner"]]
        files = {"BI_Forecast.csv": bi_fc, "BI_Metrics.csv": metrics(), "BI_Elasticidad.csv": elasticity(),
                 "BI_Inventario.csv": default_inventory(), "BI_Control_Historico.csv": bi_control(),
                 "BI_Validacion_Cobertura.csv": coverage_comparison()}
        cols = st.columns(3)
        for i, (n, t) in enumerate(files.items()):
            cols[i % 3].download_button(n, t.to_csv(index=False).encode("utf-8-sig"), n, "text/csv", width="stretch")


# ============================================================
# Página: Comparar escenarios (prescriptivo)
# ============================================================
def page_escenarios(config, run):
    st.subheader("Comparación prescriptiva de escenarios")
    st.write("Cada fila es un escenario completo del motor (forecast campeón → respuesta al precio → simulación de "
             "reposición). Se usan los mismos supuestos de stock, plazos y elasticidad definidos en la barra lateral.")
    c1, c2 = st.columns([2, 1])
    with c1:
        default = pd.DataFrame({"Escenario": ["A · Descuento", "B · Base", "C · Alza de precio"],
                                "Cambio de precio (%)": [-10., 0., 10.], "Servicio de ciclo (%)": [97.5, 95., 95.]})
        tab = st.data_editor(default, hide_index=True, num_rows="dynamic", key="cmp_editor", column_config={
            "Cambio de precio (%)": st.column_config.NumberColumn(min_value=-40., max_value=40., step=.5),
            "Servicio de ciclo (%)": st.column_config.NumberColumn(min_value=80., max_value=99., step=.5)})
    scope = c2.radio("Alcance", ["SKU de la barra lateral", "10 SKU priorizados"], key="cmp_scope")
    c2.caption("Para el conjunto se usan los supuestos académicos por defecto de cada SKU (stock en cero salvo que lo "
               "hayas editado en la sesión).")
    tab = tab.dropna()
    tab = tab[tab.Escenario.astype(str).str.strip() != ""].drop_duplicates("Escenario")
    if tab.empty:
        st.warning("Agrega al menos un escenario.")
        return
    from dataclasses import asdict, replace
    skus = [config.sku] if scope.startswith("SKU") else priority_skus()
    rows = []
    with st.spinner("Simulando escenarios…"):
        for _, sc in tab.iterrows():
            for sku in skus:
                basecfg = config if sku == config.sku else M.Scenario(**st.session_state.profiles.get(sku, asdict(M.default_scenario(sku))))
                cfg = replace(basecfg, price_change_pct=float(sc["Cambio de precio (%)"]), service=float(sc["Servicio de ciclo (%)"]) / 100)
                r = run(cfg)
                iv = r["inventory"].set_index("scenario").loc["Precio propuesto"]
                s = r["summary"]
                rows.append({"Escenario": sc.Escenario, "sku": sku, "precio": s["price_proposed"], "unidades": s["scenario_units"],
                             "ingreso": s["scenario_revenue"], "SS": float(iv.safety_stock), "objetivo": float(iv.target_stock),
                             "pedido": float(iv.suggested_order), "servicio_sim": float(iv.simulated_cycle_service),
                             "inv_prom": float(iv.safety_stock) + s["scenario_units"] / max(cfg.horizon, 1) * cfg.review_weeks / 2})
    res = pd.DataFrame(rows)
    agg = res.groupby("Escenario", sort=False).agg(unidades=("unidades", "sum"), ingreso=("ingreso", "sum"), SS=("SS", "sum"),
                                                   objetivo=("objetivo", "sum"), pedido=("pedido", "sum"),
                                                   servicio=("servicio_sim", "mean"), inv_prom=("inv_prom", "sum")).reset_index()
    agg = agg.merge(tab.rename(columns={"Cambio de precio (%)": "dp", "Servicio de ciclo (%)": "ns"}), on="Escenario")
    ref = agg.loc[agg.dp.abs().idxmin()]
    agg["Δ ingreso %"] = 100 * (agg.ingreso / ref.ingreso - 1)
    agg["Ingreso / u. inventario"] = agg.ingreso / agg.inv_prom
    cols = st.columns(min(len(agg), 4))
    for i, r in agg.iterrows():
        with cols[i % len(cols)]:
            st.metric(r.Escenario, f"{r.ingreso:,.0f} u.m.",
                      delta(r["Δ ingreso %"], 2, "%", f"vs {ref.Escenario}") if r.Escenario != ref.Escenario else None,
                      help="Escenario de referencia" if r.Escenario == ref.Escenario else None)
            st.caption(f"Precio {r.dp:+.1f}% · servicio {r.ns:.1f}% · {r.unidades:,.0f} u. · SS {r.SS:,.0f} u. · pedido {r.pedido:,.0f} u.")
    fig = go.Figure()
    fig.add_bar(x=agg.Escenario, y=agg.ingreso, name="Ingreso potencial (u.m.)", marker_color=INK)
    fig.add_scatter(x=agg.Escenario, y=agg.SS, name="Stock de seguridad (u.)", yaxis="y2", mode="lines+markers",
                    line=dict(color=SCEN, width=3), marker=dict(size=12))
    fig.update_layout(**LAYOUT, height=400, title="Ingreso esperado vs. inventario de protección",
                      yaxis=dict(title="Ingreso (u.m.)"), yaxis2=dict(title="SS (u.)", overlaying="y", side="right"))
    plot(fig)
    t = agg[["Escenario", "dp", "ns", "unidades", "ingreso", "Δ ingreso %", "SS", "objetivo", "pedido", "servicio", "Ingreso / u. inventario"]]
    t.columns = ["Escenario", "Precio %", "Servicio %", "Unidades", "Ingreso", "Δ ingreso %", "SS", "Stock objetivo",
                 "Pedido sugerido", "Servicio simulado", "Ingreso / u. inventario"]
    st.dataframe(t.style.format({"Precio %": "{:+.1f}", "Servicio %": "{:.1f}", "Unidades": "{:,.0f}", "Ingreso": "{:,.0f}",
                                 "Δ ingreso %": "{:+.2f}", "SS": "{:,.0f}", "Stock objetivo": "{:,.0f}", "Pedido sugerido": "{:,.0f}",
                                 "Servicio simulado": "{:.1%}", "Ingreso / u. inventario": "{:,.2f}"}), hide_index=True, width="stretch")
    ok = agg[agg.servicio >= agg.ns / 100 - 0.005]
    best = (ok if not ok.empty else agg).sort_values(["Ingreso / u. inventario", "ingreso"], ascending=False).iloc[0]
    st.success(f"**Escenario sugerido: {best.Escenario}** (precio {best.dp:+.1f}%, servicio {best.ns:.1f}%). Es el que obtiene "
               f"más ingreso por unidad de inventario mantenido ({best['Ingreso / u. inventario']:,.2f}) entre los que cumplen "
               "su servicio de ciclo en la simulación.")
    st.caption("Criterio: 1) cumplir el servicio de ciclo objetivo en la simulación; 2) mayor ingreso por unidad de "
               "inventario promedio (SS + D·R/2). Sin costos ni márgenes reales no se calcula utilidad; la respuesta al "
               "precio es una sensibilidad condicionada, no un efecto causal demostrado.")
    with st.expander("Detalle por SKU y escenario"):
        st.dataframe(res, hide_index=True, width="stretch")


# ============================================================
# Fundamentación (se agrega a «Método y fuentes»)
# ============================================================
REFERENCES = [
    "Burnay, C., Lega, M., & Bouraga, S. (2024). Business intelligence and cognitive loads: Proposition of a dashboard adoption model. *Data & Knowledge Engineering, 152*, 102310. https://doi.org/10.1016/j.datak.2024.102310",
    "de Castro Moraes, T., Yuan, X.-M., & Chew, E. P. (2024). Hybrid convolutional long short-term memory models for sales forecasting in retail. *Journal of Forecasting, 43*(5), 1278–1293. https://doi.org/10.1002/for.3073",
    "Holmes, A., Illowsky, B., & Dean, S. (2023). *Introductory business statistics* (2nd ed.). OpenStax.",
    "Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/",
    "Jeon, Y., Hoang, H., Thompson, W., & Abler, D. (2024). A meta-analysis of U.S. food demand elasticities to detect the impacts of scanner data. *Applied Economic Perspectives and Policy, 46*(2), 760–780. https://doi.org/10.1002/aepp.13414",
    "Kačmáry, P., Bindzár, P., Kovalčík, J., & Ondov, M. (2024). Forecast of sales of selected food products in retail using Fourier series analysis and non-linear regression. *Foresight, 26*(3), 487–504. https://doi.org/10.1108/FS-12-2022-0168",
    "Keller, K. O., & Sethuraman, R. (2026). Promotion elasticities of national and store brands. *Journal of Retailing, 102*(3), 717–734. https://doi.org/10.1016/j.jretai.2026.01.001",
    "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting, 38*(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013",
    "Mudau, T. N., Cohen, J., & Papageorgiou, E. (2026). Tracing the effects of business intelligence on management controls and organisational performance. *Journal of Enterprise Information Management*, 1–29. https://doi.org/10.1108/JEIM-10-2024-0581",
    "Oesterreich, T. D., Anton, E., & Teuteberg, F. (2022). What translates big data into business value? *Information & Management, 59*(6), 103685. https://doi.org/10.1016/j.im.2022.103685",
    "Shi, J. (2022). Application of the model combining demand forecasting and inventory decision in feature based newsvendor problem. *Computers & Industrial Engineering, 173*, 108709. https://doi.org/10.1016/j.cie.2022.108709",
    "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B, 58*(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
    "Trapero, J. R., Holgado de Frutos, E., & Pedregal, D. J. (2024). Demand forecasting under lost sales stock policies. *International Journal of Forecasting, 40*(3), 1055–1068. https://doi.org/10.1016/j.ijforecast.2023.09.004",
    "Wolters, J., & Huchzermeier, A. (2021). Joint in-season and out-of-season promotion demand forecasting in a retail environment. *Journal of Retailing, 97*(4), 726–745. https://doi.org/10.1016/j.jretai.2021.01.003",
    "Ye, L., Xie, N., Boylan, J. E., & Shang, Z. (2024). Forecasting seasonal demand for retail: A Fourier time-varying grey model. *International Journal of Forecasting, 40*(4), 1467–1485. https://doi.org/10.1016/j.ijforecast.2023.12.006",
    "Zietsman, H. J., & van Vuuren, J. H. (2023). A generic decision support framework for inventory procurement planning in distribution centres. *Computers & Industrial Engineering, 177*, 109057. https://doi.org/10.1016/j.cie.2023.109057",
]


def section_fundamentacion():
    st.subheader("Fundamentación científica de los algoritmos")
    st.markdown("""
| Dimensión | Técnica | Por qué se eligió | Evidencia |
|---|---|---|---|
| Forecast | Seasonal Naive (s = 52) | Benchmark obligatorio; en retail estacional es difícil de superar. | Makridakis et al. (2022); Hyndman & Athanasopoulos (2021) |
| Forecast | Fourier + Ridge | Estacionalidad anual con pocos armónicos (K = 1–3) + precio, promoción y calendario; la penalización L2 controla la colinealidad. | Ye et al. (2024); Wolters & Huchzermeier (2021); Kačmáry et al. (2024) |
| Forecast | SARIMAX estacional | Dependencia temporal y estacionalidad explícita a 52 semanas con exógenas. | Hyndman & Athanasopoulos (2021) |
| Forecast | DHR (Fourier + ARMA) | Modela la autocorrelación que queda en los residuos de la regresión armónica. | Hyndman & Athanasopoulos (2021); de Castro Moraes et al. (2024) |
| Forecast | Combinación (promedio simple de los modelos admisibles) | Compite como quinto candidato con la misma regla de selección; promediar reduce el error cuando los modelos fallan en direcciones distintas. | Makridakis et al. (2022) |
| Forecast | Validación temporal H12 | Calibración, selección y comprobación en bloques sucesivos, sin fuga de información. | Hyndman & Athanasopoulos (2021) |
| Elasticidad | OLS log-log con controles y error agrupado | El coeficiente de ln(P) es la elasticidad; los controles separan promoción, stock y calendario. | Holmes et al. (2023); Jeon et al. (2024); Keller & Sethuraman (2026) |
| Inventarios | Revisión periódica (R, S): SS = z·√((L+R)σ² + d²σL²) con σ = RMSE fuera de muestra; verificación por simulación | Conecta el error del forecast con el stock de seguridad; L y σL salen de delivery_days; la misma ecuación se usa en el control histórico del BI. | Trapero et al. (2024); Zietsman & van Vuuren (2023); Shi (2022) |
| Integración | Tablero BI + escenarios | El tablero reduce la carga cognitiva y facilita la decisión. | Burnay et al. (2024); Oesterreich et al. (2022); Mudau et al. (2026) |
""")
    st.subheader("¿Se usa regularización en los modelos de regresión?")
    c1, c2 = st.columns(2)
    c1.markdown("""
**Sí, en el forecast: Ridge (L2).**
`min Σ(y − Xβ)² + α·Σβ²`, con predictores estandarizados. α ∈ {0.1, 1, 10, 100} se calibra por SKU junto con
K ∈ {1, 2, 3} y tendencia lineal/cuadrática, usando solo el bloque de calibración (antes de selección y comprobación).
""")
    c2.markdown("""
**No en la elasticidad OLS.** Penalizar ln(P) sesgaría la elasticidad hacia cero y dejaría de ser interpretable.
La parsimonia se controla retirando solo columnas exactamente redundantes (prueba de rango con variables
estandarizadas), dummies con categoría de referencia y errores agrupados por semana. Lasso (Tibshirani, 1996)
es la alternativa de selección de controles cuando se necesita descartar variables.
""")
    st.subheader("Referencias")
    for r in REFERENCES:
        st.markdown(f"- {r}")


# ============================================================
# Flechas de variación (deltas) consistentes
# ============================================================
def delta(value, decimals=0, suffix="", ref=None):
    """Texto de variación con signo explícito. Devuelve None si la variación redondeada es
    cero, para que Streamlit no muestre una flecha verde «sube» cuando no hay cambio."""
    if value is None or not np.isfinite(value):
        return None
    r = round(float(value), decimals)
    if r == 0:
        return None
    txt = f"{r:+,.{decimals}f}{suffix}"
    return f"{txt} {ref}" if ref else txt


# ============================================================
# Gráficos de la página Inventarios
# ============================================================
def _window(demand_week, t, length):
    """Demanda esperada acumulada en [t, t + length] (semanas fraccionarias)."""
    d = np.asarray(demand_week, float)
    total, x, rem = 0.0, float(t), float(length)
    while rem > 1e-9:
        w = min(int(x), len(d) - 1)
        step = min(rem, np.floor(x) + 1 - x)
        total += d[w] * step
        x += step
        rem -= step
    return total


def simulate_policy(demand_week, horizon, lead, review, ss_rop, ss_target, mode="continuous",
                    start_on_hand=None, start_pipeline=(), steps=7):
    """Simulación determinista (demanda esperada, paso diario).
    ROP(t) = demanda esperada en [t, t+L] + SS del plazo;  S(t) = demanda en [t, t+L+R] + SS.
    mode='continuous': se pide hasta S cuando la posición cae a ROP (lectura de punto de reorden).
    mode='periodic'  : cada R semanas se pide hasta S (política del motor)."""
    dt = 1 / steps
    n = int(round(horizon * steps))
    lead_steps = max(int(round(lead * steps)), 1)
    review_steps = max(int(round(review * steps)), 1)
    rop_t = np.array([_window(demand_week, i * dt, lead) + ss_rop for i in range(n + 1)])
    s_t = np.array([_window(demand_week, i * dt, lead + review) + ss_target for i in range(n + 1)])
    oh = float(s_t[0] if start_on_hand is None else start_on_hand)
    pipe = [(float(a), float(q)) for a, q in start_pipeline if q > 0]
    on_hand, position, orders = np.zeros(n + 1), np.zeros(n + 1), []
    for i in range(n + 1):
        ti = i * dt
        arrived = sum(q for a, q in pipe if a <= ti + 1e-9)
        if arrived:
            oh += arrived
            pipe = [(a, q) for a, q in pipe if a > ti + 1e-9]
        ip = oh + sum(q for _, q in pipe)
        trigger = (i % review_steps == 0) if mode == "periodic" else (ip <= rop_t[i])
        if trigger and i < n and s_t[i] - ip > 0.5:
            q = s_t[i] - ip
            pipe.append(((i + lead_steps) * dt, q))
            orders.append({"pedido": ti, "llegada": (i + lead_steps) * dt, "cantidad": q, "nivel": oh})
            ip += q
        on_hand[i], position[i] = oh, ip
        if i < n:
            oh -= demand_week[min(int(ti), len(demand_week) - 1)] * dt
    sim = pd.DataFrame({"t": np.arange(n + 1) * dt, "on_hand": on_hand, "position": position, "rop": rop_t, "S": s_t})
    return sim, pd.DataFrame(orders, columns=["pedido", "llegada", "cantidad", "nivel"])


def inventory_charts(result, run=None):
    cfg = result["config"]
    iv = result["inventory"].set_index("scenario")
    lead, review = float(cfg["lead_weeks"]), float(cfg["review_weeks"])
    horizon = int(cfg["horizon"])
    which = st.radio("Escenario a graficar", ["Precio propuesto", "Base"], horizontal=True, key="inv_chart_scen",
                     help="Base = precio de referencia; Precio propuesto = escenario de la barra lateral.")
    r = iv.loc[which]
    col = "scenario_demand" if which == "Precio propuesto" else "base_demand"
    demand = result["protection_forecast"][col].to_numpy(float)
    target, rop, ss = float(r.target_stock), float(r.reorder_point), float(r.safety_stock)
    ss_rop = float(r.safety_stock_lead) if "safety_stock_lead" in r and pd.notna(r.safety_stock_lead) else max(rop - float(r.expected_lead_demand), 0.0)

    t1, t2, t3 = st.tabs(["Dinámica del inventario: reorden y reabastecimiento", "Stock de seguridad",
                          "Stock total objetivo"])

    # ---------------- 1. Dinámica: punto de reorden y tiempo de reabastecimiento ----------------
    with t1:
        modes = {"Punto de reorden (se pide al cruzar el ROP)": "continuous",
                 "Revisión periódica cada R semanas (política del motor)": "periodic"}
        c1, c2 = st.columns([1.3, 1])
        mode_lbl = c1.radio("Lógica del gráfico", list(modes), key="inv_mode",
                            help="Ambas usan el mismo ROP, S y stock de seguridad calculados por el motor.")
        start = c2.radio("Punto de partida", ["Stock objetivo", "Stock actual ingresado"], key="inv_start",
                         help="«Stock actual» usa stock, tránsito y atrasados de la barra lateral.")
        mode = modes[mode_lbl]
        ss_target = ss
        if start == "Stock objetivo":
            pipeline = []
            if mode == "periodic":   # régimen estable: pedidos previos aún en camino
                k = 0
                while lead - k * review > 1e-9:
                    arr = lead - k * review
                    pipeline.append((arr, float(demand[0]) * min(review, arr)))
                    k += 1
            s0 = _window(demand, 0, lead + review) + ss_target
            sim, orders = simulate_policy(demand, horizon, lead, review, ss_rop, ss_target, mode,
                                          s0 - sum(q for _, q in pipeline), pipeline)
        else:
            sim, orders = simulate_policy(demand, horizon, lead, review, ss_rop, ss_target, mode,
                                          float(cfg["stock"]) - float(cfg["backorders"]), [(lead, float(cfg["in_transit"]))])
        fig = go.Figure()
        fig.add_scatter(x=sim.t, y=np.full(len(sim), ss_rop), line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=sim.t, y=np.zeros(len(sim)), fill="tonexty", fillcolor="rgba(217,83,79,.10)", line=dict(width=0),
                        name="Zona de stock de seguridad", hoverinfo="skip")
        fig.add_scatter(x=sim.t, y=sim.S, name="Stock objetivo S", line=dict(color=INK, dash="dash", shape="hv"))
        fig.add_scatter(x=sim.t, y=sim.rop, name="Punto de reorden (ROP)", line=dict(color=SCEN, width=3))
        fig.add_scatter(x=sim.t, y=np.full(len(sim), ss_rop), name=f"Stock de seguridad del plazo ({ss_rop:,.0f} u.)",
                        line=dict(color="#D9534F", dash="dot"))
        fig.add_scatter(x=sim.t, y=sim.position, name="Posición de inventario (físico + en camino)",
                        line=dict(color="#94A3B8", dash="dash", width=1.5))
        fig.add_scatter(x=sim.t, y=np.maximum(sim.on_hand, 0), name="Inventario físico disponible",
                        line=dict(color=BASE, width=3))
        short = sim[sim.on_hand < 0]
        if not short.empty:
            fig.add_scatter(x=short.t, y=np.zeros(len(short)), mode="markers", name="Quiebre de stock",
                            marker=dict(color="#D9534F", size=6, symbol="x"))
        vis = orders[orders.pedido <= horizon]
        if not vis.empty:
            fig.add_scatter(x=vis.pedido, y=np.interp(vis.pedido, sim.t, np.maximum(sim.on_hand, 0)), mode="markers",
                            name="Se emite el pedido", marker=dict(symbol="triangle-down", size=12, color=SCEN),
                            customdata=vis.cantidad, hovertemplate="Pedido de %{customdata:,.0f} u.<extra></extra>")
            arr = vis[vis.llegada <= horizon]
            fig.add_scatter(x=arr.llegada, y=[np.interp(x - 1e-6, sim.t, np.maximum(sim.on_hand, 0)) for x in arr.llegada],
                            mode="markers", name="Llega el pedido", marker=dict(symbol="triangle-up", size=12, color="#27AE60"),
                            customdata=arr.cantidad, hovertemplate="Llegan %{customdata:,.0f} u.<extra></extra>")
            o = vis.iloc[0] if len(vis) == 1 else vis.iloc[1] if vis.iloc[0].pedido == 0 else vis.iloc[0]
            if o.llegada <= horizon:
                fig.add_vrect(x0=o.pedido, x1=o.llegada, fillcolor="rgba(207,98,37,.10)", line_width=0)
                ytop = float(max(sim.S.max(), sim.position.max())) * 1.06
                fig.add_annotation(x=o.llegada, y=ytop, ax=o.pedido, ay=ytop, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowside="end+start", arrowcolor=SCEN, text="")
                fig.add_annotation(x=(o.pedido + o.llegada) / 2, y=ytop * 1.045, showarrow=False,
                                   text=f"<b>Tiempo de reabastecimiento L = {lead:g} sem.</b>", font=dict(color=SCEN, size=12))
        fig.update_layout(**LAYOUT, height=540, xaxis_title="Semanas desde el corte", yaxis_title="Unidades",
                          xaxis_range=[0, horizon],
                          title="Ciclo de inventario: punto de reorden, pedido y tiempo de reabastecimiento")
        plot(fig)
        a, b_, c, d = st.columns(4)
        a.metric("Punto de reorden (inicio)", f"{rop:,.0f} u.", help="Cuantil del nivel de servicio de la demanda durante el plazo de entrega.")
        b_.metric("Tiempo de reabastecimiento", f"{lead:g} sem.", help="Plazo medio de entrega; su variabilidad (σL) se simula en el motor.")
        c.metric("Pedidos en el periodo", f"{len(vis)}")
        d.metric("Cantidad media por pedido", f"{vis.cantidad.mean():,.0f} u." if not vis.empty else "0 u.")
        st.caption("Lectura: cuando la posición de inventario llega al punto de reorden (o en cada revisión) se emite un pedido "
                   "hasta S (▼). El pedido llega L semanas después (▲; franja naranja = tiempo de reabastecimiento). Durante "
                   "ese plazo el inventario físico baja hacia el stock de seguridad, que absorbe la variación de la demanda y "
                   "del plazo. ROP y S se recalculan con el forecast de las semanas siguientes; la trayectoria usa la "
                   "demanda esperada.")

    # ---------------- 2. Stock de seguridad ----------------
    with t2:
        left, right = st.columns(2)
        dist = result.get("distributions")
        with left:
            if dist:
                x = np.asarray(dist[which]["protection"], float)
                mu = float(r.expected_protection_demand)
                fig = go.Figure()
                fig.add_histogram(x=x, nbinsx=50, marker_color="rgba(57,120,179,.55)", name="Demanda simulada en L + R")
                vline(fig, mu, None, "dash")
                vline(fig, target, None, "solid")
                fig.add_annotation(x=mu, y=1, yref="paper", text=f"Demanda esperada {mu:,.0f}", showarrow=False,
                                   xanchor="right", yanchor="bottom", font=dict(size=11, color="#475569"))
                fig.add_annotation(x=target, y=1, yref="paper", text=f"S = {target:,.0f}", showarrow=False,
                                   xanchor="left", yanchor="bottom", font=dict(size=11, color=INK))
                ytop = np.histogram(x, bins=50)[0].max()
                fig.add_annotation(x=target, y=ytop * .55, ax=mu, ay=ytop * .55, xref="x", yref="y", axref="x", ayref="y",
                                   showarrow=True, arrowhead=2, arrowside="end+start", arrowcolor="#D9534F", text="")
                fig.add_annotation(x=(mu + target) / 2, y=ytop * .62, showarrow=False,
                                   text=f"<b>SS = {ss:,.0f} u.</b>", font=dict(color="#D9534F"))
                fig.update_layout(**LAYOUT, height=420, showlegend=False, xaxis_title="Unidades demandadas en L + R",
                                  yaxis_title="Simulaciones",
                                  title=f"Verificación por simulación: S cubre el {float(r.simulated_cycle_service):.1%} de los ciclos (meta {float(cfg['service']):.0%})")
                plot(fig)
                st.caption("Histograma de 2,000 simulaciones de la demanda durante L + R (errores reales del pronóstico "
                           "remuestreados, plazo aleatorio e incertidumbre de elasticidad). S viene de la fórmula; la "
                           "simulación comprueba que la proporción de ciclos cubiertos esté cerca del nivel de servicio.")
            else:
                st.info("Escenario restaurado: vuelve a calcularlo para ver la distribución simulada.")
        with right:
            if run is not None:
                from dataclasses import replace as _replace
                levels = [0.85, 0.90, 0.925, 0.95, 0.975, 0.99]
                base_cfg = M.Scenario(**{k: (tuple(v) if k == "holiday_weeks" else v) for k, v in cfg.items()})
                rows = []
                for lv in levels:
                    rr = run(_replace(base_cfg, service=lv))["inventory"].set_index("scenario").loc[which]
                    rows.append({"Servicio": lv, "SS": float(rr.safety_stock), "S": float(rr.target_stock),
                                 "ROP": float(rr.reorder_point)})
                sens = pd.DataFrame(rows)
                fig = go.Figure()
                fig.add_bar(x=[f"{v:.1%}" for v in sens.Servicio], y=sens.SS, name="Stock de seguridad",
                            marker_color=["#D9534F" if abs(v - float(cfg["service"])) < 1e-9 else "rgba(217,83,79,.45)" for v in sens.Servicio],
                            text=[f"{v:,.0f}" for v in sens.SS], textposition="outside")
                fig.add_scatter(x=[f"{v:.1%}" for v in sens.Servicio], y=sens.ROP, name="Punto de reorden", mode="lines+markers",
                                line=dict(color=SCEN, width=2.5))
                fig.update_layout(**LAYOUT, height=420, xaxis_title="Nivel de servicio de ciclo", yaxis_title="Unidades",
                                  title="Stock de seguridad según el nivel de servicio")
                plot(fig)
                st.caption("La barra resaltada es el nivel de servicio elegido. El SS crece de forma no lineal: los últimos "
                           "puntos de servicio son los más caros en inventario.")

    # ---------------- 3. Stock total objetivo ----------------
    with t3:
        comp = []
        for sc in ["Base", "Precio propuesto"]:
            q = iv.loc[sc]
            dl = float(q.expected_lead_demand) if "expected_lead_demand" in q else float(q.expected_protection_demand)
            dr = float(q.expected_protection_demand) - dl
            comp.append({"Escenario": sc, "Demanda en el plazo de entrega (L)": dl, "Demanda en la revisión (R)": dr,
                         "Stock de seguridad": float(q.safety_stock), "S": float(q.target_stock),
                         "pos": float(q.inventory_position), "pedido": float(q.suggested_order)})
        comp = pd.DataFrame(comp)
        fig = go.Figure()
        for c_, color in [("Demanda en el plazo de entrega (L)", "#F0AD4E"), ("Demanda en la revisión (R)", BASE),
                          ("Stock de seguridad", "#D9534F")]:
            fig.add_bar(x=comp.Escenario, y=comp[c_], name=c_, marker_color=color,
                        text=[f"{v:,.0f}" for v in comp[c_]], textposition="inside")
        fig.add_scatter(x=comp.Escenario, y=comp.S, mode="markers+text", name="Stock total objetivo S",
                        marker=dict(symbol="line-ew", size=60, line=dict(width=3, color=INK)),
                        text=[f"S = {v:,.0f}" for v in comp.S], textposition="top center")
        fig.add_scatter(x=comp.Escenario, y=comp.pos, mode="markers", name="Posición de inventario actual",
                        marker=dict(symbol="diamond", size=14, color="#27AE60", line=dict(color="white", width=1)))
        if cfg.get("capacity") is not None:
            fig.add_hline(y=float(cfg["capacity"]), line_dash="dash", line_color="#7B3FA0",
                          annotation_text=f"Capacidad {float(cfg['capacity']):,.0f}")
        fig.update_layout(**LAYOUT, barmode="stack", height=480, yaxis_title="Unidades",
                          title="Composición del stock total objetivo: S = demanda en L + demanda en R + SS")
        plot(fig)
        st.caption("Pedido sugerido = S − posición de inventario (stock + tránsito − atrasados). "
                   + " · ".join(f"{r_.Escenario}: {r_.pedido:,.0f} u." for r_ in comp.itertuples()))


def mc_tolerance(result, kind="protection", n_boot=300, seed=7):
    """Error de simulación (Monte Carlo) de la diferencia escenario − base del cuantil de servicio.
    Bootstrap pareado (mismas trayectorias en ambos escenarios). Devuelve 1.96·desv. o None."""
    dist = result.get("distributions")
    if not dist:
        return None
    p = float(result["config"]["service"])
    b = np.asarray(dist["Base"][kind], float)
    s = np.asarray(dist["Precio propuesto"][kind], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), size=(n_boot, len(b)))
    diff = np.quantile(s[idx], p, axis=1) - np.quantile(b[idx], p, axis=1)
    return float(1.96 * diff.std())


def inv_delta(diff, tol):
    """Flecha solo si el cambio supera el error de simulación; si no, sin flecha."""
    if tol is not None and abs(diff) <= max(tol, 0.5):
        return None
    return delta(diff, 0, " u.", "vs base")


def formula_panel(result):
    """Ecuaciones de la política con los números del escenario (para explicar en la exposición)."""
    iv = result["inventory"].set_index("scenario").loc["Precio propuesto"]
    cfg = result["config"]
    L, sL, R = float(cfg["lead_weeks"]), float(cfg["lead_sd"]), float(cfg["review_weeks"])
    if "sigma_week" not in iv or pd.isna(iv.get("sigma_week")):
        return
    z, sig = float(iv.z), float(iv.sigma_week)
    DL, DP = float(iv.expected_lead_demand), float(iv.expected_protection_demand)
    dP, dL = DP / (L + R), DL / L
    with st.expander("Ecuaciones de la política con los valores de este escenario", expanded=False):
        st.markdown(f"""
**Datos de entrada**
- Servicio de ciclo {float(cfg['service']):.1%} → z = Φ⁻¹({float(cfg['service']):.3f}) = **{z:.3f}**
- σ = RMSE semanal del error del pronóstico fuera de muestra = **{sig:,.1f} u.**
- L = **{L:g}** sem. (σL = {sL:g}) · R = **{R:g}** sem. · demanda esperada en L = **{DL:,.0f}**, en L+R = **{DP:,.0f}** u.

**Stock de seguridad (protege L + R)**
SS = z·√((L+R)·σ² + d²·σL²) = {z:.3f}·√({L+R:g}·{sig:,.1f}² + {dP:,.1f}²·{sL:g}²) = **{float(iv.safety_stock):,.0f} u.**

**Stock objetivo**
S = D(L+R) + SS = {DP:,.0f} + {float(iv.safety_stock):,.0f} = **{int(iv.target_stock):,} u.**

**Punto de reorden (protege solo L)**
ROP = D(L) + z·√(L·σ² + d_L²·σL²) = {DL:,.0f} + {float(iv.safety_stock_lead):,.0f} = **{int(iv.reorder_point):,} u.**

**Pedido sugerido** = máx(0, S − posición) = máx(0, {int(iv.target_stock):,} − {float(iv.inventory_position):,.0f}) = **{int(iv.suggested_order):,} u.**

**Verificación por simulación:** con este S se cubre el **{float(iv.simulated_cycle_service):.1%}** de los ciclos simulados.
""")
