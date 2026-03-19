import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Forecasting Ventas | Noviembre 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Tarjetas de métricas */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea55;
        border-radius: 12px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #555 !important;
    }
    /* Botón principal */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.65rem 1rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: opacity 0.2s;
    }
    div.stButton > button:first-child:hover {
        opacity: 0.88 !important;
    }
    /* Separadores de sección */
    hr { border-color: #667eea33 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, "models", "modelo_final.joblib")
INF_PATH    = os.path.join(BASE_DIR, "data", "processed", "inferencia_df_transformado.csv")
TRAIN_PATH  = os.path.join(BASE_DIR, "data", "processed", "df.csv")

DIA_ES = {0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
          4: "Viernes", 5: "Sábado", 6: "Domingo"}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (CACHEADOS)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def cargar_modelo():
    return joblib.load(MODELO_PATH)


@st.cache_data(show_spinner=False)
def cargar_inferencia():
    df = pd.read_csv(INF_PATH, encoding="utf-8-sig")
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df.sort_values(["nombre", "fecha"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def bootstrap_lags():
    """
    Los lags del CSV de inferencia son NaN (no hay datos de oct-2025).
    Bootstrap con los últimos 7 días disponibles del dataset de entrenamiento.
    Retorna dict {nombre_producto: [lag1, lag2, ..., lag7]}
    donde lag1 = más reciente, lag7 = más antiguo.
    """
    df_tr = pd.read_csv(TRAIN_PATH)
    df_tr["fecha"] = pd.to_datetime(df_tr["fecha"])
    lags_dict = {}
    for nombre, g in df_tr.groupby("nombre"):
        vals = g.sort_values("fecha").tail(7)["unidades_vendidas"].values.astype(float)
        if len(vals) < 7:
            media = float(vals.mean()) if len(vals) > 0 else 10.0
            vals = np.concatenate([np.full(7 - len(vals), media), vals])
        # vals ordenado de más antiguo a más reciente → invertir para [lag1, ..., lag7]
        lags_dict[nombre] = list(vals[::-1].tolist())
    return lags_dict


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN: PREDICCIÓN RECURSIVA DÍA A DÍA
# ─────────────────────────────────────────────────────────────────────────────
def predecir_recursivo(df_prod, ajuste_desc_pct, ajuste_comp_pct,
                       modelo, features, lags_init):
    """
    Predicciones recursivas para los 30 días de noviembre 2025.

    Parámetros
    ----------
    df_prod        : DataFrame filtrado para UN único producto
    ajuste_desc_pct: Ajuste % sobre precio_venta (+50 = 50% más barato, -50 = 50% más caro)
    ajuste_comp_pct: Ajuste % sobre precio_competencia (+5 = sube 5%)
    modelo         : Modelo cargado con joblib
    features       : Lista de feature names en el orden exacto del modelo
    lags_init      : Dict {nombre: [lag1, ..., lag7]} con valores de bootstrap
    """
    df = df_prod.copy().sort_values("fecha").reset_index(drop=True)
    nombre = str(df["nombre"].iloc[0])

    # ── Ajustar precio de venta ───────────────────────────────────────────────
    df["precio_venta"] = (df["precio_venta"] * (1.0 - ajuste_desc_pct / 100.0)).clip(lower=0.01)

    # ── Ajustar precio de competencia ─────────────────────────────────────────
    df["precio_competencia"] = (
        df["precio_competencia"] * (1.0 + ajuste_comp_pct / 100.0)
    ).clip(lower=0.01)

    # ── Recalcular variables derivadas del precio ─────────────────────────────
    df["descuento_porcentaje"] = (df["precio_venta"] / df["precio_base"] - 1.0) * 100.0
    df["ratio_precio"] = df["precio_venta"] / df["precio_competencia"]

    # ── Inicializar buffer de lags ────────────────────────────────────────────
    # Si los lags del CSV son NaN usamos bootstrap; si no, los usamos tal cual.
    lag1_val = df.iloc[0]["unidades_vendidas_lag1"]
    if pd.isna(lag1_val):
        lag_buf = list(lags_init.get(nombre, [10.0] * 7))
    else:
        lag_buf = []
        for j in range(1, 8):
            v = df.iloc[0][f"unidades_vendidas_lag{j}"]
            lag_buf.append(float(v) if not pd.isna(v) else 10.0)

    # ── Bucle de predicción recursiva ─────────────────────────────────────────
    predicciones = []

    for i in range(len(df)):
        fila = df.iloc[i].copy()

        # Inyectar lags actualizados en la fila
        for j in range(1, 8):
            fila[f"unidades_vendidas_lag{j}"] = lag_buf[j - 1]
        fila["unidades_vendidas_mm7"] = float(np.mean(lag_buf))

        # Construir DataFrame de una fila con las features exactas del modelo
        X = pd.DataFrame([{col: fila.get(col, 0) for col in features}])
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        pred = max(0.0, float(modelo.predict(X)[0]))
        predicciones.append(pred)

        # Desplazar buffer: lag1 ← pred, lag2 ← lag1_anterior, ..., lag7 ← lag6_anterior
        lag_buf = [pred] + lag_buf[:6]

    df["unidades_predichas"] = predicciones
    df["ingresos_predichos"] = df["unidades_predichas"] * df["precio_venta"]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES DE FORMATO
# ─────────────────────────────────────────────────────────────────────────────
def fmt_eur(v):
    return f"€{v:,.2f}"

def fmt_int(v):
    return f"{int(round(v)):,}"

def fmt_pct(v):
    return f"{v:+.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO SEABORN
# ─────────────────────────────────────────────────────────────────────────────
def hacer_grafico(df_pred):
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots(figsize=(14, 4.8))
        fig.patch.set_facecolor("white")

        # Bandas suaves de fin de semana
        for _, r in df_pred.iterrows():
            if bool(r["es_fin_de_semana"]):
                ax.axvspan(
                    float(r["dia_mes"]) - 0.42,
                    float(r["dia_mes"]) + 0.42,
                    alpha=0.10, color="#a29bfe", zorder=0,
                )

        # Línea principal de predicción
        sns.lineplot(
            data=df_pred,
            x="dia_mes",
            y="unidades_predichas",
            color="#667eea",
            linewidth=2.5,
            marker="o",
            markersize=5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            ax=ax,
        )

        # Marcar Black Friday (día 28)
        bf = df_pred[df_pred["dia_mes"] == 28]
        if not bf.empty:
            bf_y = float(bf["unidades_predichas"].iloc[0])
            ax.axvline(28, color="#e74c3c", linestyle="--", linewidth=2.0,
                       alpha=0.85, zorder=3)
            ax.scatter([28], [bf_y], color="#e74c3c", s=200, zorder=5,
                       edgecolors="white", linewidths=1.5)
            y_min = float(df_pred["unidades_predichas"].min())
            y_max = float(df_pred["unidades_predichas"].max())
            y_rng = max(y_max - y_min, 2.0)
            offset_y = y_rng * 0.18
            # Ajustar posición de la anotación para que no salga del gráfico
            txt_x = 23 if bf_y > (y_min + y_rng * 0.7) else 23
            txt_y = bf_y + offset_y if bf_y + offset_y < y_max + y_rng * 0.15 else bf_y - offset_y
            ax.annotate(
                "Black Friday",
                xy=(28, bf_y),
                xytext=(txt_x, txt_y),
                fontsize=9.5,
                fontweight="bold",
                color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.5),
            )

        ax.set_xlabel("Día de Noviembre", fontsize=11, color="#444")
        ax.set_ylabel("Unidades Predichas", fontsize=11, color="#444")
        ax.set_xticks(range(1, 31))
        ax.set_xlim(0.5, 30.5)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="y", alpha=0.35)
        ax.grid(axis="x", alpha=0.10)
        sns.despine(left=False, bottom=False)
        plt.tight_layout()

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TABLA DETALLADA
# ─────────────────────────────────────────────────────────────────────────────
def hacer_tabla(df_pred):
    t = df_pred[[
        "fecha", "dia_semana", "precio_venta", "precio_competencia",
        "descuento_porcentaje", "unidades_predichas",
        "ingresos_predichos", "es_Black_Friday",
    ]].copy()

    t["Fecha"]          = t["fecha"].dt.strftime("%d/%m/%Y")
    t["Día"]            = t["dia_semana"].map(DIA_ES)
    t["Precio Venta"]   = t["precio_venta"].map(fmt_eur)
    t["Compet."]        = t["precio_competencia"].map(fmt_eur)
    t["Descuento"]      = t["descuento_porcentaje"].map(fmt_pct)
    t["Unidades"]       = t["unidades_predichas"].map(fmt_int)
    t["Ingresos"]       = t["ingresos_predichos"].map(fmt_eur)
    t["Evento"]         = t["es_Black_Friday"].map(
        lambda x: "🖤 Black Friday" if x else ""
    )

    return t[["Fecha", "Día", "Precio Venta", "Compet.",
              "Descuento", "Unidades", "Ingresos", "Evento"]]


# ═════════════════════════════════════════════════════════════════════════════
# INICIALIZACIÓN DE RECURSOS
# ═════════════════════════════════════════════════════════════════════════════
try:
    modelo        = cargar_modelo()
    df_inf        = cargar_inferencia()
    lags_boot     = bootstrap_lags()
    FEATURE_NAMES = list(modelo.feature_names_in_)
except Exception as exc:
    st.error(f"❌ Error al inicializar la app: {exc}")
    st.info("Verifica que existen `models/modelo_final.joblib` y los CSV en `data/processed/`.")
    st.stop()

productos_lista = sorted(df_inf["nombre"].unique().tolist())


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎮 Controles de Simulación")
    st.divider()

    producto_sel = st.selectbox(
        "🛍️ Producto",
        productos_lista,
        help="Selecciona el producto para simular sus ventas en noviembre 2025",
    )

    st.divider()

    ajuste_desc = st.slider(
        "💸 Ajuste de descuento",
        min_value=-50, max_value=50, value=0, step=5, format="%d%%",
        help="(+) aplica descuento → precio baja  |  (−) sube el precio de venta",
    )

    st.divider()

    escenario_sel = st.radio(
        "🎯 Escenario de competencia",
        options=["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        help="Variación del precio de la competencia respecto a los datos iniciales",
    )

    MAPA_ESC = {"Actual (0%)": 0, "Competencia -5%": -5, "Competencia +5%": 5}
    ajuste_comp = MAPA_ESC[escenario_sel]

    st.divider()

    btn_simular = st.button("🚀 Simular Ventas", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "📅 **Periodo:** Noviembre 2025  \n"
        "🤖 **Modelo:** HistGradientBoosting  \n"
        "🔄 **Método:** Predicciones recursivas"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PANTALLA DE BIENVENIDA (sin resultados previos)
# ═════════════════════════════════════════════════════════════════════════════
if "resultado" not in st.session_state and not btn_simular:
    st.markdown("# 📈 Forecasting de Ventas")
    st.markdown("### Dashboard de simulación · Noviembre 2025")
    st.divider()

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.info(
            "👋 **Bienvenido al simulador de ventas**\n\n"
            "Configura los parámetros en el panel izquierdo y pulsa **🚀 Simular Ventas**:\n\n"
            "- 📦 Predicciones día a día durante noviembre 2025\n"
            "- 💶 Ingresos proyectados totales del mes\n"
            "- 🖤 Impacto del Black Friday (28 nov)\n"
            "- 🔀 Comparativa de 3 escenarios de competencia"
        )
    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE SIMULACIÓN (cuando se pulsa el botón)
# ═════════════════════════════════════════════════════════════════════════════
if btn_simular:
    df_prod = df_inf[df_inf["nombre"] == producto_sel].copy()

    with st.spinner("⚙️ Ejecutando predicciones recursivas día a día..."):
        resultado = predecir_recursivo(
            df_prod, ajuste_desc, ajuste_comp,
            modelo, FEATURE_NAMES, lags_boot,
        )

    with st.spinner("🔀 Calculando comparativa de escenarios..."):
        escenarios = {
            nombre_e: predecir_recursivo(
                df_prod, ajuste_desc, aj, modelo, FEATURE_NAMES, lags_boot
            )
            for nombre_e, aj in [
                ("Actual (0%)", 0),
                ("Competencia -5%", -5),
                ("Competencia +5%", 5),
            ]
        }

    st.session_state["resultado"]   = resultado
    st.session_state["escenarios"]  = escenarios
    st.session_state["producto"]    = producto_sel
    st.session_state["ajuste_desc"] = ajuste_desc
    st.session_state["escenario"]   = escenario_sel


# ═════════════════════════════════════════════════════════════════════════════
# RECUPERAR RESULTADOS DESDE SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════
res       = st.session_state["resultado"]
escenarios = st.session_state["escenarios"]
prod_name = st.session_state["producto"]
desc_val  = st.session_state["ajuste_desc"]
esc_val   = st.session_state["escenario"]


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 📊 Dashboard de Predicciones de Ventas — Noviembre 2025")
st.markdown(f"### 🛍️ {prod_name}")

c1, c2, c3 = st.columns(3)
c1.caption(f"💸 Ajuste precio: **{desc_val:+d}%**")
c2.caption(f"🎯 Competencia: **{esc_val}**")
c3.caption("📅 1 – 30 noviembre 2025  ·  🖤 Black Friday: día 28")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📌 Métricas Clave del Mes")

k1, k2, k3, k4 = st.columns(4)

total_unidades  = float(res["unidades_predichas"].sum())
total_ingresos  = float(res["ingresos_predichos"].sum())
precio_prom     = float(res["precio_venta"].mean())
descuento_prom  = float(res["descuento_porcentaje"].mean())

k1.metric("📦 Unidades Proyectadas",  fmt_int(total_unidades))
k2.metric("💶 Ingresos Proyectados",  fmt_eur(total_ingresos))
k3.metric("🏷️ Precio Promedio Venta", fmt_eur(precio_prom))
k4.metric("📉 Descuento Promedio",    fmt_pct(descuento_prom))

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE PREDICCIÓN DIARIA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📈 Predicción Diaria de Unidades Vendidas")
st.caption("🟣 Banda suave = fin de semana  ·  🔴 Línea punteada = Black Friday (28 nov)")

fig = hacer_grafico(res)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# TABLA DETALLADA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📋 Detalle Día a Día — Noviembre 2025")

tabla = hacer_tabla(res)
st.dataframe(tabla, use_container_width=True, hide_index=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# COMPARATIVA DE ESCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🔀 Comparativa de Escenarios de Competencia")
st.caption(
    f"Ajuste de precio fijo: **{desc_val:+d}%**  ·  "
    "Solo varía el precio de la competencia en cada escenario"
)

# Valores de referencia del escenario "Actual (0%)" para calcular deltas
u_base = float(escenarios["Actual (0%)"]["unidades_predichas"].sum())
i_base = float(escenarios["Actual (0%)"]["ingresos_predichos"].sum())

ce1, ce2, ce3 = st.columns(3)

ESC_CONFIG = [
    (ce1, "📊 Actual (0%)",      "Actual (0%)"),
    (ce2, "📉 Competencia −5%",  "Competencia -5%"),
    (ce3, "📈 Competencia +5%",  "Competencia +5%"),
]

for col_sc, titulo, key in ESC_CONFIG:
    df_e = escenarios[key]
    u_e  = float(df_e["unidades_predichas"].sum())
    i_e  = float(df_e["ingresos_predichos"].sum())
    d_u  = u_e - u_base if key != "Actual (0%)" else None
    d_i  = i_e - i_base if key != "Actual (0%)" else None

    with col_sc:
        st.markdown(f"**{titulo}**")
        st.metric(
            "Unidades",
            fmt_int(u_e),
            delta=f"{d_u:+.0f} uds" if d_u is not None else None,
        )
        st.metric(
            "Ingresos",
            fmt_eur(i_e),
            delta=fmt_eur(d_i) if d_i is not None else None,
        )
