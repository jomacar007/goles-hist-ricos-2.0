# app.py  — Auditor Cuántico de Scalping v2.1
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from simulation_engine import MonteCarloSimulationEngine
from strategy_optimizer import SportsTradingPortfolioOptimizer
from market_comparison import MarketOddsAnalyzer

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auditor de Scalping v2.1",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_engine():
    return MonteCarloSimulationEngine(num_simulations=10_000)

motor    = load_engine()
analizador = MarketOddsAnalyzer()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("📊 Variables de Control")

st.sidebar.subheader("💰 Capital")
posicion_inicial = st.sidebar.number_input(
    "Capital Expuesto Inicial (USD)", min_value=1.0, value=100.0, step=10.0
)

st.sidebar.subheader("⏱ Estado del Partido")
minuto_actual = st.sidebar.slider("Minuto Actual", 0, 89, 70)

col_g1, col_g2 = st.sidebar.columns(2)
goles_local     = col_g1.number_input("Goles Local", min_value=0, value=0, step=1)
goles_visitante = col_g2.number_input("Goles Visitante", min_value=0, value=0, step=1)

# ── Precios Polymarket ────────────────────────────────────────────────────────
st.sidebar.subheader("📈 Precios Polymarket")
st.sidebar.caption(
    "Introduce los tres precios tal como aparecen en Polymarket. "
    "Ejemplo: Local 0.35 · Empate 0.22 · Visitante 0.50. "
    "La suma debe ser ~1.00 (Polymarket puede dar 1.01 por margen mínimo)."
)

col_p1, col_p2, col_p3 = st.sidebar.columns(3)
price_home = col_p1.number_input("Local (A)", min_value=0.01, max_value=0.99,
                                  value=0.35, step=0.01, format="%.2f")
price_draw = col_p2.number_input("Empate (X)", min_value=0.01, max_value=0.99,
                                   value=0.22, step=0.01, format="%.2f")
price_away = col_p3.number_input("Visitante (B)", min_value=0.01, max_value=0.99,
                                   value=0.50, step=0.01, format="%.2f")

suma_precios = price_home + price_draw + price_away
color_suma = "🟢" if abs(suma_precios - 1.0) <= 0.02 else "🔴"
st.sidebar.caption(f"{color_suma} Suma de precios: **{suma_precios:.3f}** (esperado ~1.00–1.02)")

st.sidebar.subheader("⚖️ Aversión al Riesgo")
gamma = st.sidebar.slider("Gamma (γ) — 0=max retorno | 4=min varianza", 0.0, 4.0, 1.5, 0.1)

with st.sidebar.expander("ℹ️ Fuente de datos del modelo"):
    st.caption(f"**Calibración:** {motor.source}")
    st.caption("Para usar datos de Mundiales reales: `python build_lambda.py`")

# ── Cálculos comunes ──────────────────────────────────────────────────────────
horizontes   = [5, 10, 15, 20]
es_empate    = (goles_local == goles_visitante)
minutos_rest = 90 - minuto_actual

with st.spinner("Simulando 10,000 trayectorias..."):
    t_prox_gol = motor.expected_minutes_to_next_goal(
        minuto_actual, goles_local, goles_visitante
    )

# ── TÍTULO ────────────────────────────────────────────────────────────────────
st.title("🛡️ Auditor Técnico de Scalping v2.1")
st.markdown(
    "Proceso de Poisson No-Homogéneo · Precios Polymarket · "
    "Optimización Markowitz · Funciona con cualquier marcador"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# MODO EMPATE
# ══════════════════════════════════════════════════════════════════════════════
if es_empate:
    with st.spinner("Calculando supervivencia del empate..."):
        prob_sup = motor.simulate_draw_survival(
            minuto_actual, goles_local, goles_visitante, horizontes
        )
        opt = SportsTradingPortfolioOptimizer(posicion_inicial).find_optimal_strategy(
            prob_sup, gamma
        )
        estrategia = opt["estrategia_optima"]
        frontera   = opt["frontera_eficiente"]
        ev_opt     = estrategia["metricas_financieras"]["scalping"]["valor_esperado"]
        std_opt    = estrategia["metricas_financieras"]["scalping"]["desviacion_estandar"]

        analisis = analizador.analyze_market(
            price_home, price_draw, price_away, prob_sup[5]
        )
        edge = analisis["edge_bruto"]

    # ── Métricas superiores ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Marcador", f"{goles_local}–{goles_visitante}", f"Min {minuto_actual}")
    c2.metric("EV Óptimo", f"{ev_opt:.2f} USD")
    c3.metric("Desv. Est. σ", f"{std_opt:.2f} USD")
    c4.metric(
        "Edge Polymarket (+5m)",
        f"{edge*100:+.2f}%",
        delta="CON VALOR" if edge > 0 else "SIN VALOR",
        delta_color="normal" if edge > 0 else "inverse",
    )
    c5.metric("⏳ Próx. Gol", f"{t_prox_gol:.1f} min")

    st.markdown("---")
    col_izq, col_der = st.columns(2)

    with col_izq:
        # ── Plan de retiro ────────────────────────────────────────────────────
        st.subheader("📋 Plan de Retiro Óptimo")
        retiros = estrategia["vector_retiros"]
        if not retiros:
            st.info(f"**HOLD TOTAL** — Con γ={gamma:.1f} el modelo no recomienda fraccionar.")
        else:
            df_plan = pd.DataFrame(retiros, columns=["Horizonte (min)", "% Retiro"])
            df_plan["Monto (USD)"] = df_plan["% Retiro"].apply(
                lambda x: f"${(x/100)*posicion_inicial:.2f}"
            )
            st.table(df_plan)

        # ── Probabilidades de supervivencia ───────────────────────────────────
        st.subheader("🎲 Supervivencia del Empate")
        df_prob = pd.DataFrame([
            {
                "Horizonte": f"+{h} min → minuto {minuto_actual+h}",
                "P(empate sobrevive)": f"{p*100:.1f}%",
                "P(gol cae)": f"{(1-p)*100:.1f}%",
            }
            for h, p in prob_sup.items()
        ])
        st.table(df_prob)

        # ── Auditoría de precios Polymarket ───────────────────────────────────
        st.subheader("🔍 Auditoría Polymarket")
        marg = analisis["margin_pct"]
        p_draw_mercado = analisis["precios_justos"]["empate"]
        p_draw_modelo  = analisis["model_draw_price"]

        col_a1, col_a2 = st.columns(2)
        col_a1.metric("Margen plataforma", f"{marg:.2f}%",
                      help="Polymarket típicamente 0–2%")
        col_a2.metric("Precio justo modelo (empate)", f"{p_draw_modelo:.3f}",
                      delta=f"Mercado: {p_draw_mercado:.3f}",
                      delta_color="normal" if edge > 0 else "inverse")

        if edge > 0:
            st.success(
                f"✅ **Edge positivo {edge*100:+.2f}%.** "
                f"El modelo valora el empate en **{p_draw_modelo:.3f}** "
                f"y Polymarket lo vende a **{price_draw:.3f}** (precio de mercado). "
                "Hay valor en mantener la posición (Hold)."
            )
        else:
            st.warning(
                f"⚠️ **Sin valor ({edge*100:+.2f}%).** "
                f"Polymarket implica {p_draw_mercado:.3f} de probabilidad de empate, "
                f"mayor que el {p_draw_modelo:.3f} del modelo. "
                "Considera scalping inmediato."
            )

    with col_der:
        # ── Frontera Eficiente ────────────────────────────────────────────────
        st.subheader("📈 Frontera Eficiente de Markowitz")
        df_f = pd.DataFrame(frontera)
        df_f["Categoría"] = "Alternativas"
        mask = (
            np.isclose(df_f["ev"], ev_opt, atol=1e-2) &
            np.isclose(df_f["std_dev"], std_opt, atol=1e-2)
        )
        df_f.loc[mask, "Categoría"] = "ÓPTIMA"

        fig = px.scatter(
            df_f, x="std_dev", y="ev", color="Categoría",
            hover_data=["allocation", "utility"],
            labels={"std_dev": "Riesgo σ (USD)", "ev": "EV (USD)"},
            color_discrete_map={"Alternativas": "#7F8C8D", "ÓPTIMA": "#E74C3C"},
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Punto rojo = máxima utilidad para γ={gamma}")

# ══════════════════════════════════════════════════════════════════════════════
# MODO NO EMPATE
# ══════════════════════════════════════════════════════════════════════════════
else:
    diff      = abs(goles_local - goles_visitante)
    lider     = "Local" if goles_local > goles_visitante else "Visitante"
    perdedor  = "Visitante" if goles_local > goles_visitante else "Local"
    marcador  = f"{goles_local}–{goles_visitante}"

    with st.spinner("Simulando escenarios del partido..."):
        resultados = motor.simulate_non_draw(
            minuto_actual, goles_local, goles_visitante, horizontes
        )
        analisis = analizador.analyze_non_draw_market(
            price_home, price_draw, price_away,
            prob_no_change    = resultados["sin_cambio"][min(horizontes[-1], minutos_rest)] if minutos_rest > 0 else 1.0,
            prob_draw_achieved= resultados["empate_logrado"][min(horizontes[-1], minutos_rest)] if minutos_rest > 0 else 0.0,
        )

    # ── Métricas superiores ───────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Marcador", marcador, f"Min {minuto_actual}")
    c2.metric("Ganando", lider, f"+{diff} gol(es)")
    c3.metric("Minutos Rest.", f"{minutos_rest} min")
    c4.metric("⏳ Próx. Gol esperado", f"{t_prox_gol:.1f} min")
    c5.metric("Margen Polymarket", f"{analisis['margin_pct']:.2f}%")

    st.markdown("---")

    col_izq, col_der = st.columns(2)

    with col_izq:
        # ── Tabla de probabilidades por horizonte ─────────────────────────────
        st.subheader(f"📊 Probabilidades para cada horizonte — Marcador {marcador}")

        filas = []
        for h in horizontes:
            actual_h = min(h, minutos_rest)
            if actual_h <= 0:
                break
            p_nc = resultados["sin_cambio"][h]
            p_eq = resultados["empate_logrado"][h]
            filas.append({
                "Horizonte":                f"+{h} min → min {minuto_actual+h}",
                f"P(sigue {marcador})":     f"{p_nc*100:.1f}%",
                f"P({perdedor} empata)":    f"{p_eq*100:.1f}%",
                "P(otro resultado)":        f"{max(0, 1 - p_nc - p_eq)*100:.1f}%",
            })

        if filas:
            st.table(pd.DataFrame(filas))
        else:
            st.info("El partido ha llegado al minuto 90 — no hay tiempo restante.")

        # ── Comparación con Polymarket ────────────────────────────────────────
        st.subheader("🔍 Auditoría Polymarket")
        p_draw_mercado = analisis["precios_justos"]["empate"]
        p_draw_modelo  = analisis["prob_draw_achieved_model"]
        p_win_mercado  = analisis["precios_justos"]["local" if goles_local > goles_visitante else "visitante"]
        p_win_modelo   = analisis["prob_no_change_model"]

        df_audit = pd.DataFrame([
            {
                "Escenario": f"Empate ({perdedor} remonta)",
                "Precio Polymarket": f"{price_draw:.3f}",
                "Precio justo mercado": f"{p_draw_mercado:.3f}",
                "Probabilidad modelo": f"{p_draw_modelo:.3f}",
                "Edge": f"{analisis['edge_draw']*100:+.2f}%",
            },
            {
                "Escenario": f"Sin cambio ({lider} mantiene)",
                "Precio Polymarket": f"{max(price_home, price_away):.3f}",
                "Precio justo mercado": f"{p_win_mercado:.3f}",
                "Probabilidad modelo": f"{p_win_modelo:.3f}",
                "Edge": f"{analisis['edge_no_change']*100:+.2f}%",
            },
        ])
        st.table(df_audit)

        # Recomendación
        edge_draw     = analisis["edge_draw"]
        edge_no_change= analisis["edge_no_change"]

        if edge_draw > 0.03:
            st.success(
                f"✅ **El modelo ve valor en el empate** ({p_draw_modelo:.3f} vs {p_draw_mercado:.3f} del mercado). "
                f"Edge: {edge_draw*100:+.2f}%. Polymarket subestima la probabilidad de remontada del {perdedor}."
            )
        elif edge_no_change > 0.03:
            st.success(
                f"✅ **El modelo ve valor en que el {lider} mantenga** ({p_win_modelo:.3f} vs {p_win_mercado:.3f}). "
                f"Edge: {edge_no_change*100:+.2f}%. Polymarket sobreestima la probabilidad de remontada."
            )
        else:
            st.info(
                f"ℹ️ No hay edge significativo en ninguno de los dos escenarios principales. "
                f"Empate: {edge_draw*100:+.2f}% | Sin cambio: {edge_no_change*100:+.2f}%."
            )

    with col_der:
        # ── Gráfico de probabilidades en el tiempo ────────────────────────────
        st.subheader("📈 Evolución de Probabilidades por Horizonte")

        if filas:
            df_plot = pd.DataFrame([
                {"Horizonte (min)": h, "Escenario": f"Sigue {marcador}", "Probabilidad": resultados["sin_cambio"][h]*100}
                for h in horizontes if min(h, minutos_rest) > 0
            ] + [
                {"Horizonte (min)": h, "Escenario": f"{perdedor} empata", "Probabilidad": resultados["empate_logrado"][h]*100}
                for h in horizontes if min(h, minutos_rest) > 0
            ])

            fig = px.line(
                df_plot, x="Horizonte (min)", y="Probabilidad",
                color="Escenario", markers=True,
                labels={"Probabilidad": "Probabilidad (%)"},
                color_discrete_map={
                    f"Sigue {marcador}":  "#2ECC71",
                    f"{perdedor} empata": "#E74C3C",
                },
            )
            fig.update_layout(yaxis_range=[0, 100])
            fig.add_hline(y=50, line_dash="dot", line_color="gray",
                          annotation_text="50%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Verde = probabilidad de que el marcador {marcador} no cambie. "
                f"Rojo = probabilidad de que el {perdedor} llegue al empate."
            )
        else:
            st.info("Sin tiempo restante para graficar.")

        # ── Tiempo esperado al próximo gol ────────────────────────────────────
        st.subheader("⏳ Tiempo Esperado al Próximo Gol")
        if t_prox_gol >= minutos_rest:
            st.success(
                f"El próximo gol se espera en **{t_prox_gol:.1f} minutos**, "
                f"pero solo quedan **{minutos_rest} minutos**. "
                f"Probabilidad alta de que el marcador {marcador} sea el resultado final."
            )
        else:
            st.warning(
                f"El próximo gol se espera en **{t_prox_gol:.1f} minutos**, "
                f"dentro del tiempo restante ({minutos_rest} min). "
                f"El marcador puede cambiar antes del final."
            )
