"""
market_comparison.py
Analizador de precios de Polymarket vs. probabilidades del modelo Poisson.

En Polymarket el precio ES la probabilidad (0.35 = 35%).
Los tres precios suman ~1.00 (o 1.01 con el margen mínimo de la plataforma).
No hay conversión de cuota decimal — se trabaja directamente con precios.
"""

from typing import Dict, Any


class MarketOddsAnalyzer:
    """
    Compara los precios de Polymarket con las probabilidades del modelo NHPP.

    Entrada: precios en formato Polymarket (ej: 0.35, 0.22, 0.50)
    Salida:  edge, margen de plataforma, cuota justa del modelo
    """

    def analyze_market(
        self,
        price_home: float,    # precio "gana local"   ej: 0.35
        price_draw: float,    # precio "empate"        ej: 0.22
        price_away: float,    # precio "gana visitante" ej: 0.50
        poisson_draw_prob: float,  # P(empate) del modelo Poisson (+5 min)
    ) -> Dict[str, Any]:
        """
        Analiza el mercado Polymarket 1X2 y lo confronta con el modelo.

        Args:
            price_home:  precio del local en Polymarket  (0–1)
            price_draw:  precio del empate en Polymarket (0–1)
            price_away:  precio del visitante en Polymarket (0–1)
            poisson_draw_prob: probabilidad de empate del simulador NHPP

        Returns:
            Dict con edge, margen, precios justos y precio justo del modelo.
        """
        total = price_home + price_draw + price_away
        # Margen de la plataforma (overround)
        # En Polymarket suele ser 0–2%; en casas tradicionales 5–10%
        margin_pct = (total - 1.0) * 100

        # Precios justos normalizados (sin margen)
        prob_home_fair = price_home / total
        prob_draw_fair = price_draw / total
        prob_away_fair = price_away / total

        # Precio justo del empate según el modelo Poisson
        # (equivale a 1/cuota_decimal en terminología tradicional)
        model_fair_price = poisson_draw_prob  # ya es una probabilidad [0,1]

        # Edge: diferencia entre lo que dice el modelo y lo que implica el mercado
        # Positivo → el modelo ve más probabilidad de empate que el mercado → valor
        edge = poisson_draw_prob - prob_draw_fair

        return {
            "margin_pct": round(margin_pct, 3),
            "precios_brutos": {
                "local":     round(price_home, 4),
                "empate":    round(price_draw, 4),
                "visitante": round(price_away, 4),
                "suma":      round(total, 4),
            },
            "precios_justos": {
                "local":     round(prob_home_fair, 4),
                "empate":    round(prob_draw_fair, 4),
                "visitante": round(prob_away_fair, 4),
            },
            "model_draw_price": round(model_fair_price, 4),
            "edge_bruto":  round(edge, 6),
            "tiene_valor": edge > 0,
        }

    def analyze_non_draw_market(
        self,
        price_home: float,
        price_draw: float,
        price_away: float,
        prob_no_change: float,    # del simulador
        prob_draw_achieved: float, # del simulador
    ) -> Dict[str, Any]:
        """
        Para partidos no empatados: compara probabilidades del modelo
        con los precios del mercado para los dos escenarios clave.
        """
        base = self.analyze_market(price_home, price_draw, price_away, prob_draw_achieved)

        # Edge del escenario "sin cambio" vs precio actual del ganador
        winning_price = max(price_home, price_away)
        winning_fair  = max(base["precios_justos"]["local"], base["precios_justos"]["visitante"])
        edge_no_change = prob_no_change - winning_fair

        return {
            **base,
            "prob_no_change_model":    round(prob_no_change, 4),
            "prob_draw_achieved_model": round(prob_draw_achieved, 4),
            "edge_no_change":  round(edge_no_change, 6),
            "edge_draw":       round(base["edge_bruto"], 6),
        }
