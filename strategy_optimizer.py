"""
strategy_optimizer.py
Optimizador de Estrategia de Retiro basado en la Teoría de Carteras de Markowitz.

Evalúa combinaciones discretas de retiro parcial de posición y selecciona
la estrategia que maximiza la función de utilidad media-varianza del trader:
    U = EV - (gamma / 2) * Var
"""

import itertools
from typing import Any, Dict, List, Tuple

import numpy as np


class SportsTradingPortfolioOptimizer:
    """
    Encuentra la estrategia óptima de scalping progresivo sobre una posición
    de apuestas en vivo, dado un conjunto de probabilidades de supervivencia
    del empate en distintos horizontes temporales y la aversión al riesgo.

    El "universo" de estrategias es el producto cartesiano de porcentajes
    de retiro discretos (0%, 25%, 50%, 75%, 100%) en cada horizonte temporal.
    Para cada combinación se calculan el Valor Esperado, la Varianza y la
    Utilidad de Markowitz.

    Args:
        initial_position (float): Capital total apostado (USD).
    """

    # Granularidad de retiros evaluados (fracción del capital restante)
    WITHDRAWAL_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]

    def __init__(self, initial_position: float):
        if initial_position <= 0:
            raise ValueError("La posición inicial debe ser un valor positivo.")
        self.initial_position = initial_position

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def find_optimal_strategy(
        self,
        survival_probabilities: Dict[int, float],
        risk_aversion: float,
    ) -> Dict[str, Any]:
        """
        Evalúa todas las estrategias del universo discreto y retorna la óptima
        junto con los datos para trazar la Frontera Eficiente.

        Args:
            survival_probabilities (Dict[int, float]): Mapa horizonte → P(empate sobrevive).
            risk_aversion (float): Coeficiente gamma de aversión al riesgo (0–4).

        Returns:
            Dict con claves:
                - "estrategia_optima": datos detallados del punto óptimo.
                - "frontera_eficiente": lista de dicts con ev, std_dev, utility, allocation.
        """
        horizons = sorted(survival_probabilities.keys())
        all_strategies = self._enumerate_strategies(horizons, survival_probabilities)

        frontera: List[Dict[str, Any]] = []
        for strat in all_strategies:
            ev, variance = self._compute_payoff_distribution(
                strat["fractions"], horizons, survival_probabilities
            )
            std_dev = float(np.sqrt(max(variance, 0.0)))
            utility = ev - (risk_aversion / 2.0) * variance
            frontera.append(
                {
                    "ev": round(ev, 4),
                    "std_dev": round(std_dev, 4),
                    "utility": round(utility, 6),
                    "allocation": strat["label"],
                }
            )

        # Estrategia óptima = máxima utilidad
        optima = max(frontera, key=lambda x: x["utility"])

        # Construir vector de retiros legible para la UI
        vector_retiros = self._build_withdrawal_plan(
            optima["allocation"], horizons, survival_probabilities
        )

        return {
            "estrategia_optima": {
                "vector_retiros": vector_retiros,
                "metricas_financieras": {
                    "scalping": {
                        "valor_esperado": optima["ev"],
                        "desviacion_estandar": optima["std_dev"],
                        "utilidad_markowitz": optima["utility"],
                    }
                },
            },
            "frontera_eficiente": frontera,
        }

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _enumerate_strategies(
        self,
        horizons: List[int],
        survival_probs: Dict[int, float],
    ) -> List[Dict[str, Any]]:
        """
        Genera todas las combinaciones posibles de retiros.
        Cada estrategia es un vector de fracciones de retiro, una por horizonte.
        """
        fraction_grid = [self.WITHDRAWAL_FRACTIONS] * len(horizons)
        strategies = []
        for combo in itertools.product(*fraction_grid):
            label = ", ".join(
                [f"+{h}m:{int(f * 100)}%" for h, f in zip(horizons, combo)]
            )
            strategies.append({"fractions": list(combo), "label": label})
        return strategies

    def _compute_payoff_distribution(
        self,
        fractions: List[float],
        horizons: List[int],
        survival_probs: Dict[int, float],
    ) -> Tuple[float, float]:
        """
        Calcula el Valor Esperado y la Varianza del payoff total de una estrategia.

        Lógica:
        - En cada horizonte t_i el trader puede retirar una fracción f_i de la
          posición restante, *condicionado a que el empate haya sobrevivido*.
        - Si en algún punto el empate cae, se pierde todo lo que queda sin retirar.
        - El "retiro" en un horizonte se modela como un ingreso cierto
          (ya salió del mercado) ponderado por la probabilidad de supervivencia.
        """
        remaining = self.initial_position
        ev = 0.0
        variance = 0.0
        prev_survival = 1.0

        for i, (h, f) in enumerate(zip(horizons, fractions)):
            p_survive = survival_probs[h]
            # Probabilidad incremental de que el gol caiga en este tramo
            p_fall_here = prev_survival - p_survive

            # Monto retirado en este horizonte si el empate sigue vivo
            withdrawal = f * remaining

            # Contribución al EV: retiro ponderado por P(sobrevive hasta h)
            ev += withdrawal * p_survive

            # Si el empate cae en este tramo, se pierde el remanente no retirado
            loss_if_fall = remaining - withdrawal
            ev -= loss_if_fall * p_fall_here

            # Varianza simplificada: distribución Bernoulli del desenlace del tramo
            # Aproximación: E[X²] - E[X]²
            payoff_survive = withdrawal
            payoff_fall = -loss_if_fall
            e_x2 = (payoff_survive ** 2) * p_survive + (payoff_fall ** 2) * p_fall_here
            e_x = payoff_survive * p_survive + payoff_fall * p_fall_here
            variance += e_x2 - e_x ** 2

            # Actualizar posición restante y probabilidad acumulada
            remaining -= withdrawal
            prev_survival = p_survive

        # Al final del partido: si sobrevive, recuperamos el remanente (apostamos a empate)
        # Si no, ya fue contabilizado
        ev += remaining * prev_survival

        return float(ev), float(variance)

    def _build_withdrawal_plan(
        self,
        allocation_label: str,
        horizons: List[int],
        survival_probs: Dict[int, float],
    ) -> List[Tuple[int, float]]:
        """
        Convierte la etiqueta de la estrategia óptima en un plan de retiros legible.
        Filtra los horizontes con retiro > 0%.
        """
        plan = []
        parts = allocation_label.split(", ")
        for part, h in zip(parts, horizons):
            pct_str = part.split(":")[1].replace("%", "")
            pct = float(pct_str)
            if pct > 0:
                plan.append((h, pct))
        return plan
