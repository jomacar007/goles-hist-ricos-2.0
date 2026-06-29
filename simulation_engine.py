"""
simulation_engine.py
Motor de Simulación Monte Carlo con Proceso de Poisson No-Homogéneo.

Carga el vector lambda empírico generado por build_lambda.py.
Si el archivo no existe, usa la distribución de respaldo basada en
los porcentajes publicados por StatsUltra (PL 2023-25, 10-min intervals).
"""

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


# ── Distribución de respaldo (StatsUltra, PL 2023-25, 10-min intervals) ──────
_FALLBACK_PCT = [7.5, 10.2, 9.5, 9.4, 12.2, 10.7, 11.1, 10.7, 18.8]
_AVG_GOALS    = 2.93

def _build_fallback_lambdas() -> np.ndarray:
    lam = np.zeros(90)
    for i, pct in enumerate(_FALLBACK_PCT):
        lam[i*10 : i*10+10] = (_AVG_GOALS * pct / 100.0) / 10.0
    return lam


# ── Multiplicadores empíricos por estado del marcador ─────────────────────────
def _score_multiplier(home: int, away: int) -> float:
    diff  = abs(home - away)
    total = home + away
    if diff == 0 and total == 0:
        return 0.82   # 0-0: partido cerrado
    if diff == 0:
        return 1.15   # 1-1, 2-2…: abierto
    if diff == 1:
        return 1.20   # perdedor presiona
    return 0.95       # 2+ goles: partido controlado


class MonteCarloSimulationEngine:
    """
    Simula trayectorias de partido mediante NHPP con ajuste por marcador.
    Funciona para cualquier marcador: empate o no empate.
    """

    def __init__(
        self,
        base_lambdas: np.ndarray | None = None,
        num_simulations: int = 10_000,
        data_path: str = "data/empirical_lambdas.json",
    ):
        self.num_simulations = num_simulations

        if base_lambdas is not None:
            self.base_lambdas = np.asarray(base_lambdas, dtype=float)
            self.source = "externo"
        else:
            json_path = Path(data_path)
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    payload = json.load(f)
                self.base_lambdas = np.array(payload["lambdas"], dtype=float)
                meta = payload.get("meta", {})
                self.source = (
                    f"empírico ({meta.get('total_partidos','?')} partidos, "
                    f"media {meta.get('promedio_goles_partido','?')} goles/partido)"
                )
            else:
                self.base_lambdas = _build_fallback_lambdas()
                self.source = "respaldo estadístico (StatsUltra PL 2023-25)"

        if len(self.base_lambdas) != 90:
            raise ValueError("base_lambdas debe tener exactamente 90 elementos.")

    # ── Núcleo: genera matriz de goles por equipo ──────────────────────────────

    def _simulate_goals_matrix(
        self,
        current_minute: int,
        multiplier: float,
    ):
        """
        Devuelve dos matrices (home_goals, away_goals) de shape
        (num_simulations, remaining_minutes). Cada celda = goles Poisson(λ/2)
        para ese equipo en ese minuto.

        Asumimos que cada gol que ocurre tiene 50/50 de ser del local o visitante
        (el NHPP modela la tasa total; la asignación es aleatoria uniforme).
        """
        remaining = max(90 - current_minute, 0)
        if remaining == 0:
            empty = np.zeros((self.num_simulations, 0), dtype=int)
            return empty, empty

        lambdas = self.base_lambdas[current_minute:90] * multiplier
        rng = np.random.default_rng()

        # Goles totales por minuto por simulación
        total_goals = rng.poisson(lam=lambdas, size=(self.num_simulations, remaining))

        # Asignar aleatoriamente a local o visitante (Binomial)
        home_goals = rng.binomial(n=total_goals, p=0.5)
        away_goals = total_goals - home_goals

        return home_goals, away_goals

    # ── API: modo EMPATE ───────────────────────────────────────────────────────

    def simulate_draw_survival(
        self,
        current_minute: int,
        home_goals: int,
        away_goals: int,
        time_horizons: List[int],
    ) -> Dict[int, float]:
        """
        P(empate sobrevive hasta +h minutos) para cada horizonte.
        Solo tiene sentido cuando home_goals == away_goals.
        """
        remaining = max(90 - current_minute, 0)
        if remaining == 0:
            return {h: 1.0 for h in time_horizons}

        multiplier = _score_multiplier(home_goals, away_goals)
        home_m, away_m = self._simulate_goals_matrix(current_minute, multiplier)

        # Acumular goles totales (cualquier gol rompe el empate)
        cumul_total = np.cumsum(home_m + away_m, axis=1)

        survival: Dict[int, float] = {}
        for h in time_horizons:
            actual_h = min(h, remaining)
            survived = np.sum(cumul_total[:, actual_h - 1] == 0)
            survival[h] = float(survived) / self.num_simulations
        return survival

    # ── API: modo NO EMPATE ────────────────────────────────────────────────────

    def simulate_non_draw(
        self,
        current_minute: int,
        home_goals: int,
        away_goals: int,
        time_horizons: List[int],
    ) -> Dict[str, Dict[int, float]]:
        """
        Para un marcador no empatado (ej. 1-0), calcula para cada horizonte:
          - P(marcador no cambia hasta FT)
          - P(el perdedor empata antes de FT)

        Returns:
            {
              "sin_cambio":   {h: prob},   # marcador igual al final del horizonte
              "empate_logrado": {h: prob}, # perdedor remonta al empate
            }
        """
        if home_goals == away_goals:
            raise ValueError("Usa simulate_draw_survival para marcadores empatados.")

        remaining = max(90 - current_minute, 0)
        if remaining == 0:
            return {
                "sin_cambio":    {h: 1.0 for h in time_horizons},
                "empate_logrado":{h: 0.0 for h in time_horizons},
            }

        # Quién va ganando y por cuánto
        diff = home_goals - away_goals   # positivo = local gana

        multiplier = _score_multiplier(home_goals, away_goals)
        home_m, away_m = self._simulate_goals_matrix(current_minute, multiplier)

        # Diferencia acumulada a lo largo del tiempo (desde el marcador actual)
        home_cumul = np.cumsum(home_m, axis=1)
        away_cumul = np.cumsum(away_m, axis=1)

        # Diferencia TOTAL (incluyendo goles ya marcados)
        diff_matrix = diff + home_cumul - away_cumul   # shape: (sims, remaining)

        sin_cambio:     Dict[int, float] = {}
        empate_logrado: Dict[int, float] = {}

        for h in time_horizons:
            actual_h = min(h, remaining)
            diff_at_h = diff_matrix[:, actual_h - 1]

            # Sin cambio: la diferencia al final del horizonte es la misma que ahora
            n_sin_cambio = np.sum(diff_at_h == diff)
            sin_cambio[h] = float(n_sin_cambio) / self.num_simulations

            # Empate logrado: diferencia llega a 0 en algún momento dentro del horizonte
            # (usamos el mínimo de |diff| en el trayecto)
            abs_diff_traj = np.abs(diff_matrix[:, :actual_h])
            n_empate = np.sum(np.min(abs_diff_traj, axis=1) == 0)
            empate_logrado[h] = float(n_empate) / self.num_simulations

        return {
            "sin_cambio":    sin_cambio,
            "empate_logrado": empate_logrado,
        }

    # ── API: tiempo esperado hasta próximo gol ─────────────────────────────────

    def expected_minutes_to_next_goal(
        self,
        current_minute: int,
        home_goals: int,
        away_goals: int,
    ) -> float:
        """
        Tiempo esperado (minutos) hasta el próximo gol (cualquier equipo).
        Fórmula analítica NHPP: E[T] = Σ exp(-Λ_k)
        """
        if current_minute >= 90:
            return 0.0

        multiplier = _score_multiplier(home_goals, away_goals)
        lambdas = self.base_lambdas[current_minute:90] * multiplier

        expected = 0.0
        cum_lam  = 0.0
        for lam in lambdas:
            expected += math.exp(-cum_lam)
            cum_lam  += lam
        return round(expected, 2)
