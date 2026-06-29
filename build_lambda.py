"""
build_lambda.py
---------------
Descarga los datos históricos de Mundiales desde openfootball/worldcup.json
(sin clonar el repo, usando URLs raw de GitHub) y construye la distribución
empírica real de goles por minuto que alimenta el motor de simulación.

Genera: data/empirical_lambdas.json  ← usado por simulation_engine.py
        data/build_report.txt        ← resumen estadístico del proceso

Uso:
    python build_lambda.py
"""

import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

# ── URLs raw de los Mundiales disponibles en openfootball/worldcup.json ──────
WORLDCUP_URLS = {
    "2018": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json",
    "2022": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json",
    "2014": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json",
    "2010": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2010/worldcup.json",
    "2006": "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2006/worldcup.json",
}


def fetch_json(url: str) -> dict | None:
    """Descarga un JSON desde una URL raw de GitHub."""
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  ⚠ No se pudo descargar {url}: {e}")
        return None


def extract_goal_minutes(data: dict) -> list[int]:
    """
    Extrae todos los minutos de gol de un JSON de openfootball.
    Maneja la estructura: data["rounds"][i]["matches"][j]["goals1"/"goals2"]
    Ignora goles en tiempo extra (minuto > 90 con offset).
    """
    minutes = []
    rounds = data.get("rounds", [])
    # Algunos años usan "matches" directamente en lugar de "rounds"
    if not rounds:
        rounds = [{"matches": data.get("matches", [])}]

    for rnd in rounds:
        for match in rnd.get("matches", []):
            for side in ("goals1", "goals2"):
                for goal in match.get(side, []):
                    minute = goal.get("minute", 0)
                    offset = goal.get("offset", 0)
                    # Solo goles dentro de los 90 minutos reglamentarios
                    # (el offset representa tiempo añadido; minuto 90+offset)
                    if isinstance(minute, int) and 1 <= minute <= 90:
                        # Capear en 90 para que no desborde el vector
                        effective = min(minute + (offset if offset else 0), 90)
                        minutes.append(effective)
    return minutes


def build_lambda_vector(all_minutes: list[int], total_matches: int) -> list[float]:
    """
    Construye el vector lambda[0..89] (una tasa por minuto).
    lambda[t] = (goles marcados en el minuto t+1) / total_matches

    Esto da la tasa media de gol POR PARTIDO POR MINUTO, que es el
    parámetro λ(t) del Proceso de Poisson No-Homogéneo.
    """
    counts = defaultdict(int)
    for m in all_minutes:
        counts[m] += 1

    lambdas = []
    for t in range(1, 91):          # minutos 1..90
        lambdas.append(counts[t] / total_matches)
    return lambdas


def tramo_percentages(lambdas: list[float]) -> dict[str, float]:
    """Calcula el % de goles por tramo de 10 minutos para el reporte."""
    total = sum(lambdas)
    tramos = {}
    for i in range(0, 90, 10):
        label = f"{i+1}-{i+10}"
        tramo_sum = sum(lambdas[i:i+10])
        tramos[label] = round(100 * tramo_sum / total, 2) if total > 0 else 0.0
    return tramos


def main():
    os.makedirs("data", exist_ok=True)

    all_minutes: list[int] = []
    total_matches = 0
    sources_used = []

    print("📥 Descargando datos históricos de Mundiales...")
    for year, url in WORLDCUP_URLS.items():
        print(f"  → Mundial {year}...")
        data = fetch_json(url)
        if data is None:
            continue

        # Contar partidos en este torneo
        match_count = 0
        rounds = data.get("rounds", [])
        if not rounds:
            rounds = [{"matches": data.get("matches", [])}]
        for rnd in rounds:
            match_count += len(rnd.get("matches", []))

        minutes = extract_goal_minutes(data)
        all_minutes.extend(minutes)
        total_matches += match_count
        sources_used.append(f"  Mundial {year}: {match_count} partidos, {len(minutes)} goles")
        print(f"     ✓ {match_count} partidos | {len(minutes)} goles extraídos")

    if not all_minutes:
        print("\n❌ No se pudieron descargar datos. Verifica tu conexión a internet.")
        print("   El archivo data/empirical_lambdas.json NO fue generado.")
        return

    print(f"\n📊 Total: {total_matches} partidos | {len(all_minutes)} goles")

    lambdas = build_lambda_vector(all_minutes, total_matches)
    tramos = tramo_percentages(lambdas)

    avg_goals = sum(lambdas) * 90
    output = {
        "meta": {
            "fuente": "openfootball/worldcup.json (GitHub raw)",
            "años": list(WORLDCUP_URLS.keys()),
            "total_partidos": total_matches,
            "total_goles": len(all_minutes),
            "promedio_goles_partido": round(avg_goals, 4),
            "descripcion": (
                "Vector de 90 lambdas para Proceso de Poisson No-Homogéneo. "
                "lambda[i] = tasa media de gol en el minuto i+1 por partido."
            ),
        },
        "lambdas": lambdas,          # lista de 90 floats, índice 0 = minuto 1
        "tramos_10min_pct": tramos,  # para verificación visual
    }

    out_path = Path("data/empirical_lambdas.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Reporte de texto ──────────────────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "REPORTE DE CALIBRACIÓN — build_lambda.py",
        "=" * 60,
        "",
        "FUENTES UTILIZADAS:",
        *sources_used,
        "",
        f"TOTAL: {total_matches} partidos | {len(all_minutes)} goles",
        f"MEDIA GOLES/PARTIDO (modelo): {avg_goals:.4f}",
        "",
        "DISTRIBUCIÓN POR TRAMOS DE 10 MINUTOS:",
    ]
    for tramo, pct in tramos.items():
        bar = "█" * int(pct / 0.7)
        report_lines.append(f"  {tramo:8s}  {pct:5.1f}%  {bar}")

    report_lines += [
        "",
        f"Archivo generado: {out_path}",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    with open("data/build_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n✅ Lambda vector guardado en {out_path}")


if __name__ == "__main__":
    main()
