import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from io import StringIO

# =========================
# Config (modifiable)
# =========================
EV_MIN_TEAM = 0.03
EV_MIN_PLAYER = 0.04
MAX_ODDS_TEAM = 5.0
MAX_ODDS_PLAYER = 3.5
KELLY_FRACTION_TEAM = 0.5
KELLY_FRACTION_PLAYER = 0.5
STAKE_MIN = 0.0025   # 0.25% de bankroll
STAKE_MAX = 0.02     # 2% de bankroll

COMBO_ODDS_MIN = 3.5
COMBO_ODDS_MAX = 7.0
COMBO_KELLY = 0.25
COMBO_STAKE_MIN = 0.0025
COMBO_STAKE_MAX = 0.005

# Compétitions à afficher en LIVE (codes football-data.org)
LIVE_COMP_IDS = ["PL", "PD", "FL1"]  # Premier League, LaLiga, Ligue 1

# =========================
# Données de démo (fallback)
# =========================
DEMO_CSV = """date,league,home,away,home_strength,away_strength,home_shots_rate,away_shots_rate,home_corners_rate,away_corners_rate,player_home,player_home_shots90,player_away,player_away_shots90,odds_home,odds_draw,odds_away,odds_ou25_over,odds_ou25_under,odds_teamshots_home_over45,odds_teamshots_home_under45,odds_corners_over85,odds_corners_under85,odds_player_home_over15,odds_player_home_under15,odds_player_away_over15,odds_player_away_under15
2025-10-18,ENG,Arsenal,Chelsea,1.3,1.1,5.8,4.6,6.5,5.2,Bukayo Saka,2.9,Nicolas Jackson,2.1,1.95,3.60,3.80,2.05,1.80,1.95,1.85,2.05,1.80,2.10,1.70,2.40,1.60
2025-10-18,ESP,Girona,Sevilla,1.1,1.0,5.1,4.7,5.8,5.4,Artem Dovbyk,2.6,Youssef En-Nesyri,2.3,1.85,3.60,4.00,1.95,1.85,2.00,1.85,1.95,1.85,2.05,1.75,2.05,1.75
2025-10-18,FRA,Lyon,Monaco,1.15,1.2,5.3,5.1,5.9,6.1,Alexandre Lacazette,3.2,Wissam Ben Yedder,2.5,2.50,3.50,2.90,2.10,1.75,1.90,1.90,2.05,1.80,1.95,1.75,2.15,1.70
"""

def load_demo_today():
    df = pd.read_csv(StringIO(DEMO_CSV), parse_dates=['date'])
    # On remplace la colonne 'date' par la date du jour
    df['date'] = pd.to_datetime("today").normalize().date()
    return df[df['date'] == dt.date.today()].reset_index(drop=True)


# =========================
# LIVE fixtures du jour
# =========================
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key)

def load_live_fixtures():
    """Retourne (df, message). df=None si indisponible."""
    api_key = get_secret("FOOTBALL_DATA_API_KEY")
    if not api_key:
        return None, "Pas de clé FOOTBALL_DATA_API_KEY — mode démo utilisé."

    today = dt.date.today().isoformat()
    url = "https://api.football-data.org/v4/matches"
    params = {"dateFrom": today, "dateTo": today, "competitions": ",".join(LIVE_COMP_IDS)}
    headers = {"X-Auth-Token": api_key}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        matches = data.get("matches", [])
        if not matches:
            return None, "Aucun match trouvé aujourd'hui pour PL/PD/FL1 — mode démo utilisé."

        rows = []
        for m in matches:
            comp = m.get("competition", {}).get("code", "") or "LIVE"
            home = m.get("homeTeam", {}).get("name", "")
            away = m.get("awayTeam", {}).get("name", "")

            # PROXIES de cotes (football-data.org ne fournit pas d’odds gratuites)
            rows.append(dict(
                date=today, league=comp,
                home=home, away=away,
                home_strength=1.10, away_strength=1.00,
                home_shots_rate=5.2, away_shots_rate=4.9,
                home_corners_rate=6.0, away_corners_rate=5.6,
                player_home=(home.split()[0] if home else "Domicile"),
                player_home_shots90=2.4,
                player_away=(away.split()[0] if away else "Extérieur"),
                player_away_shots90=2.1,
                odds_home=2.30, odds_draw=3.30, odds_away=2.90,
                odds_ou25_over=1.95, odds_ou25_under=1.85,
                odds_teamshots_home_over45=1.95, odds_teamshots_home_under45=1.85,
                odds_corners_over85=1.95, odds_corners_under85=1.85,
                odds_player_home_over15=2.05, odds_player_home_under15=1.75,
                odds_player_away_over15=2.05, odds_player_away_under15=1.75,
            ))
        return pd.DataFrame(rows), None
    except Exception as e:
        return None, f"Erreur API fixtures : {e}"

# =========================
# Modèles simples
# =========================
def win_probs_from_strength(home_strength: float, away_strength: float, home_adv: float=0.15):
    diff = (home_strength + home_adv) - away_strength
    p_home = 1/(1+np.exp(-3*diff))
    p_away = 1 - 1/(1+np.exp(-3*(-diff)))
    parity = np.exp(-abs(diff)*2.5)
    p_draw = 0.22*parity
    s = p_home + p_draw + p_away
    return p_home/s, p_draw/s, p_away/s

def poisson_prob_over(lambda_total: float):
    return 1 - sum(np.exp(-lambda_total)*lambda_total**k/np.math.factorial(k) for k in range(0,3))

def poisson_tail(lmbda: float, k_threshold: int):
    k_req = k_threshold + 1
    cdf = 0.0
    for k in range(0, k_req):
        cdf += np.exp(-lmbda) * (lmbda**k)/np.math.factorial(k)
    return 1 - cdf

def ou25_probs(home_strength, away_strength):
    base = 1.35
    gh = base * home_strength * 1.05
    ga = base * away_strength * 0.95
    lam_total = gh + ga
    p_over = poisson_prob_over(lam_total)
    return p_over, 1 - p_over

def team_shots_probs(home_rate, away_rate, side='home', line=4.5, pace=1.0):
    rate = home_rate if side=='home' else away_rate
    lam = rate * pace
    k_threshold = int(line - 0.5)
    p_over = poisson_tail(lam, k_threshold)
    return p_over, 1 - p_over

def corners_probs(home_corners_rate, away_corners_rate, line=8.5, pace=1.0):
    lam = (home_corners_rate + away_corners_rate) * pace
    k_threshold = int(line - 0.5)
    p_over = poisson_tail(lam, k_threshold)
    return p_over, 1 - p_over

def player_shots_probs(shots90, line=1.5, minutes_expected=85, team_multiplier=1.0, role_multiplier=1.0):
    lam = shots90 * (minutes_expected/90.0) * team_multiplier * role_multiplier
    k_threshold = int(line - 0.5)
    p_over = poisson_tail(lam, k_threshold)
    return p_over, 1 - p_over, lam

def ev_from_p_odds(p: float, odds: float) -> float:
    return p*odds - (1 - p)

def confidence_from(p: float, ev: float):
    return int(max(0, min(100, round(p*100*0.6 + max(ev,0)*100*0.4))))

def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0: return 0.0
    edge = (odds * p - (1 - p)) / b
    return max(0.0, edge)

def capped_stake_pct(p: float, odds: float, kelly_frac_base: float, min_pct: float, max_pct: float) -> float:
    stake = kelly_fraction(p, odds) * kelly_frac_base
    return max(min_pct, min(max_pct, stake))

# =========================
# UI
# =========================
st.set_page_config(page_title="Football Edge (MVP)", page_icon="⚽", layout="centered")
st.title("⚽ Football Edge — MVP")

# Affiche si la clé est détectée (utile pour debug)
has_key = False
try:
    has_key = bool(st.secrets.get("FOOTBALL_DATA_API_KEY"))
except Exception:
    pass
st.caption(f"🔐 Clé API détectée : {'oui' if has_key else 'non'}")

mode = st.radio("Source des matchs du jour :", ["Live (API fixtures)", "Démo (exemples)"], index=0)

df = None
note = ""
if mode.startswith("Live"):
    df, note = load_live_fixtures()
    if note:
        st.warning(note)
if df is None or df.empty:
    df = load_demo_today()
    st.caption("Mode démo actif.")

# Liste des matchs
df_show = df[['league','home','away']].copy()
df_show.index = [f"Match #{i+1}" for i in range(len(df_show))]
st.subheader("Matchs du jour")
st.dataframe(df_show, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    idx = st.number_input("Numéro du match (1..{})".format(len(df_show)),
                          min_value=1, max_value=len(df_show), value=1, step=1)
    go_match = st.button("Voir le pari conseillé")
with col2:
    go_combo = st.button("🎟️ Combiné du jour")

# --------- FICHE MATCH ---------
if go_match:
    r = df.iloc[idx-1]
    st.markdown("---")
    st.header(f"🎯 Pari conseillé — {r['league']} • {r['home']} vs {r['away']}")
    options = []

    # 1X2
    p_home, p_draw, p_away = win_probs_from_strength(r.home_strength, r.away_strength)
    for sel, p, odds in [("Home (1)", p_home, float(r.odds_home)),
                         ("Draw (X)", p_draw, float(r.odds_draw)),
                         ("Away (2)", p_away, float(r.odds_away))]:
        ev = ev_from_p_odds(p, odds); conf = confidence_from(p, ev)
        options.append(("1X2", sel, odds, p, ev, conf))

    # O/U 2.5
    p_over, p_under = ou25_probs(r.home_strength, r.away_strength)
    for sel, p, odds in [("Over 2.5", p_over, float(r.odds_ou25_over)),
                         ("Under 2.5", p_under, float(r.odds_ou25_under))]:
        ev = ev_from_p_odds(p, odds); conf = confidence_from(p, ev)
        options.append(("OU2.5", sel, odds, p, ev, conf))

    # Team shots (home over 4.5)
    p_over, _ = team_shots_probs(r.home_shots_rate, r.away_shots_rate, side='home', line=4.5)
    ev = ev_from_p_odds(p_over, float(r.odds_teamshots_home_over45)); conf = confidence_from(p_over, ev)
    options.append(("TeamShots", "Home — Over 4.5 tirs", float(r.odds_teamshots_home_over45), p_over, ev, conf))

    # Corners (O/U 8.5)
    p_over, p_under = corners_probs(r.home_corners_rate, r.away_corners_rate, line=8.5)
    for sel, p, odds in [("Over 8.5 corners", p_over, float(r.odds_corners_over85)),
                         ("Under 8.5 corners", p_under, float(r.odds_corners_under85))]:
        ev = ev_from_p_odds(p, odds); conf = confidence_from(p, ev)
        options.append(("Corners", sel, odds, p, ev, conf))

    # Player shots (home over 1.5)
    p_over, _, _ = player_shots_probs(r.player_home_shots90, line=1.5, minutes_expected=85)
    ev = ev_from_p_odds(p_over, float(r.odds_player_home_over15)); conf = confidence_from(p_over, ev)
    options.append(("PlayerShots", f"{r.player_home} — Over 1.5 tirs", float(r.odds_player_home_over15), p_over, ev, conf))

    options.sort(key=lambda x: (x[4], x[5]), reverse=True)
    best = options[0]
    st.success("Pari conseillé :")
    st.markdown(f"**Type** : {best[0]}")
    st.markdown(f"**Sélection** : **{best[1]}**")
    st.markdown(f"**Cote** : **{best[2]:.2f}**")
    st.markdown(f"**Proba modèle (≈)** : {best[3]*100:.1f}%")
    st.markdown(f"**EV** : {best[4]:.3f}")
    kf = KELLY_FRACTION_PLAYER if best[0]=="PlayerShots" else KELLY_FRACTION_TEAM
    stake = capped_stake_pct(best[3], best[2], kf, STAKE_MIN, STAKE_MAX)
    st.markdown(f"**Mise conseillée** : {stake*100:.2f}% de la bankroll")
    st.markdown(f"**Confiance** : {best[5]}/100")

# --------- COMBINÉ DU JOUR ---------
if go_combo:
    st.markdown("---")
    st.header("🎟️ Combiné du jour")
    singles = []
    for _, r in df.iterrows():
        mid = f"{r.home}-{r.away}"

        p_home, p_draw, p_away = win_probs_from_strength(r.home_strength, r.away_strength)
        for sel, p, odds in [("Home (1)", p_home, float(r.odds_home)),
                             ("Draw (X)", p_draw, float(r.odds_draw)),
                             ("Away (2)", p_away, float(r.odds_away))]:
            ev = ev_from_p_odds(p, odds); conf = confidence_from(p, ev)
            singles.append(("1X2", f"{r.home} vs {r.away} — {sel}", odds, p, ev, conf, mid))

        p_over, p_under = ou25_probs(r.home_strength, r.away_strength)
        for sel, p, odds in [("Over 2.5", p_over, float(r.odds_ou25_over)),
                             ("Under 2.5", p_under, float(r.odds_ou25_under))]:
            ev = ev_from_p_odds(p, odds); conf = confidence_from(p, ev)
            singles.append(("OU2.5", f"{r.home} vs {r.away} — {sel}", odds, p, ev, conf, mid))

        p_over, _ = team_shots_probs(r.home_shots_rate, r.away_shots_rate, side='home', line=4.5)
        ev = ev_from_p_odds(p_over, float(r.odds_teamshots_home_over45)); conf = confidence_from(p_over, ev)
        singles.append(("TeamShots", f"{r.home} — Over 4.5 tirs", float(r.odds_teamshots_home_over45), p_over, ev, conf, mid))

        p_over, p_under = corners_probs(r.home_corners_rate, r.away_corners_rate, line=8.5)
        for sel, p, odds in [("Over 8.5 corners", p_over, float(r.odds_corners_over85)),
                             ("Under 8.5 corners", p_under, float(r.odds_corners_under85))]:
            ev = ev_from_p_odds(p, odds); conf = confidence_from(p, ev)
            singles.append(("Corners", f"{r.home} vs {r.away} — {sel}", odds, p, ev, conf, mid))

        p_over, _, _ = player_shots_probs(r.player_home_shots90, line=1.5, minutes_expected=85)
        ev = ev_from_p_odds(p_over, float(r.odds_player_home_over15)); conf = confidence_from(p_over, ev)
        singles.append(("PlayerShots", f"{r.player_home} — Over 1.5 tirs", float(r.odds_player_home_over15), p_over, ev, conf, mid))

    # filtres
    valid = []
    for leg in singles:
        mkt, sel, odds, p, ev, conf, mid = leg
        if mkt == "PlayerShots":
            if ev < EV_MIN_PLAYER or odds > MAX_ODDS_PLAYER: 
                continue
        else:
            if ev < EV_MIN_TEAM or odds > MAX_ODDS_TEAM:
                continue
        valid.append(leg)

    valid.sort(key=lambda x: (x[4], x[5]), reverse=True)

    combo, combo_odds, combo_p = [], 1.0, 1.0
    used = set()
    for mkt, sel, odds, p, ev, conf, mid in valid:
        if mid in used: 
            continue
        potential = combo_odds * odds
        if potential > COMBO_ODDS_MAX: 
            continue
        combo.append((mkt, sel, odds, p, ev, conf, mid))
        used.add(mid)
        combo_odds *= odds
        combo_p *= p
        if combo_odds >= COMBO_ODDS_MIN:
            break

    if not combo:
        st.info("Aucune combinaison propre trouvée. On propose le meilleur pari simple.")
        if valid:
            mkt, sel, odds, p, ev, conf, mid = valid[0]
            combo, combo_odds, combo_p = [valid[0]], odds, p
        else:
            st.warning("Aucun pari disponible."); st.stop()

    for i, (mkt, sel, odds, p, ev, conf, mid) in enumerate(combo, 1):
        st.markdown(f"**Leg {i}** — {mkt}: {sel} • Cote **{odds:.2f}** • Confiance **{conf}/100**")

    st.markdown(f"**Cote totale** : **{combo_odds:.2f}**")
    ev_combo = combo_p * combo_odds - (1 - combo_p)
    st.markdown(f"**EV combiné** : {ev_combo:.3f} • **Proba de gain** ≈ {combo_p*100:.1f}%")

    b = combo_odds - 1.0
    edge = (combo_odds * combo_p - (1 - combo_p)) / b if b>0 else 0.0
    stake = max(COMBO_STAKE_MIN, min(COMBO_STAKE_MAX, edge * COMBO_KELLY))
    st.markdown(f"**Mise conseillée** : {stake*100:.2f}% de la bankroll")
