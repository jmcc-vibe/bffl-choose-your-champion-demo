# choose_your_champion.py
# Streamlit web app to run a league-wide "Choose Your Champion" pick'em game.
# - Single-file app with SQLite storage
# - Admin tools to set weekly matchups and declare winners
# - Members log in with a passcode and make picks from an interactive UI
# - Live leaderboard, weekly results, and pick distribution charts
#
# Quick start:
#   pip install streamlit pandas altair
#   streamlit run choose_your_champion.py
#
# Deployment tips (Render/Streamlit Cloud):
#   - Persist the SQLite DB by mounting a volume (Render) or using a managed DB.
#   - Or export/import CSVs from the Admin tab weekly.

import sqlite3
import os
from datetime import datetime, date, time
from typing import Optional, List

import pandas as pd
import streamlit as st
import altair as alt

# Support both env var and Streamlit secrets for DB path
DB_PATH = os.environ.get("CYC_DB_PATH") or st.secrets.get("CYC_DB_PATH", "choose_your_champion.db")
LEAGUE_NAME = "The BFFL"
APP_TITLE = f"{LEAGUE_NAME} — Choose Your Champion"

# ------------------------------
# Database helpers
# ------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # Managers (league members)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS managers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            passcode TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    # Weekly matchups (what people will pick)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS matchups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week INTEGER NOT NULL,
            game_slug TEXT NOT NULL,
            team_a TEXT NOT NULL,
            team_b TEXT NOT NULL,
            lock_datetime TEXT,              -- optional ISO string (UTC or local)
            winner TEXT                      -- NULL until admin declares winner ("A" or "B")
        );
        """
    )

    # Picks by manager
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week INTEGER NOT NULL,
            manager_id INTEGER NOT NULL,
            matchup_id INTEGER NOT NULL,
            pick TEXT NOT NULL,              -- "A" or "B"
            created_at TEXT NOT NULL,
            UNIQUE(week, manager_id, matchup_id),
            FOREIGN KEY(manager_id) REFERENCES managers(id) ON DELETE CASCADE,
            FOREIGN KEY(matchup_id) REFERENCES matchups(id) ON DELETE CASCADE
        );
        """
    )

    conn.commit()
    return conn


# ------------------------------
# Auth & session
# ------------------------------

def login(name: str, passcode: str) -> Optional[dict]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, passcode, is_admin FROM managers WHERE name = ?", (name.strip(),))
    row = cur.fetchone()
    if row and row[2] == passcode:
        return {"id": row[0], "name": row[1], "is_admin": bool(row[3])}
    return None


def ensure_default_admin():
    # For first run convenience; change/remove in production.
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM managers")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO managers(name, passcode, is_admin) VALUES (?,?,1)",
            ("Commissioner", "commish"),
        )
        conn.commit()


# ------------------------------
# Data operations
# ------------------------------

def upsert_manager(name: str, passcode: str, is_admin: bool = False):
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO managers(name, passcode, is_admin) VALUES (?,?,?)",
            (name.strip(), passcode.strip(), int(is_admin)),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Update passcode/admin if name exists
        cur.execute(
            "UPDATE managers SET passcode=?, is_admin=? WHERE name=?",
            (passcode.strip(), int(is_admin), name.strip()),
        )
        conn.commit()


def add_matchup(week: int, game_slug: str, team_a: str, team_b: str, lock_dt: Optional[datetime]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO matchups(week, game_slug, team_a, team_b, lock_datetime) VALUES (?,?,?,?,?)",
        (int(week), game_slug.strip(), team_a.strip(), team_b.strip(), lock_dt.isoformat() if lock_dt else None),
    )
    conn.commit()


def list_matchups(week: int) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM matchups WHERE week = ? ORDER BY id", conn, params=(int(week),)
    )
    return df


def set_winner(matchup_id: int, winner: Optional[str]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE matchups SET winner = ? WHERE id = ?", (winner, int(matchup_id)))
    conn.commit()


def record_pick(week: int, manager_id: int, matchup_id: int, pick: str) -> bool:
    # Enforce lock if present
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT lock_datetime FROM matchups WHERE id=?", (int(matchup_id),))
    row = cur.fetchone()
    if row and row[0]:
        try:
            lock_dt = datetime.fromisoformat(row[0])
            if datetime.now() > lock_dt:
                return False
        except ValueError:
            pass

    now_iso = datetime.now().isoformat(timespec="seconds")
    try:
        cur.execute(
            "INSERT INTO picks(week, manager_id, matchup_id, pick, created_at) VALUES (?,?,?,?,?)",
            (int(week), int(manager_id), int(matchup_id), pick, now_iso),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Update existing pick (before lock)
        cur.execute(
            "UPDATE picks SET pick=?, created_at=? WHERE week=? AND manager_id=? AND matchup_id=?",
            (pick, now_iso, int(week), int(manager_id), int(matchup_id)),
        )
        conn.commit()
        return True


def ensure_settings_table():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    # default reveal mode: 'always' | 'after_pick' | 'after_lock'
    cur.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", ("reveal_picks", "always"))
    conn.commit()


def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default


def set_setting(key: str, value: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()


def picks_for_matchup(matchup_id: int) -> pd.DataFrame:
    conn = get_conn()
    query = """
        SELECT m.name AS manager, p.pick, p.created_at
        FROM picks p
        JOIN managers m ON m.id = p.manager_id
        WHERE p.matchup_id = ?
        ORDER BY m.name
    """
    return pd.read_sql_query(query, conn, params=(int(matchup_id),))


def can_show_picks(reveal_mode: str, user_has_picked: bool, lock_datetime_str: Optional[str]) -> bool:
    mode = (reveal_mode or "always").lower()
    if mode == "always":
        return True
    if mode == "after_pick":
        return bool(user_has_picked)
    if mode == "after_lock":
        if not lock_datetime_str:
            return True  # no lock set -> treat as open
        try:
            return datetime.now() > datetime.fromisoformat(lock_datetime_str)
        except Exception:
            return False
    return True


def get_user_pick(week: int, manager_id: int, matchup_id: int) -> Optional[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pick FROM picks WHERE week=? AND manager_id=? AND matchup_id=?", (int(week), int(manager_id), int(matchup_id)))
    row = cur.fetchone()
    return row[0] if row else None


def picks_for_week(week: int) -> pd.DataFrame:
    conn = get_conn()
    query = """
        SELECT p.id, p.week, p.manager_id, m.name AS manager, p.matchup_id, mt.game_slug,
               mt.team_a, mt.team_b, p.pick, mt.winner, p.created_at
        FROM picks p
        JOIN managers m ON m.id = p.manager_id
        JOIN matchups mt ON mt.id = p.matchup_id
        WHERE p.week = ?
        ORDER BY m.name, mt.id
    """
    df = pd.read_sql_query(query, conn, params=(int(week),))
    return df


def leaderboard(up_to_week: Optional[int] = None) -> pd.DataFrame:
    conn = get_conn()
    params = []
    where_clause = ""
    if up_to_week is not None:
        where_clause = "WHERE p.week <= ?"
        params.append(int(up_to_week))
    query = f"""
        SELECT m.id AS manager_id, m.name AS manager,
               SUM(CASE WHEN mt.winner IS NOT NULL AND p.pick = mt.winner THEN 1 ELSE 0 END) AS correct,
               SUM(CASE WHEN mt.winner IS NOT NULL THEN 1 ELSE 0 END) AS total_graded,
               COUNT(*) AS total_picked
        FROM picks p
        JOIN managers m ON m.id = p.manager_id
        JOIN matchups mt ON mt.id = p.matchup_id
        {where_clause}
        GROUP BY m.id, m.name
        ORDER BY correct DESC, total_picked DESC, m.name ASC
    """
    df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        df["win_pct"] = (df["correct"] / df["total_graded"]).fillna(0).round(3)
    return df


def pick_distribution(week: int) -> pd.DataFrame:
    df = picks_for_week(week)
    if df.empty:
        return df
    # Count picks per matchup
    summary = (
        df.groupby(["matchup_id", "game_slug", "team_a", "team_b", "winner"])['pick']
          .value_counts()
          .rename("count").reset_index()
    )
    pivot = summary.pivot_table(
        index=["matchup_id", "game_slug", "team_a", "team_b", "winner"],
        columns="pick",
        values="count",
        fill_value=0
    ).reset_index()
    pivot.columns.name = None
    # ensure A/B columns
    for col in ["A", "B"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["A"] + pivot["B"]
    pivot["pct_A"] = (pivot["A"] / pivot["total"]).replace([pd.NA, float('inf')], 0).fillna(0).round(3)
    pivot["pct_B"] = (pivot["B"] / pivot["total"]).replace([pd.NA, float('inf')], 0).fillna(0).round(3)
    return pivot.sort_values("matchup_id")

# ---- Season-wide analytics helpers ----

def picks_all() -> pd.DataFrame:
    conn = get_conn()
    query = """
        SELECT p.id, p.week, p.manager_id, m.name AS manager, p.matchup_id,
               mt.game_slug, mt.team_a, mt.team_b, mt.winner, p.pick, p.created_at
        FROM picks p
        JOIN managers m ON m.id = p.manager_id
        JOIN matchups mt ON mt.id = p.matchup_id
        ORDER BY p.week, m.name, p.matchup_id
    """
    return pd.read_sql_query(query, conn)


def season_overview() -> pd.DataFrame:
    df = picks_all()
    if df.empty:
        return df
    df['graded'] = df['winner'].notna().astype(int)
    df['correct_flag'] = ((df['winner'].notna()) & (df['pick'] == df['winner'])).astype(int)
    agg = df.groupby(['manager']).agg(
        correct=('correct_flag', 'sum'),
        total_graded=('graded', 'sum'),
        total_picked=('id', 'count'),
        weeks_played=('week', pd.Series.nunique),
    ).reset_index()
    if not agg.empty:
        agg['win_pct'] = (agg['correct'] / agg['total_graded']).fillna(0).round(3)
        agg['avg_correct_per_week'] = (agg['correct'] / agg['weeks_played']).fillna(0).round(2)
    return agg.sort_values(['win_pct','correct'], ascending=[False, False])


def weekly_breakdown_matrix() -> pd.DataFrame:
    df = picks_all()
    if df.empty:
        return df
    df['correct_flag'] = ((df['winner'].notna()) & (df['pick'] == df['winner'])).astype(int)
    wk = df.groupby(['manager','week'])['correct_flag'].sum().reset_index()
    pivot = wk.pivot(index='manager', columns='week', values='correct_flag').fillna(0).astype(int)
    pivot = pivot.sort_index(axis=1)  # weeks ascending columns
    pivot.index.name = None
    return pivot


def team_pick_tendencies() -> pd.DataFrame:
    df = picks_all()
    if df.empty:
        return df
    # Map each pick to a chosen_team label
    chosen = []
    for _, r in df.iterrows():
        team = r['team_a'] if r['pick'] == 'A' else r['team_b']
        correct = int(r['winner'] == r['pick']) if pd.notna(r['winner']) else 0
        graded = int(pd.notna(r['winner']))
        chosen.append({'team': team, 'manager': r['manager'], 'week': r['week'], 'correct': correct, 'graded': graded})
    ch = pd.DataFrame(chosen)
    if ch.empty:
        return ch
    league = ch.groupby('team').agg(
        times_picked=('team','count'),
        times_graded=('graded','sum'),
        correct_picks=('correct','sum'),
    ).reset_index()
    league['pick_acc'] = (league['correct_picks'] / league['times_graded']).replace([pd.NA, float('inf')], 0).fillna(0).round(3)
    league = league.sort_values(['times_picked','pick_acc'], ascending=[False, False])
    return league


def manager_team_bias(manager_name: str) -> pd.DataFrame:
    df = picks_all()
    if df.empty:
        return df
    chosen = []
    for _, r in df.iterrows():
        team = r['team_a'] if r['pick'] == 'A' else r['team_b']
        correct = int(r['winner'] == r['pick']) if pd.notna(r['winner']) else 0
        graded = int(pd.notna(r['winner']))
        chosen.append({'team': team, 'manager': r['manager'], 'week': r['week'], 'correct': correct, 'graded': graded})
    ch = pd.DataFrame(chosen)
    ch = ch[ch['manager'] == manager_name]
    if ch.empty:
        return ch
    out = ch.groupby('team').agg(times_picked=('team','count'), times_graded=('graded','sum'), correct_picks=('correct','sum')).reset_index()
    out['pick_acc'] = (out['correct_picks'] / out['times_graded']).replace([pd.NA, float('inf')], 0).fillna(0).round(3)
    return out.sort_values(['times_picked','pick_acc'], ascending=[False, False])


def matchup_consensus(threshold: float = 0.5) -> pd.DataFrame:
    """Per-matchup consensus and upset flag across the season.
    Upset definition: winning side pick-share < threshold (default 0.5). Ties excluded.
    """
    df = picks_all()
    if df.empty:
        return df
    summary = (
        df.groupby(["matchup_id", "week", "game_slug", "team_a", "team_b", "winner"])['pick']
          .value_counts().rename("count").reset_index()
    )
    pivot = summary.pivot_table(
        index=["matchup_id","week","game_slug","team_a","team_b","winner"],
        columns="pick", values="count", fill_value=0
    ).reset_index()
    pivot.columns.name = None
    for col in ["A","B"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot['total'] = pivot['A'] + pivot['B']
    pivot['pct_A'] = (pivot['A']/pivot['total']).replace([pd.NA, float('inf')], 0).fillna(0)
    pivot['pct_B'] = (pivot['B']/pivot['total']).replace([pd.NA, float('inf')], 0).fillna(0)
    pivot['winning_side'] = pivot['winner']
    pivot['winning_pick_share'] = pivot.apply(lambda r: r['pct_A'] if r['winner']=="A" else (r['pct_B'] if r['winner']=="B" else pd.NA), axis=1)
    pivot['consensus_share'] = pivot[['pct_A','pct_B']].max(axis=1)
    pivot['minority_side'] = pivot.apply(lambda r: ('A' if r['pct_A'] < r['pct_B'] else ('B' if r['pct_B'] < r['pct_A'] else None)), axis=1)
    pivot['minority_share'] = pivot[['pct_A','pct_B']].min(axis=1)
    pivot['upset_flag'] = pivot.apply(lambda r: bool(pd.notna(r['winner']) and pd.notna(r['winning_pick_share']) and (r['winning_pick_share'] < threshold) and (r['pct_A'] != r['pct_B'])), axis=1)
    return pivot.sort_values(["week","matchup_id"])


def upset_metrics(threshold: float = 0.5):
    cons = matchup_consensus(threshold)
    if cons.empty:
        return {
            'overall_rate': 0.0,
            'num_upsets': 0,
            'graded_consensus': 0,
            'weekly_rates': pd.DataFrame(),
            'manager_upset_wins': pd.DataFrame(),
            'consensus': cons
        }
    graded = cons[cons['winner'].notna() & (cons['total'] > 0) & (cons['pct_A'] != cons['pct_B'])]
    num_upsets = int(graded['upset_flag'].sum())
    graded_consensus = int(len(graded))
    overall_rate = (num_upsets / graded_consensus) if graded_consensus else 0.0
    weekly_rates = graded.groupby('week')['upset_flag'].mean().reset_index().rename(columns={'upset_flag':'upset_rate'})

    # Manager upset wins: picked minority AND was correct in an upset matchup
    df = picks_all()
    mjoin = cons[['matchup_id','minority_side','upset_flag']]
    df = df.merge(mjoin, on='matchup_id', how='left')
    df['correct_flag'] = ((df['winner'].notna()) & (df['pick'] == df['winner'])).astype(int)
    df['picked_minority'] = (df['pick'] == df['minority_side']).fillna(False)
    df['upset_win'] = (df['picked_minority'] & df['correct_flag'].astype(bool) & df['upset_flag'].fillna(False))
    manager_upset_wins = df.groupby('manager')['upset_win'].sum().reset_index().rename(columns={'upset_win':'upset_wins'}).sort_values('upset_wins', ascending=False)

    return {
        'overall_rate': round(float(overall_rate), 3),
        'num_upsets': num_upsets,
        'graded_consensus': graded_consensus,
        'weekly_rates': weekly_rates,
        'manager_upset_wins': manager_upset_wins,
        'consensus': cons
    }


def weekly_manager_cumulative() -> pd.DataFrame:
    """Cumulative correct picks and win% by manager over weeks."""
    df = picks_all()
    if df.empty:
        return df
    df['graded'] = df['winner'].notna().astype(int)
    df['correct_flag'] = ((df['winner'].notna()) & (df['pick'] == df['winner'])).astype(int)
    wk = df.groupby(['manager','week']).agg(correct=('correct_flag','sum'), graded=('graded','sum')).reset_index()
    wk = wk.sort_values(['manager','week'])
    # cumulative within each manager
    wk['cum_correct'] = wk.groupby('manager')['correct'].cumsum()
    wk['cum_graded'] = wk.groupby('manager')['graded'].cumsum()
    wk['cum_win_pct'] = (wk['cum_correct'] / wk['cum_graded']).replace([pd.NA, float('inf')], 0).fillna(0)
    return wk


# ------------------------------
# Theming & style utilities
# ------------------------------

PRIMARY_DEFAULT = "#5B8CFF"


def _altair_minimal_theme():
    return {
        'config': {
            'background': 'transparent',
            'font': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial',
            'axis': {
                'labelColor': '#a0a8b8',
                'titleColor': '#a0a8b8',
                'gridColor': '#2a2f3a',
                'domainColor': '#2a2f3a'
            },
            'legend': {'labelColor': '#a0a8b8', 'titleColor': '#a0a8b8'},
            'view': {'stroke': 'transparent'}
        }
    }


def set_altair_theme():
    try:
        alt.themes.register('cyc_minimal', _altair_minimal_theme)
        alt.themes.enable('cyc_minimal')
    except Exception:
        pass


def inject_css(theme: str = 'dark', accent: str = PRIMARY_DEFAULT):
    """Injects a sleek, high-contrast theme with glassy cards and modern typography."""
    dark = theme.lower() == 'dark'
    bg = '#0b0f19' if dark else '#0f1421'
    page = '#0f1421' if dark else '#ffffff'
    card = 'rgba(255,255,255,0.05)' if dark else '#f6f8fb'
    text = '#e6e8ec' if dark else '#111827'
    muted = '#a0a8b8' if dark else '#6b7280'

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    :root {{ --accent: {accent}; }}
    .stApp {{
        background: linear-gradient(180deg, {bg} 0%, {page} 60%);
        color: {text};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }}
    /* Buttons */
    .stButton > button {{
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, var(--accent), calc(100% - 1.5rem));
        color: white;
        box-shadow: 0 6px 20px rgba(91,140,255,0.25);
    }}
    .stButton > button:focus {{ outline: none; }}
    /* Inputs */
    .stTextInput > div > div > input, .stNumberInput > div > div > input {{
        border-radius: 12px !important;
    }}
    /* Cards */
    .cyc-card {{
        background: {card};
        backdrop-filter: saturate(160%) blur(8px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }}
    /* Hero */
    .cyc-hero {{
        background: radial-gradient(1200px 400px at 20% -20%, rgba(91,140,255,0.25), transparent),
                    radial-gradient(1200px 400px at 80% -20%, rgba(255,255,255,0.06), transparent);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }}
    .cyc-hero h1 {{ margin: 0; font-size: 1.6rem; font-weight: 700; letter-spacing: 0.2px; }}
    .cyc-hero p {{ margin: 0.25rem 0 0; color: {muted}; }}
    /* Dataframes */
    div[data-testid='stDataFrame'] {{ border-radius: 14px; overflow: hidden; }}
    </style>
    """, unsafe_allow_html=True)


def inject_extra_css():
    # Additional overlay + mobile-first polish and BFFL branding styles
    st.markdown(
        """
        <style>
        /* Subtle football field lines overlay */
        .stApp::before {
          content: "";
          position: fixed; inset: 0; pointer-events: none; z-index: 0;
          background:
            repeating-linear-gradient(
              to bottom,
              rgba(255,255,255,0.03),
              rgba(255,255,255,0.03) 1px,
              transparent 1px,
              transparent 44px
            ),
            radial-gradient(1200px 320px at 10% -10%, rgba(91,140,255,0.10), transparent 60%),
            radial-gradient(1000px 260px at 90% -10%, rgba(255,255,255,0.05), transparent 60%);
          mix-blend-mode: overlay;
        }
        /* Wordmark */
        .bffl-title { margin: 0; font-size: 2.0rem; font-weight: 850; letter-spacing: 0.2px; }
        /* Buttons */
        .stButton > button {
          border-radius: 14px; border: 1px solid rgba(255,255,255,0.08);
          background: linear-gradient(180deg, var(--accent), calc(100% - 1.5rem));
          color: white; box-shadow: 0 8px 26px rgba(91,140,255,0.28);
          transition: transform .08s ease, box-shadow .12s ease;
        }
        .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 12px 30px rgba(91,140,255,0.35); }
        .stButton > button:active { transform: translateY(0px) scale(.99); }
        /* Metrics */
        [data-testid="stMetric"] {
          background: rgba(255,255,255,0.05);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 16px; padding: 12px 14px; backdrop-filter: saturate(160%) blur(8px);
        }
        /* Tables */
        thead tr th { background: rgba(255,255,255,0.03) !important; }
        tbody tr:hover { background: rgba(255,255,255,0.03) !important; }
        /* Selects / radios */
        .stRadio > div { gap: .5rem; }
        /* Sticky tabs for mobile */
        div[role='tablist'] { position: sticky; top: 0; z-index: 5; backdrop-filter: blur(8px); background: rgba(0,0,0,0.25); border-bottom: 1px solid rgba(255,255,255,0.06); }
        /* Mobile-first tweaks */
        @media (max-width: 640px) {
          .block-container { padding: 0.75rem 0.8rem 2rem; max-width: 100%; }
          .cyc-hero { padding: 1rem 1rem; }
          .bffl-title { font-size: 1.5rem; }
          .stButton > button { width: 100%; padding: 0.6rem 0.8rem; }
          div[data-testid='column'] { width: 100% !important; flex: 1 1 100% !important; display: block !important; }
          .cyc-card { padding: 0.85rem 0.95rem; }
        }

def ui_style_panel():
    with st.sidebar:
        with st.expander("Appearance", expanded=False):
            theme = st.radio("Theme", options=["Dark","Light"], horizontal=True,
                             index=0 if st.session_state.get('theme','Dark')=='Dark' else 1)
            palette = st.selectbox("Accent preset", ["Electric Blue","Neon Mint","Sunset Coral","Violet Pulse"],
                                   index={"Electric Blue":0,"Neon Mint":1,"Sunset Coral":2,"Violet Pulse":3}.get(st.session_state.get('palette','Electric Blue'),0))
            accents = {
                "Electric Blue": "#5B8CFF",
                "Neon Mint": "#22D3A6",
                "Sunset Coral": "#FB7185",
                "Violet Pulse": "#8B5CF6",
            }
            accent = accents[palette]
            st.session_state['theme'] = theme
            st.session_state['accent'] = accent
            st.session_state['palette'] = palette
            st.caption("Pro tip: Dark + Electric Blue feels the most 'startup'."):
    with st.sidebar:
        with st.expander("Appearance", expanded=False):
            theme = st.radio("Theme", options=["Dark","Light"], horizontal=True, index=0 if st.session_state.get('theme','Dark')=='Dark' else 1)
            accent = st.color_picker("Accent", value=st.session_state.get('accent', PRIMARY_DEFAULT))
            st.session_state['theme'] = theme
            st.session_state['accent'] = accent

# ------------------------------
# UI Components
# ------------------------------

def ui_header():
    st.set_page_config(page_title=APP_TITLE, page_icon="🏈", layout="wide")
    st.markdown(f"""
    <div class='cyc-hero'>
      <h1 class='bffl-title'>"🏈" {LEAGUE_NAME}</h1>
      <p class='bffl-sub'>Choose Your Champion — League Pick'em</p>
    </div>
    """, unsafe_allow_html=True)


def ui_login():
    with st.sidebar:
        st.subheader("Login")
        name = st.text_input("Manager name")
        passcode = st.text_input("Passcode", type="password")
        if st.button("Sign in", use_container_width=True):
            user = login(name, passcode)
            if user:
                st.session_state["user"] = user
                st.success(f"Welcome, {user['name']}!")
            else:
                st.error("Invalid name/passcode.")
        if st.button("Sign out", use_container_width=True):
            st.session_state.pop("user", None)
            st.rerun()


def ui_admin_panel():
    st.subheader("🛠️ Admin — Setup & Scoring")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Managers", "Add Matchups", "Declare Winners", "Exports", "Settings"])

    with tab1:
        st.markdown("**Add or update managers (passcodes, admin flag).**")
        cols = st.columns([2,2,1,1])
        with cols[0]:
            name = st.text_input("Manager name", key="adm_name")
        with cols[1]:
            code = st.text_input("Passcode", key="adm_code")
        with cols[2]:
            is_admin = st.checkbox("Is admin?", key="adm_admin")
        with cols[3]:
            if st.button("Save manager", use_container_width=True):
                if name and code:
                    upsert_manager(name, code, is_admin)
                    st.success(f"Saved {name}")
                else:
                    st.error("Name and passcode are required.")
        conn = get_conn()
        st.dataframe(pd.read_sql_query("SELECT id, name, is_admin FROM managers ORDER BY name", conn))

    with tab2:
        st.markdown("**Create weekly matchups.**")
        cols = st.columns([1,2,2,2,2])
        with cols[0]:
            week = st.number_input("Week", min_value=1, max_value=30, value=1, step=1)
        with cols[1]:
            game_slug = st.text_input("Game (e.g., DAL@PHI)")
        with cols[2]:
            team_a = st.text_input("Team A label (e.g., DAL)")
        with cols[3]:
            team_b = st.text_input("Team B label (e.g., PHI)")
        with cols[4]:
            enable_lock = st.checkbox("Set lock", key="lock_enable")
            lock_dt = None
            if enable_lock:
                ld = st.date_input("Lock date", value=date.today(), key="lock_date")
                lt = st.time_input("Lock time", value=time(13, 0), key="lock_time")
                lock_dt = datetime.combine(ld, lt)
            if st.button("Add matchup", use_container_width=True):
                if week and game_slug and team_a and team_b:
                    add_matchup(int(week), game_slug, team_a, team_b, lock_dt)
                    st.success(f"Added {game_slug} to week {week}")
                else:
                    st.error("All fields (except lock) are required.")
        st.divider()
        w = st.number_input("View week", min_value=1, max_value=30, value=1, step=1, key="view_week_adm")
        st.dataframe(list_matchups(int(w)))

    with tab3:
        st.markdown("**Declare winners to grade picks.**")
        w = st.number_input("Week to grade", min_value=1, max_value=30, value=1, step=1, key="grade_week")
        df = list_matchups(int(w))
        if df.empty:
            st.info("No matchups yet for this week.")
        else:
            for _, row in df.iterrows():
                cols = st.columns([3,2,2,2,2])
                with cols[0]:
                    st.markdown(f"**{row['game_slug']}** — A: {row['team_a']} | B: {row['team_b']}")
                with cols[1]:
                    st.write(f"Lock: {row['lock_datetime'] or '—'}")
                with cols[2]:
                    winner = st.selectbox(
                        "Winner",
                        options=["—", "A", "B"],
                        index=["—", "A", "B"].index(row["winner"] or "—"),
                        key=f"winner_{row['id']}"
                    )
                with cols[3]:
                    if st.button("Save", key=f"save_w_{row['id']}"):
                        set_winner(int(row["id"]), None if winner == "—" else winner)
                        st.toast(f"Saved winner for {row['game_slug']}")
                with cols[4]:
                    if st.button("Clear winner", key=f"clear_w_{row['id']}"):
                        set_winner(int(row["id"]), None)
                        st.toast(f"Cleared winner for {row['game_slug']}")

    with tab4:
        st.markdown("**CSV exports**")
        w = st.number_input("Week to export", min_value=1, max_value=30, value=1, step=1, key="export_week")
        dfp = picks_for_week(int(w))
        if not dfp.empty:
            st.download_button("Download Picks CSV", data=dfp.to_csv(index=False), file_name=f"picks_week_{w}.csv")
        dfl = leaderboard()
        if not dfl.empty:
            st.download_button("Download Leaderboard CSV", data=dfl.to_csv(index=False), file_name="leaderboard.csv")

    with tab5:
        st.markdown("**League Settings**")
        current_mode = get_setting('reveal_picks', 'always')
        label_to_val = {
            "Reveal picks: Always (public)": "always",
            "Reveal picks: Only after a manager has picked": "after_pick",
            "Reveal picks: Only after matchup locks": "after_lock",
        }
        val_to_label = {v:k for k,v in label_to_val.items()}
        choice = st.selectbox("Visibility of manager picks", options=list(label_to_val.keys()), index=[*label_to_val.values()].index(current_mode) if current_mode in label_to_val.values() else 0)
        new_mode = label_to_val[choice]
        if st.button("Save settings"):
            set_setting('reveal_picks', new_mode)
            st.success(f"Saved: {val_to_label[new_mode]}")
        st.caption(f"Current database file: **{DB_PATH}** (set via environment variable or Streamlit Secrets)")


def ui_make_picks(user):
    st.subheader("✅ Make Your Picks")
    week = st.number_input("Week", min_value=1, max_value=30, value=1, step=1, key="pick_week")
    df = list_matchups(int(week))
    if df.empty:
        st.info("No matchups yet for this week. Check back later ✌️")
        return

    for _, row in df.iterrows():
        lock_label = f"(locks {row['lock_datetime']})" if row['lock_datetime'] else ""
        st.markdown(f"**{row['game_slug']}** {lock_label}")
        cols = st.columns([3,1,1])
        with cols[0]:
            st.write(f"A: {row['team_a']}  |  B: {row['team_b']}")
            current = get_user_pick(int(week), user['id'], int(row['id']))
            if current:
                picked_team = row['team_a'] if current == 'A' else row['team_b']
                st.caption(f"Your pick: **{picked_team}**")
        with cols[1]:
            if st.button(f"Pick {row['team_a']}", key=f"pickA_{row['id']}"):
                ok = record_pick(int(week), user['id'], int(row['id']), "A")
                st.toast("Pick saved ✅" if ok else "Locked ⛔")
        with cols[2]:
            if st.button(f"Pick {row['team_b']}", key=f"pickB_{row['id']}"):
                ok = record_pick(int(week), user['id'], int(row['id']), "B")
                st.toast("Pick saved ✅" if ok else "Locked ⛔")

        # Show league picks based on visibility rule
        reveal_mode = get_setting('reveal_picks', 'always')
        user_pick = get_user_pick(int(week), user['id'], int(row['id']))
        allowed = can_show_picks(reveal_mode, bool(user_pick), row['lock_datetime'])
        with st.expander("League picks", expanded=False):
            if not allowed:
                if reveal_mode == 'after_pick':
                    st.caption("📌 Picks become visible after you make your pick.")
                elif reveal_mode == 'after_lock':
                    st.caption("⏱️ Picks unlock after the matchup lock time.")
            else:
                pfm = picks_for_matchup(int(row['id']))
                names_a = pfm[pfm['pick']=='A']['manager'].tolist()
                names_b = pfm[pfm['pick']=='B']['manager'].tolist()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**{row['team_a']}** — {len(names_a)} picks")
                    st.write(", ".join(names_a) if names_a else "—")
                with c2:
                    st.markdown(f"**{row['team_b']}** — {len(names_b)} picks")
                    st.write(", ".join(names_b) if names_b else "—")
        st.divider()

    st.caption("You can change your pick until the matchup lock time.")


def ui_weekly_results():
    st.subheader("📊 Weekly Results & Pick Trends")
    week = st.numberst.subheader("📊 Weekly Results & Pick Trends")
    st.markdown("<div class='cyc-card'>", unsafe_allow_html=True)=1, step=1, key="results_week")
    dfp = picks_for_week(int(week))
    if dfp.empty:
        st.info("No picks for this week yet.")
        return

    st.markdown("**All Picks**")
    st.dataframe(dfp[["manager", "game_slug", "team_a", "team_b", "pick", "winner", "created_at"]])

    dist = pick_distribution(int(week))
    if not dist.empty:
        st.markdown("**Pick Distribution**")
        st.dataframe(dist[["game_slug", "team_a", "team_b", "A", "B", "pct_A", "pct_B", "winner"]])
        chart_data = dist.melt(id_vars=["game_slug","team_a","team_b"], value_vars=["pct_A","pct_B"], var_name="side", value_name="share")
        chart_data['team'] = chart_data.apply(lambda r: r['team_a'] if r['side']=='pct_A' else r['team_b'], axis=1)
        c = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('game_slug:N', title='Matchup'),
            y=alt.Y('share:Q', title='Pick share', axis=alt.Axis(format='%')),
            color='team:N',
            tooltip=['game_slug','team','share']
        ).properties(height=240)
        st.altair_chart(c, use_container_width=True)

    # Consensus + upset flags for this week (default 50% threshold)
    cons = matchup_consensus(0.5)
    cons_wk = cons[cons['week'] == int(week)].copy()
    if not cons_wk.empty:
        cons_wk['consensus_pick'] = cons_wk.apply(lambda r: ('A' if r['pct_A']>r['pct_B'] else ('B' if r['pct_B']>r['pct_A'] else '—')), axis=1)
        view_cols = ["game_slug","team_a","team_b","pct_A","pct_B","winner","consensus_pick","upset_flag"]
        st.markdown("**Consensus & Upsets**")
        st.dataframe(cons_wk[view_cols])

        # Who picked whom (named) for a selectecons_wk[view_cols])
        st.markdown("</div>", unsafe_allow_html=True)atchups = cons_wk[['matchup_id','game_slug','team_a','team_b']].drop_duplicates().sort_values('matchup_id')
        ids = matchups['matchup_id'].tolist()
        labels = [f"{g} — {a} vs {b}" for g,a,b in zip(matchups['game_slug'], matchups['team_a'], matchups['team_b'])]
        if ids:
            sel = st.selectbox("Who picked whom (select matchup)", options=list(range(len(ids))), format_func=lambda i: labels[i])
            chosen_id = ids[sel]
            pfm = picks_for_matchup(int(chosen_id))
            a_names = pfm[pfm['pick']=='A']['manager'].tolist()
            b_names = pfm[pfm['pick']=='B']['manager'].tolist()
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"**{matchups.iloc[sel]['team_a']}** — {len(a_names)} picks")
                st.write(", ".join(a_names) if a_names else "—")
            with cb:
                st.markdown(f"**{matchups.iloc[sel]['team_b']}** — {len(b_names)} picks")
                st.write(", ".join(b_names) if b_names else "—")


def ui_leaderboard():
    st.subheader("🏅 Leaderboard")
    up_to_week = st.number_input("Include results through week", min_value=1, max_value=30, value=1, step=1, key="lb_week")
    df = leaderboard(int(up_to_week))
    if df.empty:
        st.info("Leaderboard will appear after winners are declared.")
        return
    st.dataframe(df[["manager", "correct", "total_graded", "win_pct", "total_picked"]])
    top = df.iloc[0]
    st.success(f"Current leader: {top['manager']} — {int(top['correct'])} correct (Win% {top['win_pct']:.3f})")


def ui_analytics():
    st.subheader("📈 Season Analytics")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Season Overview (to date)**")
        so = season_overview()
        if so.empty:
            st.info("No data yet — make some picks!")
        else:
            st.dataframe(so[["manager","correct","total_graded","win_pct","avg_correct_per_week","total_picked","weeks_played"]])
    with colB:
        st.markdown("**Weekly Correct Picks (matrix)**")
        mat = weekly_breakdown_matrix()
        if mat.empty:
            st.info("Appears after first graded week.")
        else:
            st.dataframe(mat)

    st.divider()
    st.markdown("**Upset Analytics**")
    threshold_pct = st.slider("Upset threshold (winner's pick share < this %)", min_value=10, max_value=50, value=50, step=5, help="Default is 50%. Lower this to only count bigger surprises.")
    um = upset_metrics(threshold_pct/100.0)
    if um['graded_consensus'] == 0:
        st.info("Upset metrics appear after at least one graded matchup with a non-tied consensus.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Upset Rate", f"{um['overall_rate']*100:.1f}%")
        m2.metric("Upsets", str(um['num_upsets']))
        m3.metric("Graded Consensus Games", str(um['graded_consensus']))
        c = alt.Chart(um['weekly_rates']).mark_line(point=True).encode(
            x=alt.X('week:O', title='Week'),
                y=alt.Y('cum_win_pct:Q', title='Cumulative Win%', axis=alt.Axis(format='%')),
                color='manager:N',
                tooltip=['manager','week','cum_correct','cum_graded','cum_win_pct']
            ).properties(height=320)
            st.altair_chart(lc, use_container_width=True)

# ------------------------------
# Main app
# ------------------------------

def main():
    init_db()
