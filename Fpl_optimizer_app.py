import streamlit as st
import pandas as pd
import requests
from pulp import LpProblem, LpVariable, LpMaximize, lpSum
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import io

# Konfigurasjon
st.set_page_config(page_title="FPL Optimaliseringsapp", layout="wide")
st.title("⚽ FPL Optimaliseringsapp")
st.caption("Bygg ditt beste Fantasy Premier League-lag – med transferplan, chips og differensialer.")

# Hent grunnleggende data
@st.cache_data
def load_fpl_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    data = r.json()
    elements = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    positions = pd.DataFrame(data['element_types'])
    elements['team'] = elements['team'].map(teams.set_index('id')['name'])
    elements['position'] = elements['element_type'].map(positions.set_index('id')['singular_name'])
    elements['is_available'] = elements['status'] == 'a'
    return elements, teams.set_index('id')['name'].to_dict(), teams.set_index('name')['id'].to_dict()

@st.cache_data
def load_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    return requests.get(url).json()

players, id_to_team, team_to_id = load_fpl_data()
fixtures = load_fixtures()

# Sidebar
st.sidebar.header("⚙️ Preferanser")
exclude_teams = st.sidebar.multiselect("Unngå spillere fra lag:", list(team_to_id.keys()))
prefer_teams = st.sidebar.multiselect("Favoriser spillere fra lag:", list(team_to_id.keys()))
filter_injured = st.sidebar.checkbox("Ekskluder skadde/suspenderte", value=True)
use_differentials = st.sidebar.checkbox("🎯 Favoriser lav eierandel (<15%)", value=False)

st.sidebar.markdown("### 💡 Chipvalg")
chip_mode = st.sidebar.radio("Simuler valgt chip:", ["Ingen chip", "Bench Boost", "Triple Captain"])

st.sidebar.markdown("### 🎯 Ditt nåværende FPL-lag")
use_manual_team = st.sidebar.checkbox("Start med mitt faktiske lag")
manual_team = []
if use_manual_team:
    manual_team = st.sidebar.multiselect("Velg dine nåværende 15 spillere:", players['web_name'].sort_values().unique())

st.sidebar.markdown("### 💰 Makspris per posisjon (i millioner)")
pos_budget = {
    "Goalkeeper": st.sidebar.slider("Keepere", 4.0, 6.0, 5.0),
    "Defender": st.sidebar.slider("Forsvar", 4.0, 8.0, 6.0),
    "Midfielder": st.sidebar.slider("Midtbane", 4.5, 13.0, 8.0),
    "Forward": st.sidebar.slider("Spisser", 4.5, 12.5, 8.0)
}

st.sidebar.markdown("### 🔄 Transfer Planner")
use_transfer_planner = st.sidebar.checkbox("Simuler flere runder fremover", value=True)
num_rounds = st.sidebar.slider("Antall runder å planlegge:", 2, 5, 3)

point_mode = st.sidebar.radio("Optimaliser basert på:", ["📊 Historiske totalpoeng", "🔮 Neste kamp", "📅 Neste 5 kamper"])

# Beregn forventede poeng
def get_expected_points(players, fixtures, team_to_id, n=1):
    team_fixtures = {tid: [] for tid in team_to_id.values()}
    for f in fixtures:
        if not f["finished"]:
            team_fixtures[f["team_h"]].append((f["team_a"], f["team_h_difficulty"], True))
            team_fixtures[f["team_a"]].append((f["team_h"], f["team_a_difficulty"], False))

    expected_points = []
    for i, row in players.iterrows():
        team_id = team_to_id.get(row['team'])
        form = row['form']
        fixtures_list = team_fixtures.get(team_id, [])[:n]
        exp = 0.0
        for opp_id, diff, is_home in fixtures_list:
            multiplier = (6 - diff) / 5
            if is_home:
                multiplier *= 1.1
            exp += float(form) * multiplier
        expected_points.append(round(exp, 2))
    players = players.copy()
    players['expected_points'] = expected_points
    return players

# Funksjon: optimaliser lag
def optimize_team(players, prefer_teams):
    prob = LpProblem("FPL_Team_Optimization", LpMaximize)
    vars = {i: LpVariable(f"player_{i}", cat='Binary') for i in players.index}

    def adjusted_score(i):
        base = players.loc[i, 'score']
        multiplier = 1.0
        if players.loc[i, 'team'] in prefer_teams:
            multiplier *= 1.10
        if use_differentials and players.loc[i, 'selected_by_percent'] < 15.0:
            multiplier *= 1.15
        return base * multiplier

    prob += lpSum([vars[i] * adjusted_score(i) for i in players.index])
    prob += lpSum([vars[i] * players.loc[i, 'now_cost'] for i in players.index]) <= 1000

    def pos_constraint(pos, count):
        prob += lpSum([vars[i] for i in players[players['position'] == pos].index]) == count

    pos_constraint('Goalkeeper', 2)
    pos_constraint('Defender', 5)
    pos_constraint('Midfielder', 5)
    pos_constraint('Forward', 3)

    for team in players['team'].unique():
        prob += lpSum([vars[i] for i in players[players['team'] == team].index]) <= 3
    prob += lpSum(vars[i] for i in players.index) == 15

    if use_manual_team and manual_team:
        locked_ids = players[players['web_name'].isin(manual_team)].index.tolist()
        for i in locked_ids:
            prob += vars[i] == 1

    prob.solve()
    selected = [i for i in players.index if vars[i].value() == 1]
    result = players.loc[selected].copy()
    result = result.sort_values(by="score", ascending=False)
    result['role'] = ['👑 Kaptein', '🎖️ Visekaptein'] + ['Spiller'] * (len(result) - 2)
    return result.sort_values(by="position")

# Filtrering
players = players[players['minutes'] > 0]
if filter_injured:
    players = players[players['is_available']]
players = players[~players['team'].isin(exclude_teams)]
for pos, max_price in pos_budget.items():
    players = players[~((players['position'] == pos) & (players['now_cost'] > max_price * 10))]

# Velg poenggrunnlag
if point_mode == "📊 Historiske totalpoeng":
    players['score'] = players['total_points']
elif point_mode == "🔮 Neste kamp":
    players = get_expected_points(players, fixtures, team_to_id, n=1)
    players['score'] = players['expected_points']
elif point_mode == "📅 Neste 5 kamper":
    players = get_expected_points(players, fixtures, team_to_id, n=5)
    players['score'] = players['expected_points']

# Optimaliser nåværende lag
if st.button("⚡ Optimaliser Laget"):
    result = optimize_team(players, prefer_teams)
    result['fixtures'] = result.apply(lambda row: get_expected_points(players, fixtures, team_to_id, n=5).loc[row.name, 'expected_points'], axis=1)
    st.success("Optimalisert lag valgt!")

    total_cost = result['now_cost'].sum() / 10
    base_points = result['score'].sum()
    captain_score = result.iloc[0]['score']

    if chip_mode == "Triple Captain":
        total_points = base_points + 2 * captain_score
    elif chip_mode == "Bench Boost":
        total_points = base_points
    else:
        total_points = base_points - result.tail(4)['score'].sum() + captain_score

    st.metric("💰 Total kostnad", f"{total_cost:.1f}M")
    st.metric("📈 Forventet poeng", f"{total_points:.1f}")

    st.dataframe(result[['web_name', 'role', 'position', 'team', 'now_cost', 'score']])

    # Transfer planner
    if use_transfer_planner:
        st.subheader("🔄 Transferplan for neste runder")
        multi_round_results = []
        for r in range(num_rounds):
            players_round = get_expected_points(players, fixtures, team_to_id, n=r+1)
            players_round['score'] = players_round['expected_points']
            res = optimize_team(players_round, prefer_teams)
            multi_round_results.append(res)

        transfers = []
        for i in range(len(multi_round_results)):
            if i == 0:
                ins, outs, hits = ["—"], ["—"], 0
            else:
                prev = set(multi_round_results[i-1]['web_name'])
                curr = set(multi_round_results[i]['web_name'])
                ins = list(curr - prev)
                outs = list(prev - curr)
                hits = max(0, len(ins) - 1) * 4
            transfers.append({
                "Runde": i+1,
                "Spillere Inn": ", ".join(ins),
                "Spillere Ut": ", ".join(outs),
                "Gratis Bytter": 1,
                "Poengtap": hits,
                "Forventede Poeng": multi_round_results[i]['score'].sum()
            })

        df_transfers = pd.DataFrame(transfers)
        st.dataframe(df_transfers)

        # Transferkjede
        st.subheader("🔁 Transferkjede")
        for i in range(1, len(multi_round_results)):
            prev_ids = set(multi_round_results[i-1]['web_name'])
            curr_ids = set(multi_round_results[i]['web_name'])
            chain = list(zip(sorted(prev_ids - curr_ids), sorted(curr_ids - prev_ids)))
            if chain:
                st.markdown(f"**Runde {i} ➝ {i+1}**")
                for o, n in chain:
                    st.markdown(f"- `{o}` ➝ `{n}`")

        # Differensialkjede
        st.subheader("🧨 Differensialer i bytter")
        for i in range(1, len(multi_round_results)):
            prev_ids = set(multi_round_results[i-1]['web_name'])
            curr_ids = set(multi_round_results[i]['web_name'])
            new_in = curr_ids - prev_ids
            diffs = []
            for name in new_in:
                sel = players[players['web_name'] == name]
                if not sel.empty and float(sel.iloc[0]['selected_by_percent']) < 15:
                    diffs.append((name, sel.iloc[0]['selected_by_percent']))
            if diffs:
                st.markdown(f"**Runde {i} ➝ {i+1}**")
                for name, sel_pct in diffs:
                    st.markdown(f"- `{name}` *(eierandel: {sel_pct:.1f} %)*")

        # Visuell graf
        st.subheader("📈 Forventet poeng og poengtap")
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(df_transfers["Runde"], df_transfers["Forventede Poeng"], marker='o', label="Poeng")
        ax1.set_ylabel("Forventede Poeng")
        ax2 = ax1.twinx()
        ax2.bar(df_transfers["Runde"], df_transfers["Poengtap"], alpha=0.3, color='red', label="Poengtap")
        ax2.set_ylabel("Poengtap (hits)")
        plt.title("Forventede poeng og poengtap per runde")
        st.pyplot(fig)

    # Eksport
    if st.button("💾 Lagre laget som plan"):
        export_team = result[['web_name', 'role', 'position', 'team', 'now_cost', 'score']].copy()
        export_team['dato'] = datetime.now().strftime("%Y-%m-%d %H:%M")
        export_team['chip'] = chip_mode
        export_team['differensial_strategi'] = "Ja" if use_differentials else "Nei"
        export_team['rundeplan'] = f"{num_rounds} runder" if use_transfer_planner else "Kun neste runde"
        csv_out = io.StringIO()
        export_team.to_csv(csv_out, index=False)
        st.download_button(
            label="📥 Last ned planlagt lag (.csv)",
            data=csv_out.getvalue(),
            file_name=f"FPL_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
