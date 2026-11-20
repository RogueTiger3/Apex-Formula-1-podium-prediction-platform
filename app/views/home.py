import json

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app import ui
from apex import config, predict

ui.setup("Race weekend dashboard")

nr = predict.next_race()
names, codes, tnames = ui.names(), ui.codes(), ui.team_names()

# ------------------------------------------------------------------ next race
c1, c2 = st.columns([1.3, 1])
with c1:
    if nr is None:
        st.info("No upcoming race in the calendar. The season may be over; the next calendar appears when Jolpica publishes it.")
    else:
        start = pd.Timestamp(nr["race_start_utc"])
        st.markdown(f'<div class="apex-card"><div class="apex-kicker">Next Grand Prix · Round {int(nr["round"])}</div>'
                    f'<h2 style="margin:.2rem 0">{nr["race_name"]}</h2>'
                    f'<div class="apex-sub">{nr["circuit_name"]} · {nr["locality"]}, {nr["country"]}</div>'
                    f'<div class="apex-sub">Lights out {start:%a %d %b %Y, %H:%M} UTC</div></div>',
                    unsafe_allow_html=True)
with c2:
    if nr is not None:
        iso = pd.Timestamp(nr["race_start_utc"]).isoformat()
        components.html(f"""
        <div style="font-family:sans-serif;background:#141924;border:1px solid #232A38;border-radius:10px;
             padding:16px 18px;color:#E6E8EE;height:118px;box-sizing:border-box">
          <div style="color:#8A93A6;text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:600">Countdown to lights out</div>
          <div id="cd" style="font-size:2.2rem;font-weight:700;font-variant-numeric:tabular-nums;margin-top:10px"></div>
        </div>
        <script>
          const t = new Date("{iso}").getTime();
          function tick() {{
            let s = Math.max(0, Math.floor((t - Date.now())/1000));
            const d = Math.floor(s/86400); s%=86400; const h=Math.floor(s/3600); s%=3600;
            const m = Math.floor(s/60); s%=60;
            document.getElementById("cd").textContent = s+m+h+d===0 ? "Race underway or finished" :
              `${{d}}d ${{String(h).padStart(2,'0')}}h ${{String(m).padStart(2,'0')}}m ${{String(s).padStart(2,'0')}}s`;
          }}
          tick(); setInterval(tick, 1000);
        </script>""", height=125)

# ------------------------------------------------------------------ predictions
st.subheader("Podium prediction for the next race")
run = ui.latest_run(nr["race_id"]) if nr is not None else None
if run is None:
    st.info("No prediction stored for the next race yet. Open **Race Predictions** to generate one.")
else:
    preds = ui.run_predictions(run["run_id"])
    st.caption(f"Stage {int(run['stage'])} · {config.STAGES[int(run['stage'])]} · model {run['model_version']} · "
               f"generated {ui.fmt_ts(run['created_at'])}")
    cols = st.columns(3)
    for i, (col, (_, r)) in enumerate(zip(cols, preds.head(3).iterrows())):
        with col:
            ui.card(f"P{i + 1} most likely · {r['team']}", r["name"],
                    f"Podium {ui.pct(r['p_podium'])} · Win {ui.pct(r['p_win'])}", bar=ui.colour(r["constructor_id"], i))
    st.write("")
    a, b = st.columns([1.75, 1])
    with a:
        show = preds[["code", "name", "team", "p_win", "p_podium", "p_top10"]].head(10)
        st.dataframe(ui.as_pct(show), hide_index=True, width="stretch",
                     column_config={"code": "Code", "name": "Driver", "team": "Team", **ui.prob_columns()})
    with b:
        inp = json.loads(run["inputs_json"])
        st.markdown("**Most likely podium combinations**")
        combos = pd.DataFrame([dict(Podium=" · ".join(codes.get(x, x) for x in c["podium"]), Probability=c["probability"])
                               for c in inp.get("podium_combos", [])[:6]])
        if len(combos):
            st.dataframe(ui.as_pct(combos), hide_index=True, width="stretch",
                         column_config={"Probability": ui.combo_column(float(combos["Probability"].max()) * 100)})
            st.caption(f"Share of {inp.get('n_sims', 20000):,} simulated races. Even the most likely exact podium "
                       "usually happens in well under a fifth of them.")

# ------------------------------------------------------------------ standings / last race
st.divider()
s1, s2, s3 = st.columns([1, 1, 1.1])
standings = ui.q("SELECT * FROM standings ORDER BY kind, position")
with s1:
    st.subheader("Drivers' championship")
    d = standings[standings["kind"] == "driver"].head(10)
    if len(d):
        st.caption(f"{int(d['season'].iloc[0])} after round {int(d['after_round'].iloc[0])}")
        st.dataframe(d[["position", "entity_name", "points", "wins"]], hide_index=True, width="stretch",
                     column_config={"position": "Pos", "entity_name": "Driver", "points": st.column_config.NumberColumn("Pts", format="%d"), "wins": "Wins"})
with s2:
    st.subheader("Constructors' championship")
    d = standings[standings["kind"] == "constructor"].head(10)
    if len(d):
        st.caption(f"{int(d['season'].iloc[0])} after round {int(d['after_round'].iloc[0])}")
        st.dataframe(d[["position", "entity_name", "points", "wins"]], hide_index=True, width="stretch",
                     column_config={"position": "Pos", "entity_name": "Team", "points": st.column_config.NumberColumn("Pts", format="%d"), "wins": "Wins"})
with s3:
    last = ui.q("SELECT r.* FROM races r WHERE result_status IN ('final','provisional') ORDER BY race_start_utc DESC LIMIT 1")
    if len(last):
        lr = last.iloc[0]
        st.subheader("Last race")
        status = "provisional" if lr["result_status"] == "provisional" else "confirmed"
        st.caption(f"{lr['race_name']} · {pd.Timestamp(lr['race_start_utc']):%d %b %Y} · result {status}")
        res = ui.q("SELECT position, position_text, driver_id, constructor_id, grid, status FROM results WHERE race_id=?"
                   " ORDER BY COALESCE(position, 99)", (lr["race_id"],))
        res["Driver"] = res["driver_id"].map(names)
        res["Team"] = res["constructor_id"].map(tnames)
        st.dataframe(res[["position_text", "Driver", "Team", "grid"]].head(10), hide_index=True, width="stretch",
                     column_config={"position_text": "Pos", "grid": "Grid"})
        prev = ui.q("SELECT * FROM prediction_runs WHERE race_id=? ORDER BY stage DESC, created_at DESC LIMIT 1", (lr["race_id"],))
        if len(prev):
            pp = json.loads(prev.iloc[0]["predicted_podium"])
            actual = res.dropna(subset=["position"]).query("position <= 3")["driver_id"].tolist()
            st.caption(f"Apex predicted podium: {', '.join(codes.get(x, x) for x in pp)} · "
                       f"{len(set(pp) & set(actual))} of 3 correct")

# ------------------------------------------------------------------ model status
st.divider()
reg = ui.q("SELECT * FROM model_registry WHERE is_active=1 ORDER BY stage")
log = ui.q("SELECT * FROM update_log WHERE status IN ('success','partial') ORDER BY id DESC LIMIT 1")
m1, m2, m3, m4 = st.columns(4)
with m1:
    ui.card("Model version", reg["model_version"].iloc[0].split("-")[0] if len(reg) else "none",
            f"trained {ui.ago(reg['trained_at'].iloc[0])}" if len(reg) else "")
with m2:
    ui.card("Trained on races up to", reg["data_cutoff_race"].iloc[0].replace("_", " R") if len(reg) else "–",
            f"{int(reg['n_train'].max()):,} driver-race rows" if len(reg) else "")
with m3:
    ui.card("Last data update", ui.ago(log["finished_at"].iloc[0]) if len(log) else "never",
            ui.fmt_ts(log["finished_at"].iloc[0]) if len(log) else "")
with m4:
    if len(reg):
        sel = reg.set_index("stage")["algorithm"].map(lambda a: a.replace("_", " ").title())
        ui.card("Selected models", sel.get(4, "–"), f"pre-race stage · pre-weekend: {sel.get(1, '–')}")
