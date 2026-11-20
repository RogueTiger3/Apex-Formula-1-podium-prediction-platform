"""Apex dashboard entry point:  streamlit run streamlit_app.py"""
import streamlit as st

st.set_page_config(page_title="Apex · F1 Predictive Analytics", page_icon="🏁", layout="wide")

pages = [
    st.Page("app/views/home.py", title="Home", icon=":material/dashboard:", default=True),
    st.Page("app/views/predictions.py", url_path="predictions", title="Race Predictions", icon=":material/emoji_events:"),
    st.Page("app/views/drivers.py", url_path="drivers", title="Driver Analytics", icon=":material/person:"),
    st.Page("app/views/constructors.py", url_path="constructors", title="Constructor Analytics", icon=":material/groups:"),
    st.Page("app/views/simulator.py", url_path="simulator", title="Race Simulator", icon=":material/casino:"),
    st.Page("app/views/circuits.py", url_path="circuits", title="Circuit Analytics", icon=":material/route:"),
    st.Page("app/views/performance.py", url_path="performance", title="Model Performance", icon=":material/monitoring:"),
    st.Page("app/views/data.py", url_path="data", title="Data Management", icon=":material/database:"),
]
st.navigation(pages).run()
