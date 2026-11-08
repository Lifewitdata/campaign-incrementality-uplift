# Streamlit decision dashboard

Run `python -m pip install -r requirements.txt`, then
`python -m streamlit run streamlit_app.py` from the repository root.

The four app pages cover campaign impact, uplift targeting, scenario economics,
and evidence/experiment planning. The app reads `outputs/tables/` using paths
relative to its own file and needs no cloud credentials or training step.

`outputs/dashboard/` contains reporting exports for the warehouse preparation
workflow. The dashboard itself uses the primary evidence tables to avoid separate
copies drifting apart. Filter selections never alter the source data.

The project includes no BI workbooks or external dashboard links. The main README
contains the local setup and optional Streamlit Community Cloud deployment guide.
