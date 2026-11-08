"""Exercise dashboard navigation and filters against the committed evidence."""
from pathlib import Path
import pytest
pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def app():
    return AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()

def healthy(at):
    assert not at.exception
    assert not at.error

def test_campaign_outcomes_and_segments():
    at=app();healthy(at)
    assert at.metric[0].value=='64,000'
    for outcome in ['visit','spend','conversion']:
        at.selectbox(key='outcome').select(outcome).run();healthy(at)
    at.selectbox(key='comparison').select('womens_vs_control').run()
    at.selectbox(key='segment').select('newbie').run();healthy(at)

def test_uplift_models_and_budget():
    at=app();at.sidebar.radio[0].set_value('Uplift targeting').run();healthy(at)
    assert at.metric[0].value=='8,523'
    for learner in ['s_learner','propensity','t_learner']:
        at.selectbox(key='learner').select(learner).run();healthy(at)
    at.selectbox(key='budget').select(0.05).run();healthy(at)
    assert '-0.73' in at.metric[3].value

def test_economics_controls_and_value():
    import pandas as pd
    at=app();at.sidebar.radio[0].set_value('Campaign economics').run();healthy(at)
    at.selectbox(key='policy').select('uplift_t_learner')
    at.selectbox(key='share').select(0.05)
    at.selectbox(key='margin').select(0.3)
    at.selectbox(key='cost').select(0.002)
    at.run();healthy(at)
    assert at.metric[0].value=='$104.69'
    assert at.metric[1].value=='193'
    for strategy in ['random','uplift_s_learner','propensity_response_model']:
        at.selectbox(key='policy').select(strategy).run();healthy(at)

def test_evidence_navigation():
    at=app();at.sidebar.radio[0].set_value('Evidence & planning').run();healthy(at)
    for table in ['power_analysis','uplift_ranking_summary']:
        at.selectbox(key='evidence').select(table).run();healthy(at)
