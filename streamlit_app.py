"""Evidence-based campaign decisions. Runs from committed aggregate CSVs offline."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / 'outputs' / 'tables'
COLORS = ['#147D92', '#356AC3', '#A77527', '#8896A5']
NAMES = {'propensity': 'Purchase-response baseline', 's_learner': 'S-learner',
         't_learner': 'T-learner', 'uplift_t_learner': 'T-learner',
         'uplift_s_learner': 'S-learner', 'propensity_response_model': 'Purchase-response baseline',
         'random': 'Random targeting'}


def read(name, required=()):
    frame = pd.read_csv(TABLES / f'{name}.csv')
    missing = set(required) - set(frame)
    if frame.empty or missing:
        raise ValueError(f'{name}: empty data or missing columns {sorted(missing)}')
    return frame


def chart(fig):
    fig.update_layout(template='plotly_white', colorway=COLORS, margin=dict(l=20,r=20,t=50,b=30),
                      font=dict(family='Arial', size=13), legend_title_text='')
    st.plotly_chart(fig, width='stretch')


def intervals(frame, label, value, low, high, title, axis, scale=1):
    fig = go.Figure()
    for row in frame.to_dict('records'):
        fig.add_trace(go.Scatter(x=[row[low]*scale,row[high]*scale], y=[str(row[label])]*2,
            mode='lines', line=dict(color=COLORS[0],width=3), showlegend=False,
            hovertemplate='%{x:.3f}<extra>Interval bound</extra>'))
    fig.add_trace(go.Scatter(x=frame[value]*scale,y=frame[label].astype(str),mode='markers',
        marker=dict(size=10,color=COLORS[1]),showlegend=False,hovertemplate='%{x:.3f}<extra>Estimate</extra>'))
    fig.add_vline(x=0,line_dash='dash',line_color='#8896A5')
    fig.update_layout(title=title,xaxis_title=axis,height=max(300,45*len(frame)+100))
    return fig


def download(frame, name):
    with st.expander('Inspect and download the displayed evidence'):
        st.dataframe(frame,hide_index=True,width='stretch')
        st.download_button('Download CSV',frame.to_csv(index=False).encode(),file_name=name+'.csv',mime='text/csv',key=name)


def campaign():
    st.title('Did the campaigns work?')
    st.caption('Randomized email experiment · 64,000 customers · Two-week outcomes · USD')
    arms=read('arm_summary',['n_customers','conversion_rate'])
    effects=read('experiment_effects',['effect','ci_low_fwer'])
    conv=effects[(effects.outcome=='conversion') & (effects.measure=='risk_difference')]
    cols=st.columns(3)
    cols[0].metric('Customers randomized',f'{arms.n_customers.sum():,}')
    for col, label in zip(cols[1:],['Mens E-Mail','Womens E-Mail']):
        row=conv[conv.treatment_label==label].iloc[0]
        col.metric(label+' lift',f'{100*row.effect:+.2f} pp')
    st.info('Both campaigns increased conversion versus no email. A larger point estimate is not a direct statistical test between the two campaigns.')
    outcome=st.selectbox('Outcome',['conversion','visit','spend'],key='outcome')
    metric={'conversion':'conversion_rate','visit':'visit_rate','spend':'spend_per_customer'}[outcome]
    scale=1 if outcome=='spend' else 100
    unit='USD per customer' if outcome=='spend' else 'Percentage points'
    a,b=st.columns(2)
    with a:
        display=arms.assign(display_value=arms[metric]*scale)
        chart(px.bar(display,x='arm',y='display_value',color='arm',title='Observed outcome by campaign',
                     labels={'display_value':'USD per customer' if outcome=='spend' else 'Rate (%)','arm':''},
                     color_discrete_map={'No E-Mail':'#8896A5','Mens E-Mail':'#356AC3','Womens E-Mail':'#147D92'}))
    with b:
        selected=effects[(effects.outcome==outcome)&effects.measure.isin(['risk_difference','mean_difference'])]
        chart(intervals(selected,'treatment_label','effect','ci_low_fwer','ci_high_fwer','Increment versus no email',unit,scale))
    st.caption('Headline intervals use the supplied family-adjusted bounds (97.5% per comparison); p-values use Holm adjustment. Conversion/visit use proportion methods; spend uses the recorded bootstrap analysis.')
    download(selected,'campaign_effects')
    st.subheader('Where might the effect differ?')
    sub=read('subgroup_effects')
    c,d=st.columns(2)
    comp=c.selectbox('Campaign comparison',list(sub.comparison.unique()),key='comparison')
    variable=d.selectbox('Customer characteristic',list(sub.subgroup_variable.unique()),key='segment')
    subset=sub[(sub.comparison==comp)&(sub.subgroup_variable==variable)]
    chart(intervals(subset,'subgroup','effect','ci_low','ci_high','Exploratory segment effects','Conversion lift (pp)',100))
    het=read('heterogeneity_tests')
    finding=het[(het.comparison==comp)&(het.subgroup_variable==variable)].iloc[0]
    st.write(f"Adjusted interaction p-value: **{finding.p_holm:.4g}**. {finding.interpretation}.")
    st.caption('Subgroup intervals are exploratory and unadjusted. Confirm any segment hypothesis in a new experiment. Mens/womens flags describe past merchandise purchases, not gender.')


def targeting():
    st.title('Can ML improve targeting?')
    st.caption('Mens email versus no email · Held-out comparison · Individual treatment effects are not observed')
    deciles=read('uplift_deciles_holdout',['learner','n_customers'])
    learner=st.selectbox('Ranking model',list(deciles.learner.unique()),index=2,format_func=lambda x:NAMES[x],key='learner')
    selected=deciles[deciles.learner==learner].sort_values('decile')
    ranking=read('uplift_ranking_summary')
    q=ranking[ranking.learner==learner].iloc[0]
    a,b,c=st.columns(3)
    a.metric('Holdout customers',f'{int(selected.n_customers.sum()):,}')
    b.metric('Conversion Qini coefficient',f'{q.holdout_qini_coefficient:.3f}')
    c.metric('Top-decile observed uplift',f'{selected.iloc[0].observed_uplift_conversion*100:+.2f} pp')
    st.warning('The saved holdout does not establish an advantage over purchase-response targeting. Do not deploy a targeting policy based only on this ranking.')
    chart(intervals(selected,'decile','observed_uplift_conversion','uplift_ci_low','uplift_ci_high',
                    'Observed uplift by predicted rank decile','Conversion lift (pp)',100))
    st.caption('Decile 1 has the highest predicted score. The purchase-response baseline ranks conversion probability, not causal uplift. Decile intervals are unadjusted diagnostics.')
    curves=read('uplift_curves_holdout');curves=curves[curves.outcome=='conversion'].copy()
    curves['gain_above_random']=curves.qini_gain-curves.random_line_gain
    curves['model']=curves.learner.map(NAMES)
    fig=px.line(curves,x='share_targeted',y='gain_above_random',color='model',markers=True,
                title='Cumulative gain above random targeting',labels={'share_targeted':'Share targeted','gain_above_random':'Additional conversions above random'})
    fig.update_xaxes(tickformat='.0%');fig.add_hline(y=0,line_dash='dash');chart(fig)
    head=read('policy_head_to_head');budget=st.selectbox('Targeting budget',sorted(head.share_targeted.unique()),format_func=lambda x:f'{x:.0%}',key='budget')
    h=head[head.share_targeted==budget].iloc[0]
    st.metric('T-learner minus response baseline',f'{h.incremental_conversions_difference:+.2f} conversions')
    st.write(f'Paired bootstrap interval: **{h.difference_ci_low:+.2f} to {h.difference_ci_high:+.2f}**. Selected records: **{int(h.n_selected):,}**.')
    st.caption('This head-to-head comparison is always T-learner versus response baseline, regardless of the decile model selector above.')
    download(head,'targeting_comparison')


def economics():
    st.title('What could a campaign be worth?')
    st.warning('SCENARIO - NOT REALISED ROI. Costs and margins are assumptions. Values refer to the selected historical treated subset.')
    df=read('economics_scenarios',['strategy','scenario_net_value_usd'])
    a,b,c,d=st.columns(4)
    strategy=a.selectbox('Policy',list(df.strategy.unique()),format_func=lambda x:NAMES.get(x,x),key='policy')
    share=b.selectbox('Share targeted',sorted(df.share_targeted.unique()),index=len(df.share_targeted.unique())-1,format_func=lambda x:f'{x:.0%}',key='share')
    margin=c.selectbox('Assumed gross margin',sorted(df.assumed_gross_margin_rate.unique()),index=1,format_func=lambda x:f'{x:.0%}',key='margin')
    cost=d.selectbox('Assumed cost per send',sorted(df.assumed_cost_per_send_usd.unique()),index=1,format_func=lambda x:f'${x:.3f}',key='cost')
    row=df[(df.strategy==strategy)&(df.share_targeted==share)&(df.assumed_gross_margin_rate==margin)&(df.assumed_cost_per_send_usd==cost)].iloc[0]
    a,b,c=st.columns(3)
    a.metric('Scenario net contribution',f'${row.scenario_net_value_usd:,.2f}')
    b.metric('Historical treated records',f'{int(row.n_selected_treated):,}')
    c.metric('Assumed send cost',f'${row.scenario_send_cost_usd:,.2f}')
    st.write(f'Conditional bootstrap range: **${row.scenario_net_value_ci_low_usd:,.2f} to ${row.scenario_net_value_ci_high_usd:,.2f}**.')
    grid=df[(df.strategy==strategy)&(df.share_targeted==share)].pivot(index='assumed_gross_margin_rate',columns='assumed_cost_per_send_usd',values='scenario_net_value_usd')
    fig=go.Figure(go.Heatmap(z=grid.values,x=[f'${x:.3f}' for x in grid.columns],y=[f'{x:.0%}' for x in grid.index],
        colorscale='RdBu',zmid=0,text=grid.round(0).astype(int).astype(str).values,texttemplate='$%{text}',colorbar=dict(title='USD')))
    fig.update_layout(title='Sensitivity across all cost and margin assumptions',xaxis_title='Cost per send',yaxis_title='Gross margin');chart(fig)
    st.caption('The heatmap shows the entire assumption grid for the selected policy and targeting share. The KPI cards use the selected margin and cost.')
    selected=df[(df.assumed_gross_margin_rate==margin)&(df.assumed_cost_per_send_usd==cost)].copy();selected['policy']=selected.strategy.map(NAMES)
    fig=px.line(selected,x='share_targeted',y='scenario_net_value_usd',color='policy',markers=True,
        title='Policy value across targeting budgets',labels={'share_targeted':'Share targeted','scenario_net_value_usd':'Scenario net contribution (USD)'})
    fig.update_xaxes(tickformat='.0%');fig.add_hline(y=0,line_dash='dash');chart(fig)
    st.caption('Net contribution = incremental spend in historical treated subset × assumed margin − historical treated count × assumed send cost. Intervals condition on fixed assumptions and exclude long-term effects.')
    download(selected,'economics_selected_assumptions')


def evidence():
    st.title('Evidence and next experiment')
    st.write('These pages read version-controlled aggregate results. Opening the app does not retrain models, require raw customer rows or connect to Google Cloud.')
    choice=st.selectbox('Evidence table',['data_quality_checks','design_checks','baseline_balance','power_analysis','uplift_ranking_summary'],key='evidence')
    download(read(choice),'evidence_'+choice)
    st.info('Keep a randomized no-email holdout. Choose the minimum worthwhile lift and campaign budget before the next test; then compare a frozen uplift policy with response and random baselines.')
    st.markdown('[Analysis and reproducibility repository](https://github.com/Lifewitdata/campaign-incrementality-uplift)')
    st.caption('BigQuery deployment is a separate batch workflow. A local deployment receipt is required to verify cloud job status. No scheduled service or hosted Streamlit URL is claimed here.')


def main():
    st.set_page_config(page_title='Campaign Decision Lab',page_icon='📊',layout='wide')
    st.sidebar.title('Campaign Decision Lab')
    st.sidebar.caption('Incrementality · Uplift · Economics')
    page=st.sidebar.radio('Explore',['Campaign impact','Uplift targeting','Campaign economics','Evidence & planning'])
    st.sidebar.markdown('**Source:** Hillstrom (2008)\n\n**Scope:** one randomized campaign, two-week follow-up. All spending is USD.')
    try:
        {'Campaign impact':campaign,'Uplift targeting':targeting,'Campaign economics':economics,'Evidence & planning':evidence}[page]()
    except (OSError,ValueError,KeyError,IndexError) as exc:
        st.error(f'Unable to display project evidence: {exc}')
        st.info('Restore outputs/tables from the repository, or regenerate the analysis using the README pipeline commands.')
        st.stop()

if __name__=='__main__':
    main()
