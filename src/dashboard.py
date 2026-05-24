import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

df = pd.read_csv('data/ed_patient_data.csv', parse_dates=['arrival_datetime'])
BENCH = {1:5, 2:15, 3:30, 4:60, 5:120}
df['benchmark_min'] = df['triage_level'].map(BENCH)
df['breach']        = (df['wait_time_doctor_min'] > df['benchmark_min']).astype(int)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Hospital ED Dashboard"

dept_opts   = [{'label':'All Departments','value':'ALL'}] + \
              [{'label':d,'value':d} for d in sorted(df['department'].unique())]
ins_opts    = [{'label':'All Insurance','value':'ALL'}] + \
              [{'label':i,'value':i} for i in sorted(df['insurance_type'].unique())]
triage_opts = [{'label':'All Triage','value':0}] + \
              [{'label':f'ESI {i}','value':i} for i in range(1,6)]

app.layout = dbc.Container([
    dbc.Row([dbc.Col([
        html.H2("🏥 Hospital ED — Operations Dashboard"),
        html.P("Patient Flow & Performance Analytics · 2023", className="text-muted")
    ])]),
    html.Hr(),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id='dept',   options=dept_opts,   value='ALL', clearable=False), width=4),
        dbc.Col(dcc.Dropdown(id='ins',    options=ins_opts,    value='ALL', clearable=False), width=4),
        dbc.Col(dcc.Dropdown(id='triage', options=triage_opts, value=0,     clearable=False), width=4),
    ], className="mb-3"),
    dbc.Row(id='kpis', className="mb-3"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-hourly'),  width=8),
        dbc.Col(dcc.Graph(id='g-breach'),  width=4),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='g-los'),     width=6),
        dbc.Col(dcc.Graph(id='g-revenue'), width=6),
    ], className="mb-2"),
    dbc.Row([dbc.Col(dcc.Graph(id='g-sat'), width=12)]),
], fluid=True)


@app.callback(
    Output('kpis','children'),
    Output('g-hourly','figure'), Output('g-breach','figure'),
    Output('g-los','figure'),    Output('g-revenue','figure'),
    Output('g-sat','figure'),
    Input('dept','value'), Input('ins','value'), Input('triage','value')
)
def update(dept, ins, triage):
    d = df.copy()
    if dept   != 'ALL': d = d[d['department']    == dept]
    if ins    != 'ALL': d = d[d['insurance_type'] == ins]
    if triage !=  0:    d = d[d['triage_level']   == triage]
    if len(d) == 0:
        empty = go.Figure()
        return [], empty, empty, empty, empty, empty

    # KPI cards
    kpi_vals = [
        ("Total Visits",     f"{len(d):,}",                              "primary"),
        ("Avg Doctor Wait",  f"{d['wait_time_doctor_min'].mean():.0f} min", "danger"),
        ("Avg LOS",          f"{d['length_of_stay_hrs'].mean():.2f} hrs",   "warning"),
        ("Readmission Rate", f"{d['readmitted_30d'].mean()*100:.1f}%",      "info"),
        ("Avg Satisfaction", f"{d['satisfaction_score'].mean():.1f}/10",    "success"),
        ("Total Billed",     f"${d['billed_amount_usd'].sum()/1e6:.1f}M",   "secondary"),
    ]
    cards = [dbc.Col(dbc.Card(dbc.CardBody([
        html.P(lbl, className="card-text text-muted small mb-1"),
        html.H4(val, className=f"card-title text-{color}")
    ]), className="text-center"), width=2) for lbl, val, color in kpi_vals]

    # Hourly volume
    h = d.groupby('arrival_hour').size().reset_index(name='count')
    fig_h = px.bar(h, x='arrival_hour', y='count', title='Hourly Arrivals',
                   color='count', color_continuous_scale='Blues',
                   labels={'arrival_hour':'Hour','count':'Arrivals'})
    fig_h.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=30))

    # Breach rate by triage
    bt = d.groupby('triage_level').agg(b=('breach','sum'), t=('visit_id','count')).reset_index()
    bt['pct'] = (bt['b']/bt['t']*100).round(1)
    fig_b = px.bar(bt, x='triage_level', y='pct', title='Breach Rate by Triage (%)',
                   color='pct', color_continuous_scale=[[0,'#1D9E75'],[0.5,'#BA7517'],[1,'#E24B4A']],
                   labels={'triage_level':'ESI','pct':'Breach %'})
    fig_b.add_hline(y=50, line_dash='dash', line_color='gray')
    fig_b.update_layout(coloraxis_showscale=False, margin=dict(t=40,b=30))

    # LOS by department
    ld = d.groupby('department')['length_of_stay_hrs'].mean().reset_index().sort_values('length_of_stay_hrs')
    fig_l = px.bar(ld, x='length_of_stay_hrs', y='department', orientation='h',
                   title='Avg LOS by Department (hrs)', color='length_of_stay_hrs',
                   color_continuous_scale='Purples',
                   labels={'length_of_stay_hrs':'Hours','department':''})
    fig_l.update_layout(coloraxis_showscale=False, margin=dict(t=40,l=120,b=30))

    # Revenue by insurance
    rv = d.groupby('insurance_type')['billed_amount_usd'].sum().reset_index()
    rv.columns = ['insurance','total']
    fig_r = px.bar(rv.sort_values('total', ascending=False), x='insurance', y='total',
                   title='Total Revenue by Insurance', color='insurance',
                   labels={'insurance':'Insurance','total':'Total Billed ($)'},
                   color_discrete_sequence=['#185FA5','#534AB7','#1D9E75','#BA7517','#E24B4A'])
    fig_r.update_layout(showlegend=False, margin=dict(t=40,b=30))

    # Satisfaction vs wait
    samp = d.sample(min(len(d),2000), random_state=42)
    r_val = d['wait_time_doctor_min'].corr(d['satisfaction_score'])
    fig_s = px.scatter(samp, x='wait_time_doctor_min', y='satisfaction_score',
                       color='triage_level', opacity=0.4, trendline='ols',
                       title=f'Satisfaction vs Doctor Wait  (r = {r_val:.2f})',
                       color_continuous_scale='RdBu_r',
                       labels={'wait_time_doctor_min':'Doctor Wait (min)',
                               'satisfaction_score':'Satisfaction (1–10)',
                               'triage_level':'Triage'})
    fig_s.update_layout(margin=dict(t=40,b=40))

    return cards, fig_h, fig_b, fig_l, fig_r, fig_s


if __name__ == '__main__':
    print("Open your browser → http://127.0.0.1:8050")
    app.run(debug=True)