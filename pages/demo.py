import os.path as osp
import numpy as np
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import tensorflow as tf
import plotly.graph_objects as go
import plotly.express as px

from dash import html, Input, State, Output, dcc, register_page, callback, ALL, callback_context
from model import read_data, get_codebook, load_data, calculate_throughput
from odl import Odl
from collections import deque

register_page(__name__)

node_id = 'openflow:8796752125764'
table_id = 0
priority = 1000
q = 99

odl = Odl('127.0.0.1')


X, Y, C = read_data('baseline.csv')
n_antennas = Y.shape[1]
codebook = get_codebook(n_antennas)

model_fpath = 'model'
model = tf.keras.models.load_model(model_fpath)

data_dpath = 'data'
data, _, _ = load_data()

attack_options = ['baseline', 'random', 'white-box', 'black-box']
eps_options = [0.01, 0.1, 1.0]

attack_value = 'baseline'
eps_value = 0.01

thr = 100
beam_true = 0
beam_pred = 0

thr_queue = deque(maxlen=100)
mean_thr_queue = deque(maxlen=100)

layout = html.Div(className='full black', children=[

    dbc.Row(className='padded white', justify='start', style={'height': '50%'}, children=[

        dbc.Col(align='center', width={"size": 2}, children=[
            dbc.Label("Choose adversarial perturbation type:"),
            dbc.RadioItems(
                className='padded',
                options=[
                    {"label": x.capitalize(), "value": x} for x in attack_options
                ],
                value=attack_value,
                id='select_attack',
            ),
            dbc.Label("Choose adversarial perturbation size:"),
            dbc.RadioItems(
                className='padded',
                options=[
                    {"label": f'{x * 100}% of the user signal', "value": x} for x in eps_options
                ],
                value=eps_value,
                id='select_eps',
            )
        ]),

        dbc.Col(align='center', width={"size": 6}, children=[

            html.Div("Throughput", style={'text-align': 'center'}),
            dcc.Graph(
                className='padded',
                id='throughput',
                figure={},
                #style={'height': '50vh', 'width': '50vh'}
            )

        ]),

        dbc.Col(align='center', width={"size": 4}, children=[

            html.Iframe(
                width='100%',
                height="500",
                src = "https://www.youtube.com/embed/GE_SfNVNyqk?autoplay=1&controls=0"
            ),

        ])

    ]),

    dbc.Row(className='padded white', justify='start', style={'height': '50%'}, children=[

        dbc.Col(align='center', width={"size": 4}, children=[

            html.Div("Input signal", style={'text-align': 'center'}),
            dcc.Graph(
                className='padded',
                id='input',
                figure={},
            )

        ]),

        dbc.Col(align='center', width={"size": 4}, children=[

            html.Div("Adversarial perturbation", style={'text-align': 'center'}),
            dcc.Graph(
                className='padded',
                id='perturbation',
                figure={},
            )
        ]),

        dbc.Col(align='center', width={"size": 4}, children=[

            html.Div("Beam direction", style={'text-align': 'center'}),
            dcc.Graph(
                className='padded',
                id='output',
                figure={},
            )
        ])

    ]),

    html.Div(id='placeholder', children=[]),

    dcc.Interval(id="interval", n_intervals=0, interval=1000)

])

@callback(
    Output("throughput", "figure"),
    Output("input", "figure"),
    Output("perturbation", "figure"),
    Output("output", "figure"),
    Input("interval", "n_intervals"),
    State("select_attack", "value"),
    State("select_eps", "value"),
)
def update(n_intervals, attack, eps):

    idx = np.random.choice(X.shape[0])
    x = data[attack][eps][idx:idx + 1, :]

    baseline = data['baseline'][eps][idx, :]
    delta = x[0] - baseline

    fig_b = go.Figure(data=[
        go.Surface(z=baseline[:, :, 0]),
        go.Surface(z=baseline[:, :, 1] + 3*np.max(X), showscale=False),
    ])
    fig_b.update_traces(showscale=False)
    fig_b.update_layout(template='none')
    fig_b.update_layout(
        scene=dict(
            xaxis=dict(title='Subcarriers'), yaxis=dict(title='Antennas'),
            zaxis=dict(title='Signal strength', range=[np.min(X), 4*np.max(X)])
        ),
        margin={'l': 0, 'b': 20, 'r': 0, 't': 0}
    )

    fig_d = go.Figure(data=[
        go.Surface(z=delta[:, :, 0]),
        go.Surface(z=delta[:, :, 1] + 3 * np.max(X), showscale=False),
    ])
    fig_d.update_traces(showscale=False)
    fig_d.update_layout(template='none')
    fig_d.update_layout(
        scene=dict(
            xaxis=dict(title='Subcarriers'), yaxis=dict(title='Antennas'),
            zaxis=dict(title='Signal strength', range=[np.min(X), 4 * np.max(X)])
        ),
        margin={'l': 0, 'b': 20, 'r': 0, 't': 0}
    )

    p = model.predict(x, verbose=0)
    y = Y[idx, :]
    c = C[idx, :]
    thr, beam_true, beam_pred = calculate_throughput(x[0], y, p, c, codebook)

    thr_queue.append(thr)
    mean_thr_queue.append(np.mean(thr_queue))

    flow_ids = odl.find_config_flows(node_id, table_id)
    for flow_id in flow_ids:
        odl.delete_config_flow(node_id, table_id, flow_id)

    flow_ids = odl.find_operational_flows(node_id, table_id)
    for flow_id in flow_ids:
        odl.delete_operational_flow(node_id, table_id, flow_id)

    for port in [1, 2]:
        odl.in_port_set_queue_and_output_normal(node_id, table_id, priority, port, np.clip(int(thr), 0, 99))

    thr_vals = np.array(list(thr_queue))
    mean_thr_vals = np.array(list(mean_thr_queue))

    #fig_t = px.line(mean_thr_vals, markers=True)
    #fig_t.add_trace(go.Bar(y=thr_vals))
    #fig_t.add_trace(go.Scatter(y=mean_thr_vals))
    fig_t = px.bar(thr_vals, color=thr_vals, color_continuous_scale='RdBu')
    fig_t.add_trace(go.Scatter(y=mean_thr_vals, mode='lines+markers'))
    #fig_t.add_trace(px.bar(thr_vals))
    fig_t.update_layout(template='none', showlegend=False)
    fig_t.update_layout(xaxis=dict(range=[0, thr_queue.maxlen]))
    fig_t.update_layout(xaxis_title=f'Last {thr_queue.maxlen} measurements', yaxis_title='Throughput (% of the baseline)', coloraxis_colorbar=dict(title='Throughput'))

    ids = []
    for i in range(n_antennas):
        if i == beam_true:
            ids.append('Real')
        elif i == beam_pred:
            ids.append('Predicted')
        else:
            ids.append('Other')
    df = pd.DataFrame(np.vstack([np.ones(n_antennas), np.arange(0, 360, 360/n_antennas), ids]).T, columns=['r', 'theta', 'id'], index=None)
    fig_o = px.bar_polar(df, r='r', theta='theta', color=ids)
    if beam_pred == beam_true:
        names = {'Real': 'Real / predicted', 'Other': 'Other'}
        for i in range(len(fig_o.data)):
            fig_o.data[i].name = names[fig_o.data[i].name]
    fig_o.update_layout(template='none', legend_title_text='Beam')

    return fig_t, fig_b, fig_d, fig_o