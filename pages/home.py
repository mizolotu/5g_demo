import os, json
import os.path as osp
import numpy as np

import dash_bootstrap_components as dbc
import plotly.express as px

from dash import html, Input, State, Output, dcc, register_page, callback, ALL, callback_context

register_page(__name__, path='/')

layout = html.Div(className='full', children=[

    dbc.Row(className='padded', justify='start', style={'height': '20%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 7}, children=[

            html.H3('Introduction'),

            dcc.Markdown(
                '''
                As artificial intelligence (AI) and machine learning (ML) become a core part of almost every industry, including 5G, 
                there is an increasing concern about the vulnerability of AI/ML to adversarial effects.
                One such vulnerability is related to adversarial example generation during the inference time. 
                An adversary aims to feed such features to the target model that it returns certain wrong output.
                '''
            ),

            dcc.Markdown(
                '''
                In our research project, we focus on adversarial example attacks that may take place in the inference stage in one of the 
                AI/ML-based components of future 5G networks.
                '''
            )

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 5}, children=[

            html.Img(
                src='assets/pig.png', style={'height': '100%'}

            )

        ])

    ]),

    dbc.Row(className='padded', justify='start', style={'height': '50%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 5}, children=[

            html.Img(
                src='assets/5g_attack.png', style={'height': '100%'}
            )

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 7}, children=[

            dbc.Row(className='padded', justify='start', style={'height': '50%'}, children=[

                dbc.Col(align='center', style={'height': '100%'}, width={"size": 12}, children=[

                    html.H3('Use case'),

                    dcc.Markdown(
                        '''
                        In the radio access network (RAN) domain, due to the nature of wireless medium, AI/ML based frameworks deployed 
                        may be susceptible to adversaries that manipulate the inputs to the models during the inference stage over the air. 
                        Unless there is a serious flaw in the system, the adversary is not able to manipulate the input data to the target model directly, 
                        however, it can only add its own transmissions on top of existing transmissions over the air to change the input data indirectly.
                        A deep learning solution for fast and accurate initial access (IA) in 5G mmWave networks is proposed by [Cousik et al](https://arxiv.org/abs/2101.01847).
                        The IA time consists of two components: time for beam sweeping, i.e. measuring the received signal strengths (RSSs) for different beams, and time for beam prediction, 
                        i.e. identifying the beam for a given transmitter-receiver pair to communicate with. Since the beam sweep time dominates the overall IA time, it is essential to 
                        improve the IA time by utilizing fewer beams. 
                        '''
                    ),

                ])

            ]),

            dbc.Row(className='padded', justify='start', style={'height': '50%'}, children=[

                dbc.Col(align='center', style={'height': '100%'}, width={"size": 6}, children=[

                    dcc.Markdown(
                        '''                         
                        The study attempts to reduce the beam sweep time by measuring RSSs from only a subset of all available beams and 
                        mapping them to the best selection from the entire set of beams. An adversary may search for a perturbation that causes any misclassification at 
                        the receiver’s classifier. In case the beam selected by the classifier is different from the best one, the network throughput may be reduced for 
                        the user under attack. The attack resembles radio jamming, i.e. an adversary uses a fake base station to emit a radio signal on top of existing 
                        transmissions over the air to negatively affect the model output, i.e. the antenna beam selected.
                        '''
                    ),

                ]),

                dbc.Col(align='center', style={'height': '100%'}, width={"size": 6}, children=[

                    html.Img(
                        src='assets/attack_schema.png', style={'height': '100%'}
                    )

                ])

            ])

        ]),

    ]),

    dbc.Row(className='padded', justify='start', style={'height': '20%'}, children=[

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 6}, children=[

            html.H3('Simulations'),

            dcc.Markdown(
                '''
                We tested the attack using Outdoor scenario O1 from [DeepMIMO](https://deepmimo.net/) simulator. One of the base stations is selected as the serving 
                base station. UEs are located in rows 700 - 1300, frequency is 28 GHz, BS antenna shape is (1, 64, 1), UE antenna shape is (1, 1, 1), subset of 16 (out of 64) 
                channels is used as the input. A fully-connected neural network consisting of 5 layers is used, each hidden layer in the network selected consists 
                of 2048 neurons. Size of the output layer is 64 which corresponds to the number of antennas at the BS, the input layer has shape (2, 16, 32), i.e.
                it can be viewed as an image and various adversarial example generation attacks can be applied to cause musclassification at the target model.                    
                '''
            ),

            dbc.Button(
                'Go to demo',
                href='demo'
            )

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 3}, children=[

            html.Img(
                src='assets/scenario_bird.png', style={'height': '100%'}
            )

        ]),

        dbc.Col(align='center', style={'height': '100%'}, width={"size": 3}, children=[

            html.Img(src='assets/input_r.png', style={'height': '100%'}),
            html.Img(src='assets/input_i.png', style={'height': '100%'}),
            html.Img(src='assets/output.png', style={'height': '100%'}),

        ]),

    ]),

])