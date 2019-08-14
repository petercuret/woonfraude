###############################################################################
# Dit script implementeert een dashboard-applicatie voor het efficiënt plannen
# van het handhavingsproces op basis van woonfraude-meldingen binnen
# de gemeente van Amsterdam.
#
# Thomas Jongstra & Swaan Dekkers 2019
#
#
# Basic intro on working with Dash: https://dash.plot.ly/getting-started
#
# Example dashboards using maps in Dash (from dash-gallery.plotly.host/Portal):
# github.com/plotly/dash-sample-apps/blob/master/apps/dash-oil-and-gas/app.py
# github.com/plotly/dash-oil-gas-ternary
#
#
#
# This application took some inspiration from this video:
# https://www.youtube.com/watch?v=lu0PtsMor4E
#
# Inspiration has also been taken from the corresponding codebase:
# https://github.com/amyoshino/Dash_Tutorial_Series  (full of errors etc!!)
###############################################################################


###############################################################################
# TODO #
########
#
# - Eigen legenda bouwen (ingebouwde legenda breekt eigen custom point selection functionality)
# - Deploy code on VAO.
# - Check RandomForestRegressor confidence precision:
#     Is there a correlation between high confidence and true positives?
#
###############################################################################


###############################################################################
# Import public modules.
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
import dash_table.FormatTemplate as FormatTemplate
from dash.dependencies import Input, Output, State, ClientsideFunction

import pandas as pd
import urllib
import json
import re
import q
from copy import deepcopy

import plotly.graph_objs as go

# Import own modules.
import config
###############################################################################


###############################################################################
# Load mock-up data for prototyping purposes.
df = pd.read_csv('mockup_dataset.csv', sep=';', skipinitialspace=True)
df_proactief = pd.read_csv('mockup_dataset_proactief.csv', sep=';', skipinitialspace=True)
df_unsupervised = pd.read_csv('mockup_dataset_unsupervised.csv', sep=';', skipinitialspace=True)
###############################################################################


###############################################################################
# Define site visuals.
colors = {'paper': '#DDDDDD',
          'background': '#F2F2F2',
          'container_background': '#F9F9F9',
          'text': '#1E4363',
          'marker': '#1E4363',
          'fraud': 'rgb(200, 50, 50)',
          'no_fraud': 'rgb(150, 150, 150)',
          'selected': 'rgb(75, 75, 75)',
          }
###############################################################################


###############################################################################
#############################
# Set some global variables #
#############################

# Get dictionary of columns for DataTable.
SELECTED_COLUMNS = ['fraude_kans', 'woonfraude', 'adres_id', 'stadsdeel', 'categorie', 'eigenaar']
TABLE_COLUMNS = [{'name': i, 'id': i} for i in SELECTED_COLUMNS]

# Define styling for the first column (fraude_kans), to reduce the decimals after comma.
TABLE_COLUMNS[0]['name'] = 'Fraude kans (%)'
TABLE_COLUMNS[0]['type'] = 'numeric'
TABLE_COLUMNS[0]['format'] = FormatTemplate.percentage(2)
###############################################################################


###############################################################################
# Define the dashboard.
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash.Dash(__name__)
server = app.server
app.title = 'Woonfraude Dashboard'

# Defines the meldingen tab.
meldingen_tab = html.Div(
    [

        # Div containing a selection of the data based on dropdown selection.
        html.Div(id='intermediate_value', style={'display': 'none'}),

        # Divs contain a lists of points which have been selected with on-clicks on the map.
        html.Div(id='point_selection', style={'display': 'none'}),
        html.Div(id='filtered_point_selection', style={'display': 'none'}),

        # Row containing filters, info boxes, and map.
        html.Div(
            [
                # Filters div.
                html.Div(
                    [

                        # Create drop down filter for categories.
                        html.P('Selecteer categorieën:', className="control_label"),
                        dcc.Dropdown(
                            id='categorie_dropdown',
                            placeholder='Selecteer categorieën',
                            options=[{'label': x, 'value': x} for x in sorted(df.categorie.unique())],
                            multi=True,
                            value=df.categorie.unique(),
                        ),

                        # Create drop down filter for city parts.
                        html.P('Selecteer stadsdelen:', className="control_label"),
                        dcc.Dropdown(
                            id='stadsdeel_dropdown',
                            placeholder='Selecteer stadsdelen',
                            options=[{'label': x, 'value': x} for x in sorted(df.stadsdeel.unique())],
                            multi=True,
                            value=sorted(df.stadsdeel.unique()),
                            ),

                        # Show info of items selected on map (using click).
                        html.Div(
                            [
                                html.P('Geselecteerde adressen:', className="control_label"),
                                dt.DataTable(
                                    id='filtered_point_selection_table',
                                    columns = TABLE_COLUMNS[1:-1],
                                    sort_action='native',
                                    sort_by=[{'column_id': 'fraude_kans', 'direction': 'desc'}],
                                    page_action='native',
                                    page_current=0,
                                    page_size=20,
                                    style_data_conditional=[
                                        {
                                            'if': {
                                                'column_id': 'woonfraude',
                                                'filter_query': '{woonfraude} eq True'
                                            },
                                            'backgroundColor': colors['fraud'],
                                        },
                                        {
                                            'if': {
                                                'column_id': 'woonfraude',
                                                'filter_query': '{woonfraude} eq False'
                                            },
                                            'backgroundColor': colors['no_fraud'],
                                        },
                                        {
                                            'if': {
                                                'column_id': 'fraude_kans',
                                                'filter_query': '{fraude_kans} ge 0.5'
                                            },
                                            'backgroundColor': colors['fraud'],
                                        },
                                        {
                                            'if': {
                                                'column_id': 'fraude_kans',
                                                'filter_query': '{fraude_kans} lt 0.5'
                                            },
                                            'backgroundColor': colors['no_fraud'],
                                        },
                                    ]
                                ),
                            ],
                        ),

                        # Link to download csv with all selected addresses.
                        html.A(
                            'Download lijst geselecteerde adressen (CSV)',
                            id='download_selected_addresses_list',
                            download="geselecteerde_adressen.csv",
                            href="",
                            target="_blank",
                        ),

                    ],
                    id='leftCol',
                    className="pretty_container four columns",
                ),

                # Widgets and map div.
                html.Div(
                    [

                        # # Row with 4 statistics widgets
                        # html.Div(
                        #     [

                        #         # Aantal meldingen (info box).
                        #         html.Div(
                        #             [
                        #                 html.P("Aantal meldingen"),
                        #                 html.H6(
                        #                     id="aantal_meldingen",
                        #                     className="info_text"
                        #                 )
                        #             ],
                        #             className="pretty_container"
                        #         ),

                        #         # Percentage fraude verwacht (info box).
                        #         html.Div(
                        #             [
                        #                 html.P("% Fraude verwacht"),
                        #                 html.H6(
                        #                     id="percentage_fraude_verwacht",
                        #                     className="info_text"
                        #                 )
                        #             ],
                        #             className="pretty_container"
                        #         ),

                        #         # Aantal geselecteerde meldingen (info box).
                        #         html.Div(
                        #             [
                        #                 html.P("Aantal geselecteerde meldingen"),
                        #                 html.H6(
                        #                     id="aantal_geselecteerde_meldingen",
                        #                     className="info_text"
                        #                 )
                        #             ],
                        #             className="pretty_container",
                        #             style={'backgroundColor': '#F7D7D7'}
                        #         ),

                        #         # Percentage fraude verwacht bij geselecteerde meldingen (info box).
                        #         html.Div(
                        #             [
                        #                 html.P("% Fraude verwacht bij geselecteerde meldingen"),
                        #                 html.H6(
                        #                     id="percentage_fraude_verwacht_geselecteerd",
                        #                     className="info_text"
                        #                 )
                        #             ],
                        #             className="pretty_container",
                        #             style={'backgroundColor': '#F7D7D7'}
                        #         ),

                        #     ],
                        #     id="infoContainer",
                        #     className="row"
                        # ),


                        # Map with selectable points.
                        html.Div(
                            dcc.Graph(
                                id='map',
                                config={'displayModeBar': False},  # Turned off to disable selection with box/lasso etc.
                            ),
                            className="pretty_container",
                            # style={'height': 500}
                        ),
                    ],
                    id="rightCol",
                    className="eight columns"
                ),

            ],
            className="row",
        ),

        # Data table div.
        html.Div(
            [
                # Filtered entries data table.
                html.Div(
                    [
                        html.P('Gefilterde meldingen'),
                        dt.DataTable(
                            id='filtered_table',
                            columns = TABLE_COLUMNS,
                            sort_action='native',
                            sort_by=[{'column_id': 'fraude_kans', 'direction': 'desc'}],
                            # filter_action='native',  # Maybe turn off? A text field to filter feels clunky..
                            # row_selectable='multi',
                            # selected_rows=[],
                            page_action='native',
                            page_current=0,
                            page_size=20,
                            style_data_conditional=[
                                {
                                    'if': {
                                        'column_id': 'woonfraude',
                                        'filter_query': '{woonfraude} eq True'
                                    },
                                    'backgroundColor': colors['fraud'],
                                },
                                {
                                    'if': {
                                        'column_id': 'woonfraude',
                                        'filter_query': '{woonfraude} eq False'
                                    },
                                    'backgroundColor': colors['no_fraud'],
                                },
                                {
                                    'if': {
                                        'column_id': 'fraude_kans',
                                        'filter_query': '{fraude_kans} ge 0.5'
                                    },
                                    'backgroundColor': colors['fraud'],
                                },
                                {
                                    'if': {
                                        'column_id': 'fraude_kans',
                                        'filter_query': '{fraude_kans} lt 0.5'
                                    },
                                    'backgroundColor': colors['no_fraud'],
                                }
                            ]
                        ),

                    ],
                    className="pretty_container eight columns",
                ),

                # Filtered entries stadsdeel split (pie chart).
                html.Div(
                    [
                        html.P("Gefilterde meldingen - Stadsdeel split"),
                        dcc.Graph(
                            id="stadsdeel_split",
                            config={'displayModeBar': False},
                        )
                    ],
                    id="stadsdeel",
                    className="pretty_container two columns"
                ),

                # Filtered entries categorie split (pie chart).
                html.Div(
                    [
                        html.P("Gefilterde meldingen - Categorie split"),
                        dcc.Graph(
                            id="categorie_split",
                            config={'displayModeBar': False},
                        )
                    ],
                    id="categorie",
                    className="pretty_container two columns"
                ),

            ],
            className="row"
        ),

    ],
    id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    }

)


# Defines the proactief tab.
proactief_tab = html.Div(
    [
        # For creating a map_proactief callback function with an empty input.
        html.Div(id='none_proactief',children=[],style={'display': 'none'}),

        # Div for containing a selection of the data based on filters.
        html.Div(id='intermediate_value_proactief', style={'display': 'none'}),

        # Row containing filters, info boxes, and map.
        html.Div(
            [
                # Filters div.
                html.Div(
                    [

                        # Create range slider for number of meldingen.
                        html.P('Minimaal aantal meldingen op adres:', className="control_label"),
                        dcc.RangeSlider(
                            id='aantal_meldingen_rangeslider_proactief',
                            min=min(df_proactief.aantal_meldingen),
                            max=max(df_proactief.aantal_meldingen),
                            marks={i: f"{i}" for i in range(min(df_proactief.aantal_meldingen), max(df_proactief.aantal_meldingen)+1)},
                            value=[min(df_proactief.aantal_meldingen), max(df_proactief.aantal_meldingen)]
                        ),

                        # Padding (temporary hack)
                        html.P(' '),

                        # Create slider for number of adults.
                        html.P('Aantal volwassenen', className="control_label"),
                        dcc.RangeSlider(
                            id='aantal_volwassenen_rangeslider_proactief',
                            min=min(df_proactief.aantal_volwassenen),
                            max=max(df_proactief.aantal_volwassenen),
                            marks={i: f"{i}" for i in range(min(df_proactief.aantal_volwassenen), max(df_proactief.aantal_volwassenen)+1)},
                            value=[min(df_proactief.aantal_volwassenen), max(df_proactief.aantal_volwassenen)]
                        ),

                        # Padding (temporary hack)
                        html.P(' '),

                        # Create m2 per person slider.
                        html.P('Aantal m2 per persoon:', className="control_label"),
                        dcc.RangeSlider(
                            id='aantal_m2_per_persoon_rangeslider_proactief',
                            min=min(df_proactief.m2_per_persoon),
                            max=max(df_proactief.m2_per_persoon),
                            marks={i: f"{i}" for i in range(min(df_proactief.m2_per_persoon), max(df_proactief.m2_per_persoon)+1, 3)},
                            value=[min(df_proactief.m2_per_persoon), max(df_proactief.m2_per_persoon)]
                        ),

                        # Padding (temporary hack)
                        html.P(' '),

                        # Create drop down filter for city parts.
                        html.P('Selecteer stadsdelen:', className="control_label"),
                        dcc.Dropdown(
                            id='stadsdeel_dropdown_proactief',
                            placeholder='Selecteer stadsdelen',
                            options=[{'label': x, 'value': x} for x in sorted(df_proactief.stadsdeel.unique())],
                            multi=True,
                            value=sorted(df_proactief.stadsdeel.unique()),
                        ),

                        # Create hotline dropdown.
                        html.P('Is hotline melding:', className="control_label"),
                        dcc.Dropdown(
                            id='hotline_dropdown_proactief',
                            placeholder='Selecteer waarden',
                            options=[{'label': 'Ja', 'value': 'True'}, {'label': 'Nee', 'value': 'False'}],
                            multi=True,
                            value=['True', 'False']
                        ),

                        # Create gebruikersdoel dropdown.
                        html.P('Selecteer gebruikersdoel:', className="control_label"),
                        dcc.Dropdown(
                            id='gebruikersdoel_dropdown_proactief',
                            placeholder='Selecteer gebruikersdoel',
                            options=[{'label': x, 'value': x} for x in sorted(df_proactief.gebruikersdoel.unique())],
                            multi=True,
                            value=sorted(df_proactief.gebruikersdoel.unique()),
                        ),

                        # Create profiel dropdown.
                        html.P('Selecteer profiel:', className="control_label"),
                        dcc.Dropdown(
                            id='profiel_dropdown_proactief',
                            placeholder='Selecteer profiel',
                            options=[{'label': x, 'value': x} for x in sorted(df_proactief.profiel.unique())],
                            multi=True,
                            value=sorted(df_proactief.profiel.unique()),
                        ),

                    ],
                    id='leftCol_proactief',
                    className="pretty_container four columns",
                ),

                # Map div.
                html.Div(
                    [

                        # Map with selectable points.
                        html.Div(
                            dcc.Graph(
                                id='map_proactief',
                                config={'displayModeBar': False},  # Turned off to disable selection with box/lasso etc.
                            ),
                            className="pretty_container",
                            # style={'height': 500}
                        ),
                    ],
                    id="rightCol_proactief",
                    className="eight columns"
                ),

            ],
            className="row",
        ),


        # Data table div.
        html.Div(
            [
                # Filtered entries data table.
                html.Div(
                    [
                        html.P('Gefilterde meldingen'),
                        dt.DataTable(
                            id='filtered_table_proactief',
                            columns = TABLE_COLUMNS,
                            sort_action='native',
                            sort_by=[{'column_id': 'fraude_kans', 'direction': 'desc'}],
                            # filter_action='native',  # Maybe turn off? A text field to filter feels clunky..
                            # row_selectable='multi',
                            # selected_rows=[],
                            page_action='native',
                            page_current=0,
                            page_size=20,
                            style_data_conditional=[
                                {
                                    'if': {
                                        'column_id': 'woonfraude',
                                        'filter_query': '{woonfraude} eq True'
                                    },
                                    'backgroundColor': colors['fraud'],
                                },
                                {
                                    'if': {
                                        'column_id': 'woonfraude',
                                        'filter_query': '{woonfraude} eq False'
                                    },
                                    'backgroundColor': colors['no_fraud'],
                                },
                                {
                                    'if': {
                                        'column_id': 'fraude_kans',
                                        'filter_query': '{fraude_kans} ge 0.5'
                                    },
                                    'backgroundColor': colors['fraud'],
                                },
                                {
                                    'if': {
                                        'column_id': 'fraude_kans',
                                        'filter_query': '{fraude_kans} lt 0.5'
                                    },
                                    'backgroundColor': colors['no_fraud'],
                                }
                            ]
                        ),

                    ],
                    className="pretty_container ten columns",
                ),

                # Filtered entries stadsdeel split (pie chart).
                html.Div(
                    [
                        html.P("Gefilterde meldingen - Stadsdeel split"),
                        dcc.Graph(
                            id="stadsdeel_split_proactief",
                            config={'displayModeBar': False},
                        )
                    ],
                    id="stadsdeel_proactief",
                    className="pretty_container two columns"
                ),

            ],
            className="row"
        ),




        # html.Div(
        #     dcc.Graph(
        #         id='map_proactief',
        #         config={'displayModeBar': False},  # Turned off to disable selection with box/lasso etc.
        #     ),
        #     className="pretty_container",
        # ),

    ],
    style={
        "display": "flex",
        "flex-direction": "column"
    }
)


# Defines the unsupervised tab.
unsupervised_tab = html.Div(
    [
        # For creating a map_unsupervised callback function with an empty input.
        html.Div(id='none_unsupervised',children=[],style={'display': 'none'}),

        # Div for containing a selection of the data based on filters.
        html.Div(id='intermediate_value_unsupervised', style={'display': 'none'}),

        html.Div(
            dcc.Graph(
                id='map_unsupervised',
                config={'displayModeBar': False},  # Turned off to disable selection with box/lasso etc.
            ),
            className="pretty_container",
        ),
    ],
    style={
        "display": "flex",
        "flex-direction": "column"
    }
)



# Combines the two tabs into a single app.
app.layout = html.Div([

    # Title
    html.H1("Woonfraude Dashboard", style={'textAlign': 'center'}),

    # Tabs for meldingen & proactieve handhaving.
    dcc.Tabs(id='tabs', value='meldingen_tab', children=[
        dcc.Tab(label='Meldingen', value='meldingen_tab', children=[meldingen_tab]),
        dcc.Tab(label='Proactieve handhaving', value='proactief_tab', children=[proactief_tab]),
        dcc.Tab(label='Unsupervised', value='unsupervised_tab', children=[unsupervised_tab]),
    ])
])



# Updates the intermediate data based on the dropdown selection.
@app.callback(
    Output('intermediate_value', 'children'),
    [Input('categorie_dropdown', 'value'),
    Input('stadsdeel_dropdown', 'value')]
)
def create_data_selection(selected_categories, selected_stadsdelen):
    # Create a copy of the original dataframe.
    df_filtered = deepcopy(df)

    # Filter the original dataframe by selected categories.
    df_filtered = df_filtered[df_filtered.categorie.isin(selected_categories)]

    # Filter the dataframe by selected stadsdelen.
    df_filtered = df_filtered[df_filtered.stadsdeel.isin(selected_stadsdelen)]

    return df_filtered.to_json(date_format='iso', orient='split')

'''
# Updates the aantal_meldingen info box.
@app.callback(
    Output('aantal_meldingen', 'children'),
    [Input('intermediate_value', 'children')]
)
def count_items(intermediate_value):
    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')
    return len(df)


# Updates the percentage_fraude_verwacht info box.
@app.callback(
    Output('percentage_fraude_verwacht', 'children'),
    [Input('intermediate_value', 'children')]
)
def compute_fraud_percentage(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Compute what percentage of cases is expected to be fraudulent. If/else to prevent division by 0.
    if len(df.woonfraude) > 0:
        fraude_percentage = len(df.woonfraude[df.woonfraude == True]) / len(df.woonfraude) * 100
    else:
        fraude_percentage = 0

    # Return truncated value (better for printing on dashboard)
    return round(fraude_percentage, 1)


# Updates the aantal_geselecteerde_meldingen info box.
@app.callback(
    Output('aantal_geselecteerde_meldingen', 'children'),
    [Input('filtered_point_selection', 'children')]
)
def count_items_selected(filtered_point_selection):
    # Just return the amount of filtered selected points.
    return len(filtered_point_selection)


# Updates the percentage_fraude_verwacht_geselecteerd info box.
@app.callback(
    Output('percentage_fraude_verwacht_geselecteerd', 'children'),
    [Input('intermediate_value', 'children'),
    Input('filtered_point_selection', 'children')]
)
def compute_fraud_percentage_selected(intermediate_value, filtered_point_selection):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Reduce the dataframe using the point selection.
    df = df[df.adres_id.isin(filtered_point_selection)]

    # Compute what percentage of cases is expected to be fraudulent. If/else to prevent division by 0.
    if len(df.woonfraude) > 0:
        fraude_percentage = len(df.woonfraude[df.woonfraude == True]) / len(df.woonfraude) * 100
    else:
        fraude_percentage = 0

    # Return truncated value (better for printing on dashboard)
    return round(fraude_percentage, 1)
'''

# Updates the map based on dropdown-selections.
@app.callback(
    Output('map', 'figure'),
    [Input('intermediate_value', 'children'),
    Input('point_selection', 'children')],
    [State('map', 'figure')]
)
def plot_map(intermediate_value, point_selection, map_state):

    # Define which input triggers the callback (map.figure or intermediate_value.children).
    trigger_event = dash.callback_context.triggered[0]['prop_id']

    # Load the pre-filtered version of the dataframe.
    df_map = pd.read_json(intermediate_value, orient='split')

    # Select positive and negative samples for plotting.
    pos = df_map[df_map.woonfraude==True]
    neg = df_map[df_map.woonfraude==False]

    # Create a df of the selected points, for highlighting.
    selected_point_ids = [int(x) for x in point_selection]
    sel = df_map.loc[df_map.adres_id.isin(selected_point_ids)]

    # Create texts for when hovering the mouse over items.
    def make_hover_string(row):
        return f"Adres id: {row.adres_id}\
                 <br>Categorie: {row.categorie}\
                 <br>Aantal inwoners: {row.aantal_personen}\
                 <br>Aantal achternamen: {row.aantal_achternamen}\
                 <br>Eigenaar: {row.eigenaar}"
    pos_text = pos.apply(make_hover_string, axis=1)
    neg_text = neg.apply(make_hover_string, axis=1)
    sel_text = sel.apply(make_hover_string, axis=1)

    figure={
        'data': [
            # Plot border for selected samples (plot first, so its behind the pos/neg samples).
            go.Scattermapbox(
                name='Geselecteerd',
                lat=sel['wzs_lat'],
                lon=sel['wzs_lon'],
                text=sel_text,
                mode='markers',
                marker=dict(
                    size=17,
                    color=colors['selected'],
                ),
            ),
            # Plot positive samples.
            go.Scattermapbox(
                name='Woonfraude verwacht',
                lat=pos['wzs_lat'],
                lon=pos['wzs_lon'],
                text=pos_text,
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['fraud'],
                ),
            ),
            # Plot negative samples.
            go.Scattermapbox(
                name='Geen woonfraude verwacht',
                lat=neg['wzs_lat'],
                lon=neg['wzs_lon'],
                text=neg_text,
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['no_fraud'],
                ),
            ),
        ],
        'layout': go.Layout(
            uirevision='never',
            autosize=True,
            hovermode='closest',
            # width=1000,
            height=700,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            showlegend=False,  # Set to False, since legend selection breaks custom point selection.
            legend=dict(orientation='h'),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['paper'],
            mapbox=dict(
                accesstoken=config.mapbox_access_token,
                style="light",
                center=dict(
                    lat=52.36,
                    lon=4.89
                ),
                zoom=11,
            ),
        )
    }

    return figure


# Updates the table showing all data points after dropdown-selections.
@app.callback(
    Output('filtered_table', 'data'),
    [Input('intermediate_value', 'children')]
)
def generate_filtered_table(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df_table = pd.read_json(intermediate_value, orient='split')

    # Transform True and False boolean values to strings.
    df_table.woonfraude = df_table.woonfraude.replace({True: 'True', False: 'False'})

    # Only use a selection of the columns.
    df_table = df_table[SELECTED_COLUMNS]

    # Create a table, with all positive woonfraude examples at the top.
    columns = [{"name": i, "id": i} for i in df_table.columns]
    data = df_table.to_dict('records')

    return data


# Enable the selection of map points using click-events.
@app.callback(
    Output('point_selection', 'children'),
    [Input('map', 'clickData'),
    Input('intermediate_value', 'children')],
    [State('point_selection', 'children')]
)
def update_point_selection_on_click(clickData, intermediate_value, existing_point_selection):
    """
    Update point selection with newly selected points, or according to dropdown filters.

    The input "intermediate_value:children" is only used to activate a callback.
    """

    # Define which input triggers the callback (map.clickData or intermediate_value.children).
    trigger_event = dash.callback_context.triggered[0]['prop_id']

    # Re-use previous point selection (if it already existed).
    point_selection = []
    if existing_point_selection != None:
        point_selection = existing_point_selection

    # Add a clicked point to the selection, or remove it when it already existed in the selection.
    if trigger_event == 'map.clickData':
        if clickData != None:
            point_id = re.match("Adres id: (\d+)", clickData['points'][0]['text']).group(1)
            if point_id in point_selection:
                point_selection.remove(point_id)
            else:
                point_selection.append(point_id)

    return point_selection


# Create a filtered version of the point_selection, based on the categorie and stadsdeel filters.
@app.callback(
    Output('filtered_point_selection', 'children'),
    [Input('point_selection', 'children'),
    Input('intermediate_value', 'children')]
)
def show_selected(existing_point_selection, intermediate_value):
    # Re-use previous point selection (if it already existed).
    point_selection = []
    if existing_point_selection != None:
        point_selection = existing_point_selection

    # Filter any previously selected points, if the dropdown selections rule them out.
    df = pd.read_json(intermediate_value, orient='split') # Load the pre-filtered version of the dataframe.
    point_ids_list = [str(x) for x in list(df.adres_id)]
    for point_id in point_selection:
        if point_id not in point_ids_list:
            point_selection.remove(point_id)

    return point_selection


# Updates the table showing a list of the selected & filtered points.
@app.callback(
    Output('filtered_point_selection_table', 'data'),
    [Input('intermediate_value', 'children'),
    Input('filtered_point_selection', 'children')]
)
def generate_filtered_point_selection_table(intermediate_value, filtered_point_selection):

    # First check if any points have been selected.
    if filtered_point_selection == []:
        return []
    else:
        # Turn list of point_ids into a list of numbers instead of strings
        point_selection = [int(x) for x in filtered_point_selection]

        # Load the pre-filtered version of the dataframe.
        df = pd.read_json(intermediate_value, orient='split')

        # Reduce the dataframe using the point selection.
        df = df[df.adres_id.isin(point_selection)]

        # Transform True and False boolean values to strings.
        df.woonfraude = df.woonfraude.replace({True: 'True', False: 'False'})

        # Only use a selection of the columns.
        df = df[SELECTED_COLUMNS]

        # Create a table, with all positive woonfraude examples at the top.
        columns = [{"name": i, "id": i} for i in df.columns]
        data = df.to_dict('records')
        return data


# TODO: CHANGE WHEN THE DOWNLOAD LINK IS UPDATED WITH NEW DATA.
#       NOW THIS CODE BELOW IS RAN EVERY TIME A POINT IS (DE)SELECTED,
#       THIS IS TERRIBLY INEFFICIENT. ACCEPTABLE FOR THE MVP, BUT SHOULD BE CHANGED.

# Creates a download link for the filtered_point_selection_table data.
@app.callback(
    Output('download_selected_addresses_list', 'href'),
    [Input('filtered_point_selection_table', 'data')])
def update_download_link(filtered_point_selection_table):
    """Updates the csv download link with the data in the filtered point selection table."""
    if filtered_point_selection_table == []:
        point_selection = []
    else:
        # Turn list of point_ids into a list of numbers instead of strings
        point_selection = filtered_point_selection_table

    # Convert to df, then to csv string, then return for downloading.
    df = pd.DataFrame(point_selection)
    csv_string = df.to_csv(index=False, encoding='utf-8', sep=';')
    csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
    return csv_string


# Updates the stadsdeel split PIE chart.
@app.callback(
    Output('stadsdeel_split', 'figure'),
    [Input('intermediate_value', 'children')]
)
def make_stadsdeel_pie_chart(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Create value counts per stadsdeel.
    stadsdeel_value_counts = df.stadsdeel.value_counts().sort_index()

    figure={
        'data': [
            go.Pie(
                labels=stadsdeel_value_counts.index,
                values=stadsdeel_value_counts.values
            )
        ],
        'layout': go.Layout(
            height=300,
            margin=go.layout.Margin(l=0, r=0, b=100, t=0, pad=0),
            showlegend=True,
            legend=dict(orientation='h', font={'size':10}),
            paper_bgcolor=colors['container_background'],
        )
    }
    return figure


# Updates the categorie split pie chart.
@app.callback(
    Output('categorie_split', 'figure'),
    [Input('intermediate_value', 'children')]
)
def make_categorie_pie_chart(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Create value counts per categorie.
    categorie_value_counts = df.categorie.value_counts().sort_index()

    figure={
        'data': [
            go.Pie(
                labels=categorie_value_counts.index,
                values=categorie_value_counts.values
            )
        ],
        'layout': go.Layout(
            height=300,
            margin=go.layout.Margin(l=0, r=0, b=100, t=0, pad=0),
            showlegend=True,
            legend=dict(orientation='h', x=0, y=0, font={'size':10}),
            paper_bgcolor=colors['container_background'],
        )
    }
    return figure


'''
# Updates the stadsdeel split BAR chart.
@app.callback(
    Output('stadsdeel_split', 'figure'),
    [Input('intermediate_value', 'children')]
)
def make_stadsdeel_split_bar_chart(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Create value counts per stadsdeel.
    stadsdeel_value_counts = df.stadsdeel.value_counts(ascending=True)
    x=stadsdeel_value_counts.values
    y=stadsdeel_value_counts.index
    percentages = [(val/sum(x))*100 for val in x]

    # Create annotations for showing the percentages on top of the bars.
    annotations = []
    for num, p in enumerate(percentages):
        annotation = dict(xref='x1',
                          yref='y1',
                          x=0.5,
                          y=num,
                          text=f"{p}%",
                          showarrow=False,
                          align='left'
                      )
        annotations.append(annotation)

    figure={
        'data': [
            go.Bar(
                y=y,
                x=x,
                text=percentages,
                marker=dict(
                    color='rgba(50, 171, 96, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 96, 1.0)',
                        width=2),
                ),
                showlegend=False,
                orientation='h'
            )
        ],
        'layout': go.Layout(
            height=200,
            margin=go.layout.Margin(l=100, r=0, b=0, t=0, pad=0),
            paper_bgcolor=colors['container_background'],
            annotations=annotations
        )
    }
    return figure


# Updates the categorie split BAR chart.
@app.callback(
    Output('categorie_split', 'figure'),
    [Input('intermediate_value', 'children')]
)
def make_stadsdeel_split_bar_chart(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Create value counts per categorie.
    stadsdeel_value_counts = df.categorie.value_counts(ascending=True)
    x=stadsdeel_value_counts.values
    y=stadsdeel_value_counts.index
    percentages = [(val/sum(x))*100 for val in x]

    # Create annotations for showing the percentages on top of the bars.
    annotations = []
    for num, p in enumerate(percentages):
        annotation = dict(xref='x1',
                          yref='y1',
                          x=0.5,
                          y=num,
                          text=f"{p}%",
                          showarrow=False,
                          align='left'
                      )
        annotations.append(annotation)

    figure={
        'data': [
            go.Bar(
                y=y,
                x=x,
                text=percentages,
                marker=dict(
                    color='rgba(50, 171, 96, 0.6)',
                    line=dict(
                        color='rgba(50, 171, 96, 1.0)',
                        width=2),
                ),
                showlegend=False,
                orientation='h'
            )
        ],
        'layout': go.Layout(
            height=200,
            margin=go.layout.Margin(l=100, r=0, b=0, t=0, pad=0),
            paper_bgcolor=colors['container_background'],
            annotations=annotations
        )
    }
    return figure
'''


###############################################################################
# Proactief tab functies #
###########################

# @app.callback(
#     Output('intermediate_value_proactief', 'children'),
#     [Input('none_proactief', 'children')]
# )
# def create_data_selection(_):
#     return df_proactief.to_json(date_format='iso', orient='split')


# Updates the intermediate data based on the dropdown selection.
@app.callback(
    Output('intermediate_value_proactief', 'children'),
    [Input('aantal_meldingen_rangeslider_proactief', 'value'),
    Input('aantal_volwassenen_rangeslider_proactief', 'value'),
    Input('aantal_m2_per_persoon_rangeslider_proactief', 'value'),
    Input('stadsdeel_dropdown_proactief', 'value'),
    Input('hotline_dropdown_proactief', 'value'),
    Input('gebruikersdoel_dropdown_proactief', 'value'),
    Input('profiel_dropdown_proactief', 'value')]
)
def create_data_selection(aantal_meldingen_range, aantal_volwassenen,
                          aantal_m2_per_persoon, selected_stadsdelen, is_hotline,
                          selected_gebruikersdoelen, selected_profielen):

    # Create a copy of the original dataframe.
    df_filtered = deepcopy(df_proactief)

    # Filter the original dataframe by aantal meldingen.
    min_meldingen = aantal_meldingen_range[0]
    max_meldingen = aantal_meldingen_range[1]
    df_filtered = df_filtered[(df_filtered.aantal_meldingen >= min_meldingen) & (df_filtered.aantal_meldingen <= max_meldingen)]

    # Filter on number of adults
    min_adults = aantal_volwassenen[0]
    max_adults = aantal_volwassenen[1]
    df_filtered = df_filtered[(df_filtered.aantal_volwassenen >= min_adults) & (df_filtered.aantal_volwassenen <= max_adults)]

    # Filter on the amount of m2 per person.
    min_m2_pp = aantal_m2_per_persoon[0]
    max_m2_pp = aantal_m2_per_persoon[1]
    df_filtered = df_filtered[(df_filtered.m2_per_persoon >= min_m2_pp) & (df_filtered.m2_per_persoon <= max_m2_pp)]

    # Filter the dataframe by selected stadsdelen.
    df_filtered = df_filtered[df_filtered.stadsdeel.isin(selected_stadsdelen)]

    # Filter the dataframe based on whether the meldingen are hotline meldingen.
    # To do this, first convert the is_hotline values (strings) to booleans for matching.
    is_hotline = [True if x=='True' else x for x in is_hotline]
    is_hotline = [False if x=='False' else x for x in is_hotline]
    df_filtered = df_filtered[df_filtered.is_hotline.isin(is_hotline)]

    # Filter the dataframe by selected gebruikersdoelen.
    df_filtered = df_filtered[df_filtered.gebruikersdoel.isin(selected_gebruikersdoelen)]

    # Filter the dataframe by selected profiles.
    df_filtered = df_filtered[df_filtered.profiel.isin(selected_profielen)]

    return df_filtered.to_json(date_format='iso', orient='split')


@app.callback(
    Output('map_proactief', 'figure'),
    [Input('intermediate_value_proactief', 'children')]
)
def plot_map(intermediate_value_proactief):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value_proactief, orient='split')

    # Select positive and negative samples for plotting.
    pos = df[df.woonfraude==True]
    neg = df[df.woonfraude==False]

    # Create texts for when hovering the mouse over items.
    def make_hover_string(row):
        return f"Adres id: {row.adres_id}\
                 <br>Categorie: {row.categorie}\
                 <br>Aantal inwoners: {row.aantal_personen}\
                 <br>Aantal achternamen: {row.aantal_achternamen}\
                 <br>Eigenaar: {row.eigenaar}"
    pos_text = pos.apply(make_hover_string, axis=1)
    neg_text = neg.apply(make_hover_string, axis=1)

    figure={
        'data': [
            # Plot positive samples.
            go.Scattermapbox(
                name='Woonfraude verwacht',
                lat=pos['wzs_lat'],
                lon=pos['wzs_lon'],
                text=pos_text,
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['fraud'],
                ),
            ),
            # Plot negative samples.
            go.Scattermapbox(
                name='Geen woonfraude verwacht',
                lat=neg['wzs_lat'],
                lon=neg['wzs_lon'],
                text=neg_text,
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['no_fraud'],
                ),
            ),
        ],
        'layout': go.Layout(
            uirevision='never',
            autosize=True,
            hovermode='closest',
            # width=1000,
            height=700,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            showlegend=False,  # Set to False, since legend selection breaks custom point selection.
            legend=dict(orientation='h'),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['paper'],
            mapbox=dict(
                accesstoken=config.mapbox_access_token,
                style="light",
                center=dict(
                    lat=52.36,
                    lon=4.89
                ),
                zoom=11,
            ),
        )
    }

    return figure


# Updates the table showing all data points after dropdown-selections.
@app.callback(
    Output('filtered_table_proactief', 'data'),
    [Input('intermediate_value_proactief', 'children')]
)
def generate_filtered_table(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df_table = pd.read_json(intermediate_value, orient='split')

    # Transform True and False boolean values to strings.
    df_table.woonfraude = df_table.woonfraude.replace({True: 'True', False: 'False'})

    # Only use a selection of the columns.
    df_table = df_table[SELECTED_COLUMNS]

    # Create a table, with all positive woonfraude examples at the top.
    columns = [{"name": i, "id": i} for i in df_table.columns]
    data = df_table.to_dict('records')

    return data


# Updates the stadsdeel split PIE chart.
@app.callback(
    Output('stadsdeel_split_proactief', 'figure'),
    [Input('intermediate_value_proactief', 'children')]
)
def make_stadsdeel_pie_chart(intermediate_value):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value, orient='split')

    # Create value counts per stadsdeel.
    stadsdeel_value_counts = df.stadsdeel.value_counts().sort_index()

    figure={
        'data': [
            go.Pie(
                labels=stadsdeel_value_counts.index,
                values=stadsdeel_value_counts.values
            )
        ],
        'layout': go.Layout(
            height=300,
            margin=go.layout.Margin(l=0, r=0, b=100, t=0, pad=0),
            showlegend=True,
            legend=dict(orientation='h', font={'size':10}),
            paper_bgcolor=colors['container_background'],
        )
    }
    return figure


###############################################################################
# Unsupervised tab functies #
#############################

@app.callback(
    Output('intermediate_value_unsupervised', 'children'),
    [Input('none_unsupervised', 'children')]
)
def create_data_selection(_):
    return df_unsupervised.to_json(date_format='iso', orient='split')


@app.callback(
    Output('map_unsupervised', 'figure'),
    [Input('intermediate_value_unsupervised', 'children')]
)
def plot_map(intermediate_value_unsupervised):

    # Load the pre-filtered version of the dataframe.
    df = pd.read_json(intermediate_value_unsupervised, orient='split')

    # Select positive and negative samples for plotting.
    pos = df[df.woonfraude==True]
    neg = df[df.woonfraude==False]

    # Create texts for when hovering the mouse over items.
    def make_hover_string(row):
        return f"Adres id: {row.adres_id}\
                 <br>Categorie: {row.categorie}\
                 <br>Aantal inwoners: {row.aantal_personen}\
                 <br>Aantal achternamen: {row.aantal_achternamen}\
                 <br>Eigenaar: {row.eigenaar}"
    pos_text = pos.apply(make_hover_string, axis=1)
    neg_text = neg.apply(make_hover_string, axis=1)

    figure={
        'data': [
            # Plot positive samples.
            go.Scattermapbox(
                name='Woonfraude verwacht',
                lat=pos['wzs_lat'],
                lon=pos['wzs_lon'],
                text=pos_text,
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['fraud'],
                ),
            ),
            # Plot negative samples.
            go.Scattermapbox(
                name='Geen woonfraude verwacht',
                lat=neg['wzs_lat'],
                lon=neg['wzs_lon'],
                text=neg_text,
                hoverinfo='text',
                mode='markers',
                marker=dict(
                    size=12,
                    color=colors['no_fraud'],
                ),
            ),
        ],
        'layout': go.Layout(
            uirevision='never',
            autosize=True,
            hovermode='closest',
            # width=1000,
            height=700,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            showlegend=False,  # Set to False, since legend selection breaks custom point selection.
            legend=dict(orientation='h'),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['paper'],
            mapbox=dict(
                accesstoken=config.mapbox_access_token,
                style="light",
                center=dict(
                    lat=52.36,
                    lon=4.89
                ),
                zoom=11,
            ),
        )
    }

    return figure


###############################################################################
# Run dashboard when calling this script.
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
###############################################################################