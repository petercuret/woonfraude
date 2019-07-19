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
import re
import q
from copy import deepcopy

import plotly.plotly as py
import plotly.graph_objs as go

# Import own modules.
import config
###############################################################################


###############################################################################
# Load mock-up data for prototyping purposes.
df = pd.read_csv('mockup_dataset.csv', sep=';', skipinitialspace=True)
###############################################################################


###############################################################################
# Define site visuals.
colors = {'paper': '#DDDDDD',
          'background': '#FCF2CB',
          'text': '#1E4363',
          'marker': '#1E4363',
          'fraud': 'rgb(175, 0, 0)',
          'no_fraud': 'rgb(80, 80, 80)',
          'selected': 'rgb(50, 150, 50)',
          }
###############################################################################


###############################################################################
# Helper functions.
def filter_df(df, selected_categories, selected_stadsdelen):
    # Create a copy of the original dataframe.
    df_filtered = deepcopy(df)

    # Filter the original dataframe by selected categories.
    df_filtered = df_filtered[df_filtered.categorie.isin(selected_categories)]

    # Filter the dataframe by selected stadsdelen.
    df_filtered = df_filtered[df_filtered.stadsdeel.isin(selected_stadsdelen)]

    return df_filtered

# Get dictionary of columns for DataTable.
SELECTED_COLUMNS = ['float_test', 'woonfraude', 'adres_id', 'categorie', 'eigenaar']
TABLE_COLUMNS = [{'name': i, 'id': i} for i in SELECTED_COLUMNS]

# Define styling for the first column (float_test), to reduce the decimals after comma.
TABLE_COLUMNS[0]['name'] = 'Fraude kans (%)'
TABLE_COLUMNS[0]['type'] = 'numeric'
TABLE_COLUMNS[0]['format'] = FormatTemplate.percentage(2)
print(TABLE_COLUMNS)
###############################################################################


###############################################################################
# Define the dashboard.
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = 'Woonfraude Dashboard'

app.layout = html.Div([

    # Div containing a selection of the data based on dropdown selection.
    html.Div(id='intermediate_value', style={'display': 'none'}),

    # Divs contain a lists of points which have been selected with on-clicks on the map.
    html.Div(id='point_selection', style={'display': 'none'}),

    # Title
    html.Div([
        html.Img(src='/assets/house_bw.png', height=50, width=50),
        dcc.Markdown(
            '# Woonfraude Dashboard',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        )
    ], style={'padding': 2, 'display': 'flex'}),

    # Create map.
    html.Div(
        dcc.Graph(
            id='map',
            config={'displayModeBar': False},  # Turned off to disable selection with box/lasso etc.
        ),
        style={'padding': 2}
    ),

    # Create drop down filter for categories.
    html.Div([
        html.H4('Selecteer categorieën:'),
        dcc.Dropdown(
            id='categorie_dropdown',
            placeholder='Selecteer categorieën',
            options=[{'label': x, 'value': x} for x in df.categorie.unique()],
            multi=True,
            value=df.categorie.unique(),
            style={'width': '500px'}
        ),
    ], style={'padding': 2}),

    # Create drop down filter for city parts.
    html.Div([
        html.H4('Selecteer stadsdelen:'),
        dcc.Dropdown(
            id='stadsdeel_dropdown',
            placeholder='Selecteer stadsdelen',
            options=[{'label': x, 'value': x} for x in df.stadsdeel.unique()],
            multi=True,
            value=df.stadsdeel.unique(),
            style={'width': '500px'}
        ),
    ], style={'padding': 2}),

    # Show the data in a table.
    html.Div([
        html.H4('Gefilterde meldingen:'),
        dt.DataTable(
            id='table',
            columns = TABLE_COLUMNS,
            sort_action='native',
            sort_by=[{'column_id': 'float_test', 'direction': 'desc'}],
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
                        'column_id': 'float_test',
                        'filter_query': '{float_test} ge 0.5'
                    },
                    'backgroundColor': colors['fraud'],
                },
                {
                    'if': {
                        'column_id': 'float_test',
                        'filter_query': '{float_test} lt 0.5'
                    },
                    'backgroundColor': colors['no_fraud'],
                },
            ]
        ),
    ], style={'padding': 2}),

    # Show info of items selected on map (using Dash tools like box and lasso).
    # html.Div([
    #     html.H4('Selectie op kaart:', style={'padding': 2}),
    #     html.Ul(id='lasso_map_selection', style={'padding': 2})
    # ]),


    # Show info of items selected on map (using click).
    html.Div([
        html.H4('Klik-selectie (adres_id\'s):', style={'padding': 2}),
        html.Ul(id='filtered_point_selection', style={'padding': 2})
    ])

])


# Callback function for updating the intermediate data based on the dropdown selection.
@app.callback(
    Output('intermediate_value', 'children'),
    [Input('categorie_dropdown', 'value'),
    Input('stadsdeel_dropdown', 'value')]
)
def create_data_selection(selected_categories, selected_stadsdelen):
    # Create a filtered copy of the original dataframe.
    filtered_df = filter_df(df, selected_categories, selected_stadsdelen)
    return filtered_df.to_json(date_format='iso', orient='split')


# Callback function for filling the map based on the dropdown-selections.
@app.callback(
    Output('map', 'figure'),
    [Input('intermediate_value', 'children'),
    Input('point_selection', 'children')],
    [State('map', 'figure')]
)
def plot_map(intermediate_value, point_selection, map_state):

    # Define which input triggers the callback (map.figure or intermediate_value.children).
    trigger_event = dash.callback_context.triggered[0]['prop_id']

    # center=dict(
    #     lat=52.36,
    #     lon=4.89
    # )
    # zoom=11

    # if map_state != None:
    #     center = map_state['layout']['mapbox']['center']
    #     print(center)
    #     zoom = map_state['layout']['mapbox']['zoom']
    #     print(zoom)

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
                    size=20,
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
            width=800,
            height=600,
            margin=go.layout.Margin(l=0, r=0, b=0, t=25, pad=0),
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


# Callback function filling the data table based on the dropdown-selections.
@app.callback(
    Output('table', 'data'),
    [Input('intermediate_value', 'children')]
)
# def plot_map(selected_categories, selected_stadsdelen):
def generate_dash_table(intermediate_value):

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


# Callback function for creating a map selection summarization.
# @app.callback(
#     Output('lasso_map_selection', 'children'),
#     [Input('map', 'selectedData')]
# )
# def summarize_map_selection(selectedData):
#     # # Functionality to print the hover text of the map-selected data points.
#     hover_text = [html.Li("Nog geen selectie gemaakt. Maak een selectie.")]
#     if selectedData != None:
#         hover_text = [html.Li(x['text']) for x in selectedData['points']]
#     return hover_text


# Callback function for selecting map points using click-events.
@app.callback(
    Output('point_selection', 'children'),
    [Input('map', 'clickData'),
    Input('intermediate_value', 'children')],
    [State('point_selection', 'children')]
)
def update_point_selection_on_click(clickData, intermediate_value, existing_point_selection):
    """
    Update point selection with newly selected points, or according to dropdown filters.

    The input "intermediate_value:children" is only used to update the
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

    # Remove any previously selected points, if the dropdown selections rule them out.
    # if trigger_event == 'intermediate_value.children':
    #     df = pd.read_json(intermediate_value, orient='split') # Load the pre-filtered version of the dataframe.
    #     point_ids_list = [str(x) for x in list(df.adres_id)]
    #     for point_id in point_selection:
    #         if point_id not in point_ids_list:
    #             point_selection.remove(point_id)

    return point_selection


# Show list of click-selected points.
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

    return str(point_selection)


###############################################################################
# Run dashboard when calling this script.
if __name__ == '__main__':
    app.run_server(debug=True)
###############################################################################