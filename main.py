from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
from zipfile import ZipFile
import os
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import shutil
import sys
import json
import dash

# Figure
fig = go.Figure()
fig.update_yaxes(showline=True, mirror=True, zeroline=False, autorange=True, ticks="inside", tickwidth=1,
tickcolor='black', ticklen=2, title="-Z'' [\u03A9]")
fig.update_xaxes(showline=True, mirror=True, zeroline=False, autorange=True, ticks="inside", tickwidth=1,
                 tickcolor='black', ticklen=2, title="Z' [\u03A9]")
fig.update_layout(width=1300, height=400, template=None)
fig.update_layout(font=dict(family='Arial', size=16), margin=dict(l=120, r=50, b=50, t=50, pad=0))

# File list 
fl = []

# Column list 
keysz = [' No columns found.']

# Cloud ID file
with open('cloud_id.txt', 'r') as data:
    try:
        file = json.loads(data.read())
    except:
        file = dict({'null':'null'})

with open("cloud_id.txt", 'w') as output:
    output.write(json.dumps(file))

# Layout file
with open('layout.txt', 'r') as data:
    try:
        file = json.loads(data.read())
    except:
        file = dict({'null':'null'})

with open("layout.txt", 'w') as output:
    output.write(json.dumps(file))

# APPLICATION
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
colors = {'background': '#FFFFFF', 'text': '#000080'}

app.layout = html.Div(style={'backgroundColor': colors['background'], 'display': 'inline-block', 'width': '100%'},
children=[   
    dbc.Navbar([html.A(
        dbc.Row([
            dbc.Col(html.Div(children='''   ''')),
            dbc.Col(dbc.NavbarBrand("bk_Dash", className="ml-2"))],align="center")),
            dbc.NavbarToggler(id="navbar-toggler"), 
            ], color="#484848", dark=True),
    
    dbc.Row([  

        dbc.Col([
            
            dbc.Card(
                [
                    dbc.CardBody(
                        [   
                            html.H5("Add new experiment", className="card-title"),
                            dbc.Input(placeholder="Name", id="cloud_link_name", type="text", style={'width': '100%'}),
                            dbc.Input(placeholder="https://cloud.tugraz.at/index.php/s/XYZxyzXYZxyzXYZ", id="cloud_link", type="text", style={'width': '100%'}),
                            html.H5(""),
                            dbc.Button("Add", id='button', color="primary", n_clicks=0, className="me-1"),
                            html.Div(id='hidden-div1', children='', style={'display': 'none'}),
                        ],
                    ),
                ], style={"width": "30rem"}
            ),

            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5("List of experiments", className="card-title"),
                            dcc.Dropdown(id="dropdown_ids", options = file, placeholder="Select experiment", multi=False, style={'width': '100%'}),
                            html.Div(id="loading-output"),
                            html.H5(""),
                            dbc.Button("Remove selected experiment", id='button2', color="warning", n_clicks=0, className="me-1"),
                            dbc.Button("Delete data folder", id='button4', color="danger", n_clicks=0, className="me-1"),
                            html.Div(id='hidden-div4', children='', style={'display': 'none'}),
                        ]
                    )
                ], style={"width": "30rem"}
            ),


            dbc.Card(
                [
                    dbc.CardBody(
                        [
                            html.H5("Select column", className="card-title"),
                            dcc.RadioItems(id="ratio_items", options=[{'label': file, 'value': file} for file in keysz]),
                            html.H5(""),
                            html.H5("Select files", className="card-title"),
                            dbc.Input(placeholder="Display last x files", id="latest_files", type="number", step=1, style={'width': '50%'}),
                            dcc.Dropdown(id="dropdown_eis", options=[{'label': file, 'value': file} for file in fl], 
                            placeholder="Select files", multi=True, style={'width': '100%'}),
                        ]
                    )
                ], style={"width": "30rem"}
            ),
            ]),

        dbc.Col([

            dbc.Card(
                dbc.CardBody(
                    [
                    html.H5("Figure", className="card-title"),
                    dcc.Graph(id='eis_graph', figure=fig),
                    ]
                )
            )
        ]),

        dbc.Col([

            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5("Layout", className="card-title"),
                        dbc.Checklist(options=[{"label": "Reverse x axis", "value": 0}], id="rev_x", switch=True,),
                        dbc.Checklist(options=[{"label": "Reverse y axis", "value": 0}], id="rev_y", switch=True,),  

                        html.Div("Set x-axis limits"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("x_max", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="x_max",id="x1_input",type="number",step="0.0001"),
                            ]
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("x_min", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="x_min",value= 0,id="x0_input",type="number",step="0.0001"),
                            ]
                        ),
                                                     
                        html.Div("Set y-axis limits"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("y_max", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="y_max", id="y1_input",type="number",step="0.0001"),
                            ]
                            ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("y_min", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="y_min",value= 0,id="y0_input",type="number",step="0.0001"),
                            ]
                        ),          

                        html.Div("Correction factors"),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("x_corr", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="x_corr", value= "x_corr", id="x_corr",type="number",step="0.0001"),
                            ]
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("y_corr", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="y_corr",value= "y_corr",id="y_corr",type="number",step="0.0001"),
                            ]
                        ), 

                        html.Div("Curve names"),
                        dbc.Input(placeholder="name1, name2, ...", id="curve_names", type="string", style={'width': '100%'}),
                
                        html.Div("Area specific impedance"),
                        dbc.Checklist(options=[{"label": "Activated", "value": 0}], id="area_switch", switch=True,),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("cm\u00b2", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="Cell area", id="area",type="number"),
                            ]
                        ),

                        html.Div("Frequency range"),
                        dcc.RadioItems(id="ratio_items_freq", options=[{'label': file, 'value': file} for file in keysz]),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("f_max", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="f_max",value= "f_max", id="f_max",type="number",step="0.01"),
                            ]
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("f_min", style={"color":"#484848", "width":"5rem"}),
                                dbc.Input(placeholder="f_min",value="f_min",id="f_min",type="number",step="0.01"),
                            ]
                        ),  

                        html.H5(""),
                        dbc.Button("Load layout", id='load_button', color="primary", n_clicks=0, className="me-1"),
                        html.Div(id='hidden-div-load', children='', style={'display': 'none'}),     
                        dbc.Button("Save layout", id='save_button', color="secondary", n_clicks=0, className="me-1"),
                        html.Div(id='hidden-div-save', children='', style={'display': 'none'}),       

                        ]
                    )
                )     
        ])
])
])                                         

# FUNCTIONS & CALLBACKS             

def update_list():
    with open('cloud_id.txt', 'r') as data:
        file = json.loads(data.read())
        file_keys = list(file.keys())
    return file_keys

@app.callback(Output(component_id='hidden-div-save', component_property='children'),
              Input(component_id='save_button', component_property='n_clicks'),
              Input(component_id='dropdown_ids', component_property='value'),
              Input(component_id='x0_input', component_property='value'),
              Input(component_id='x1_input', component_property='value'),
              Input(component_id='y0_input', component_property='value'),
              Input(component_id='y1_input', component_property='value'),
              Input(component_id='rev_x', component_property='value'),
              Input(component_id='rev_y', component_property='value'),
              Input(component_id='x_corr', component_property='value'),
              Input(component_id='y_corr', component_property='value'),
              Input(component_id='curve_names', component_property='value'),
              Input(component_id='area', component_property='value'),
              Input(component_id='area_switch', component_property='value'),
              Input(component_id='f_min', component_property='value'),
              Input(component_id='f_max', component_property='value'),
              Input(component_id='ratio_items_freq', component_property='value'),)
def save_layout(n_clicks, name, x0_input, x1_input, y0_input, y1_input, rev_x, rev_y, x_corr, y_corr, curve_names, area, area_switch, f_min, f_max, ratio_items_freq):
    
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'save_button' in changed_id:

        with open('layout.txt', 'r') as data:
            file = json.loads(data.read())

        if name and name != 'null':
            file[name+'_'+'x0_input'] = x0_input
            file[name+'_'+'x1_input'] = x1_input
            file[name+'_'+'y0_input'] = y0_input
            file[name+'_'+'y1_input'] = y1_input
            file[name+'_'+'rev_x'] = rev_x
            file[name+'_'+'rev_y'] = rev_y
            file[name+'_'+'x_corr'] = x_corr
            file[name+'_'+'y_corr'] = y_corr
            file[name+'_'+'curve_names'] = curve_names
            file[name+'_'+'area'] = area
            file[name+'_'+'area_switch'] = area_switch
            file[name+'_'+'f_min'] = f_min
            file[name+'_'+'f_max'] = f_max
            file[name+'_'+'ratio_items_freq'] = ratio_items_freq

        with open("layout.txt", 'w') as output:
            output.write(json.dumps(file))
    return

@app.callback(Output(component_id='x0_input', component_property='value'),
              Output(component_id='x1_input', component_property='value'),
              Output(component_id='y0_input', component_property='value'),
              Output(component_id='y1_input', component_property='value'),
              Output(component_id='rev_x', component_property='value'),
              Output(component_id='rev_y', component_property='value'),
              Output(component_id='x_corr', component_property='value'),
              Output(component_id='y_corr', component_property='value'),
              Output(component_id='curve_names', component_property='value'),
              Output(component_id='area', component_property='value'),
              Output(component_id='area_switch', component_property='value'),
              Output(component_id='f_min', component_property='value'),
              Output(component_id='f_max', component_property='value'),
              Output(component_id='ratio_items_freq', component_property='value'),
              Input(component_id='load_button', component_property='n_clicks'),
              Input(component_id='dropdown_ids', component_property='value'),)
def load_layout(n_clicks, name):

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'load_button' in changed_id:

         with open('layout.txt', 'r') as data:
            file = json.loads(data.read())       
        
         x0_input = file[name+'_'+'x0_input']
         x1_input = file[name+'_'+'x1_input']
         y0_input = file[name+'_'+'y0_input']
         y1_input = file[name+'_'+'y1_input']
         rev_x = file[name+'_'+'rev_x']
         rev_y = file[name+'_'+'rev_y']
         x_corr = file[name+'_'+'x_corr']
         y_corr = file[name+'_'+'y_corr']
         curve_names = file[name+'_'+'curve_names']
         area = file[name+'_'+'area']
         area_switch = file[name+'_'+'area_switch']
         f_min = file[name+'_'+'f_min']
         f_max = file[name+'_'+'f_max']
         ratio_items_freq = file[name+'_'+'ratio_items_freq']
    
    else:
        x0_input = 0
        x1_input = ""
        y0_input = 0
        y1_input  = ""
        rev_x = []
        rev_y = []
        x_corr = ""
        y_corr = ""
        curve_names = ""
        area = ""
        area_switch = False
        f_min = ""
        f_max  = ""
        ratio_items_freq = " No columns found."

    return x0_input, x1_input, y0_input, y1_input, rev_x, rev_y, x_corr, y_corr, curve_names, area, area_switch, f_min, f_max, ratio_items_freq

@app.callback(Output(component_id='hidden-div4', component_property='children'),
              Input(component_id='button4', component_property='n_clicks'))
def remove_folder(c4):
    if c4 is None:
        raise PreventUpdate
    else:
        f1 = Path('EIS_data/')
        f2 = os.listdir(f1)
        for fi in f2:
            try:
                shutil.rmtree(f1 / fi)
            except:
                pass
    return

@app.callback(Output(component_id='ratio_items', component_property='options'),
            Output(component_id='ratio_items_freq', component_property='options'),
              Input(component_id='dropdown_eis', component_property='value'), )
def ratio_items(dataset):
    keysz = []
    if dataset is not None:
        for ind in dataset:
            folder = Path('EIS_data/')
            inside = os.listdir(folder)
            for ins in inside:
                if 'git' not in ins:
                    sub_folder = folder / ins
            if 'npz' in ind:
                data = np.load(sub_folder / ind)
                keysz = list(data.keys())
            else:
                pass
    else:
        keysz = []
    return keysz, keysz

@app.callback(Output(component_id='dropdown_ids', component_property='options'),
              Output(component_id='cloud_link', component_property='value'),
              Output(component_id='cloud_link_name', component_property='value'),
              Input(component_id='button', component_property='n_clicks'),
              State(component_id='cloud_link', component_property='value'),
              State(component_id='cloud_link_name', component_property='value'),)
def add_to_list(c1, link, name): 
    n_clicks = 0
    if c1 == n_clicks:
        raise PreventUpdate
    else:
        with open('cloud_id.txt', 'r') as data:
            try:
                file = json.loads(data.read())
            except:
                file = dict({'null':'null'})
        if name not in file.keys() and name != '':
            file[name] = link
            with open("cloud_id.txt", 'w') as output:
                output.write(json.dumps(file))     
        n_clicks = c1
    file_keys = update_list()
    return file_keys, '', ''

@app.callback(Output(component_id='button', component_property='n_clicks'),
              Input(component_id='button2', component_property='n_clicks'),
              Input(component_id='dropdown_ids', component_property='value'),)
def remove_from_list(c2,value): 

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'button2' in changed_id:

        with open('cloud_id.txt', 'r') as data:
            file = json.loads(data.read())
        del file[value]

        if file == {}:
            file = dict({'null':'null'})

        with open('cloud_id.txt', 'w') as output:
            output.write(json.dumps(file))

    return 

@app.callback(Output(component_id='dropdown_eis', component_property='value'),
              [Input(component_id='latest_files', component_property='value')])
def update_graph_latest(number): 

    folder = Path('EIS_data/')
    inside = os.listdir(folder)

    for ins in inside:
        if 'git' not in ins:
            sub_folder = folder / ins
    last_files = np.sort(os.listdir(sub_folder))[-number:]

    return last_files

@app.callback([Output(component_id='dropdown_eis', component_property='options'),
               Output("loading-output", "children"),],
              [Input(component_id='dropdown_ids', component_property='value')])
def cloud_download(value): 
    
    with open('cloud_id.txt', 'r') as data:
        file = json.loads(data.read())
    link = file[value]

    if link is not None:
        folder = Path('EIS_data/')
        file_name = "zipped.zip"

        r = requests.request(method="get", url=link + "/download", verify=False)
        if r.status_code == 200:
            with open(folder / file_name, "wb") as file:
                file.write(r.content)

            with ZipFile(folder / file_name, 'r') as zip2:
                zip2.extractall(path=folder)
            os.remove(folder / file_name)

            inside = os.listdir(folder)
            for ins in inside:
                if 'git' not in ins:
                    sub_folder = folder / ins
            fl = np.sort(os.listdir(sub_folder))
            message = 'Download successful.'
        else:
            message = 'Error.'
    else:
        fl = []
        message = 'Error.'

    return fl, message

@app.callback(Output(component_id='eis_graph', component_property='figure'),
              Input(component_id='dropdown_eis', component_property='value'),
              Input(component_id='ratio_items', component_property='value'),
              Input(component_id='x0_input', component_property='value'),
              Input(component_id='x1_input', component_property='value'),
              Input(component_id='y0_input', component_property='value'),
              Input(component_id='y1_input', component_property='value'),
              Input(component_id='rev_x', component_property='value'),
              Input(component_id='rev_y', component_property='value'),
              Input(component_id='x_corr', component_property='value'),
              Input(component_id='y_corr', component_property='value'),
              Input(component_id='curve_names', component_property='value'),
              Input(component_id='area', component_property='value'),
              Input(component_id='area_switch', component_property='value'),
              Input(component_id='f_min', component_property='value'),
              Input(component_id='f_max', component_property='value'),
              Input(component_id='ratio_items_freq', component_property='value'),)
def update_figure(dataset, ratio_value, x0, x1, y0, y1, xr, yr, xc, yc, curve_names, area, area_switch, f_min, f_max, ratio_value_freq):
    
    # Revert axis
    rev_x, rev_y = 1, 1
    if xr:
        rev_x = -1
    if yr:
        rev_y = -1

    # Correct axis
    if type(xc) not in [float, int]:
        xc = 1
    if type(yc) not in [float, int]:
        yc = 1   

    # Rename traces
    try:
        names = curve_names.split(',')
    except:
        names = []

    # Change units
    if area_switch and area_switch != 'null' and area != None and area != 'null':
        area_corr = area
        x_axis_title = "Z' [\u03A9.cm\u00b2]"
        y_axis_title = "-Z'' [\u03A9.cm\u00b2]"
    else:
        area_corr = 1
        x_axis_title = "Z' [\u03A9]"
        y_axis_title = "-Z'' [\u03A9]"

    fig = go.Figure()

    if ratio_value is not None:
        if dataset is not None:
            for i, ind in enumerate(dataset):
                folder = Path('EIS_data/')
                inside = os.listdir(folder)
                for ins in inside:
                    if 'git' not in ins:
                        sub_folder = folder / ins
                data = np.load(sub_folder / ind)
                x_data = np.real(data[ratio_value])
                y_data = np.imag(data[ratio_value])
                
                # Frequency range
                if ratio_value_freq and ratio_value_freq != ' No columns found.':
                    f_data = data[ratio_value_freq]
                    if f_data[0] > f_data[-1]:
                        f_data = np.flip(f_data)
                        x_data = np.flip(x_data)
                        y_data = np.flip(y_data)
                    if type(f_min) in [float, int] and type(f_max) in [float, int]:
                        if f_min < f_max:
                            start_f = np.where(f_data >= f_min)[0][0]
                            stop_f = np.where(f_data <= f_max)[0][-1]
                        else:
                            start_f, stop_f = 0, -1
                    else:
                        start_f, stop_f = 0, -1
                else:
                    start_f, stop_f = 0, -1
           
                # Trace name
                if len(names) < len(dataset):
                    cname = ind
                else:
                    cname = names[i]

                fig.add_trace(go.Scatter(
                    x=x_data[start_f:stop_f]*rev_x*xc*area_corr,
                    y=y_data[start_f:stop_f]*rev_y*yc*area_corr, 
                    name=cname, mode='lines', line=dict(width=3)))
 
    fig.update_yaxes(showline=True, mirror=True, zeroline=False, range=[y0,y1], ticks="inside", tickwidth=1,
                     tickcolor='black', ticklen=2, title=y_axis_title)
    fig.update_xaxes(showline=True, mirror=True, zeroline=False, range=[x0,x1], ticks="inside", tickwidth=1,
                     tickcolor='black', ticklen=2, title=x_axis_title)
    fig.update_layout(width=900, height=400, template=None)
    fig.update_layout(font=dict(family='Arial', size=16), margin=dict(l=120, r=50, b=50, t=50, pad=0))
    return fig

if __name__ == '__main__':

    development = True

    if development == True:
        app.run_server(debug=False, port=8051)
    else:
        if len(sys.argv) != 2:
            print('Usage <port>, default=8051')
            sys.exit(-2)
        port = int(sys.argv[1])
        app.run_server(debug=False, port=port, host='192.168.100.207')
