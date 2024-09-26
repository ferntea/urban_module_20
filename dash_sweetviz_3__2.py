import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import sweetviz as sv
from rdkit import Chem
from rdkit.Chem import AllChem
import tempfile
import base64
import io
import os
import matplotlib
matplotlib.use('Agg')


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Automated EDA Report with SweetViz"),
    dcc.Upload(
        id='upload-data',
        children=html.A('Select CSV File', style={'color': 'white', 'background-color': 'black'}),
        style={'margin-bottom': '20px'},  # Add margin to create space between input line and buttons
        multiple=False
    ),
    html.Button("Generate SweetViz Report", id='generate-button'),
    html.Div(id='output-data-upload')
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    # decoded = base64.b64decode(content_string).decode('utf-8')
    decoded = base64.b64decode(content_string).decode('utf-8', errors='ignore')

    try:
        dataset = pd.read_csv(io.StringIO(decoded))
        if 'SMILES' in dataset.columns and 'target_variable' in dataset.columns:
            targets = dataset['target_variable']
            y = np.array(targets)

            return y
        else:
            return None
    except Exception as e:
        print(e)
        return None

@app.callback(
    Output('output-data-upload', 'children'),
    [Input('generate-button', 'n_clicks')],
    [State('upload-data', 'contents')]
)
def update_output(n_clicks, contents):
    if n_clicks is not None and contents is not None:
        y = parse_contents(contents)

        if y is not None:
            df = pd.DataFrame({'target_variable': y})
            report = sv.analyze(df)

            # report_path = os.path.join(tempfile.gettempdir(), 'SWEETVIZ_REPORT.html')
            #report_path = os.path.join(current_directory, 'SWEETVIZ_REPORT.html')
            report_path = os.path.join('.', 'SWEETVIZ_REPORT.html')
            report.show_html(filepath=report_path)

            return html.Div([
                html.H4('SweetViz Report for Uploaded Data (Numerical Parameter only):'),
                html.Iframe(srcDoc=open(report_path, 'r').read(), width='100%', height='600')
            ])
    else:
        return html.Div(['The uploaded CSV file is missing required columns.'])


if __name__ == '__main__':
    app.run_server(debug=True, port=8052)
