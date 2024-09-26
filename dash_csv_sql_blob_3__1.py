# 0_dash_csv_sql_blob_2.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import sqlite3
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64
import io
import subprocess
import os

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("CSV to SQL and SMILES to Images"), className="text-center")
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(id="upload-data", children=dbc.Button("Upload CSV", color="primary"), multiple=False),
                width=4),
        dbc.Col(dbc.Button("Generate Images", id="btn-generate-images", color="success"), width=4),
        dbc.Col(dbc.Button("Open DB Browser", id="btn-open-db", color="info"), width=4)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id="output-div"), className="text-center")
    ])
])

# Function to check if a column exists
def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns

# Callback for uploading CSV and generating images
@app.callback(
    Output("output-div", "children"),
    Input("upload-data", "contents"),
    Input("btn-generate-images", "n_clicks"),
    Input("btn-open-db", "n_clicks"),
    State("upload-data", "filename"),
    prevent_initial_call=True
)
def perform_operations(contents, generate_clicks, open_db_clicks, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ""

    if contents is not None and ctx.triggered[0]["prop_id"].split(".")[0] == "upload-data":
        # Read CSV from uploaded file
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Create a connection to the SQLite database
            db = sqlite3.connect('compounds.db')
            table_name = 'compounds'
            df.to_sql(table_name, db, if_exists='replace', index=False)
            db.commit()
            db.close()
            return "CSV data imported successfully!"
        except Exception as e:
            return f"Error importing CSV: {str(e)}"

    elif ctx.triggered[0]["prop_id"].split(".")[0] == "btn-generate-images":
        # Generate images from SMILES
        try:
            db = sqlite3.connect('compounds.db')
            cursor = db.cursor()

            # Check if the image column exists
            if not column_exists(cursor, "compounds", "image"):
                cursor.execute(f"ALTER TABLE compounds ADD COLUMN image BLOB")

            query = "SELECT id, smiles FROM compounds"
            cursor.execute(query)
            results = cursor.fetchall()
            images = []

            for row in results:
                smiles = row[1]
                mol = Chem.MolFromSmiles(smiles)
                img = Draw.MolToImage(mol)
                byte_io = BytesIO()
                img.save(byte_io, format='PNG')
                images.append(byte_io.getvalue())

            # Update the database with images
            for i, image in enumerate(images):
                cursor.execute(f"UPDATE compounds SET image = ? WHERE id = ?", (image, results[i][0]))

            db.commit()
            db.close()
            return "Images generated and saved successfully!"
        except Exception as e:
            return f"Error generating images: {str(e)}"

    elif ctx.triggered[0]["prop_id"].split(".")[0] == "btn-open-db":
        # Open the database browser
        db_file_path = "compounds.db"
        if os.path.exists(db_file_path):
            # Full path to the DB Browser executable
            db_browser_path = r"C:\Program Files\DB Browser for SQLite\DB Browser for SQLite.exe"
            subprocess.Popen([db_browser_path, db_file_path])
            return "DB Browser opened!"
        else:
            return "Database file does not exist."

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)