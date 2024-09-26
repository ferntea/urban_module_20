import dash
from dash.dependencies import Input, Output
from dash import html, dcc, Input, Output
import base64
import os
import subprocess
from PIL import Image
import io
from io import BytesIO



# Define the Python interpreter path
python_interpreter = r'C:\Users\fernt\PycharmProjects\dash_trial\.venv\Scripts\python.exe'

# Read the GIF image file in binary mode
with open('fig_tnse_anim.gif', 'rb') as f:
    gif_data = f.read()

# Convert the binary data to base64 format
tsne_demo_encoded = base64.b64encode(gif_data).decode('ascii')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the layout of the app with buttons at the left side in a column
app.layout = html.Div([
    html.H1('Embedding of chemical structure descriptors to a 2D-space',
            style={'color': 'black', 'font-size': '30px', 'text-align': 'left'}),
    html.Div([
        # Button and message div
        html.Div([
            html.Button("Generate Plots", id="generate-button",
                        style={'margin-top': '0px', 'color': 'white', 'background-color': 'red'}),
            html.Pre(id="output-message",
                     style={'margin-left': '15px', 'display': 'inline', 'color': 'red'})
        ]),
        html.Button('TSNE demo', id='tsne_demo-button',
                    style={'margin-top': '10px', 'color': 'yellow', 'background-color': 'green'}),
        html.Button('Histogram', id='hist-button', style={'margin-top': '10px'}),
        html.Button('Overview', id='tile-button', style={'margin-top': '5px'}),
        html.Button('TSNE', id='tsne-button', style={'margin-top': '5px'}),
        html.Button('UMAP', id='umap-button', style={'margin-top': '5px'}),
        html.Button('Spectral', id='spectral-button', style={'margin-top': '5px'}),
        html.Button('Random Tree', id='random-tree-button', style={'margin-top': '5px'}),
        html.Button('MDS', id='mds-button', style={'margin-top': '5px'}),
        html.Button('LTSA LLE', id='ltsa_lle-button', style={'margin-top': '5px'}),
        html.Button('Isomap', id='isomap-button', style={'margin-top': '5px'}),
        html.Button('LDA', id='lda-button', style={'margin-top': '5px'}),
        html.Button('Truncated SVD', id='tsvd-button', style={'margin-top': '5px'}),
        html.Button('SRP', id='srp-button', style={'margin-top': '5px'}),
        html.Button('PCA', id='pca-button', style={'margin-top': '5px'}),
        html.Button('Kernel PCA', id='kpca-button', style={'margin-top': '5px'}),
        html.Button('Parametric TSNE', id='ptsne-button', style={'margin-top': '5px'}),
    ], style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-start'}),

    html.Div(id="output"),
    html.Img(id='display-image', style={'max-width': '60%', 'margin-left': '320px',
                                        'margin-top': '-750px'})
])


# Output message after pushing the button for generating plots
@app.callback(
    Output("output-message", "children"),
    [Input("generate-button", "n_clicks")]
)
def generate_plots(n_clicks):
    if n_clicks:
        script_path = 'embedding_py_2.py'
        with open(os.devnull, 'w') as devnull:
            process = subprocess.Popen([python_interpreter, script_path], stdout=devnull, stderr=devnull)
            process.communicate()
# This function does not need to return anything. The purpose of this function is to execute a subprocess
# that generates plots when the 'Generate Plots' button is clicked. The function does not need to return
# any value as it is performing the necessary actions internally without needing to return any result.

        return "Done!"
    # else:
    #     return ""


# Callback function for displaying figures
@app.callback(
    Output('display-image', 'src'),
    Input('tsne_demo-button', 'n_clicks'),
    Input('hist-button', 'n_clicks'),
    Input('tile-button', 'n_clicks'),
    Input('tsne-button', 'n_clicks'),
    Input('umap-button', 'n_clicks'),
    Input('spectral-button', 'n_clicks'),
    Input('random-tree-button', 'n_clicks'),
    Input('mds-button', 'n_clicks'),
    Input('ltsa_lle-button', 'n_clicks'),
    Input('isomap-button', 'n_clicks'),
    Input('lda-button', 'n_clicks'),
    Input('tsvd-button', 'n_clicks'),
    Input('srp-button', 'n_clicks'),
    Input('pca-button', 'n_clicks'),
    Input('kpca-button', 'n_clicks'),
    Input('ptsne-button', 'n_clicks'),
    prevent_initial_call=True
)
def display_figure(
        tsne_demo_clicks,
        hist_clicks,
        tile_clicks,
        tsne_clicks,
        umap_clicks,
        spectral_clicks,
        random_tree_clicks,
        mds_clicks,
        ltsa_lle_clicks,
        isomap_clicks,
        lda_clicks,
        tsvd_clicks,
        srp_clicks,
        pca_clicks,
        kpca_clicks,
        ptsne_clicks
):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if triggered_id == 'tsne_demo-button' and tsne_demo_clicks:
       # tsne_image = open('fig_tsne.png', 'rb').read()
       # tsne_demo_encoded = base64.b64encode(tsne_image).decode()
        return f'data:image/png;base64,{tsne_demo_encoded}'
    elif triggered_id == 'hist-button' and hist_clicks:
        hist_image = open('fig_hist.png', 'rb').read()
        hist_encoded = base64.b64encode(hist_image).decode()
        return f'data:image/png;base64,{hist_encoded}'
    # elif triggered_id == 'tile-button' and tile_clicks:
    #     tile_image = open('fig_tile.png', 'rb').read()
    #     tile_encoded = base64.b64encode(tile_image).decode()
    #     return f'data:image/png;base64,{tile_encoded}'
    elif triggered_id == 'tile-button' and tile_clicks:
        image_bytes = open('fig_tile.png', 'rb').read()
        image = Image.open(io.BytesIO(image_bytes))
        resized_image = image.resize((610, 450))
        image_bytes_resized = io.BytesIO()
        resized_image.save(image_bytes_resized, format='PNG')
        image_bytes_resized.seek(0)
        resized_image_data = image_bytes_resized.read()
        resized_image_encoded = base64.b64encode(resized_image_data).decode()
        return f'data:image/png;base64,{resized_image_encoded}'
    elif triggered_id == 'tsne-button' and tsne_clicks:
        tsne_image = open('fig_tsne.png', 'rb').read()
        tsne_encoded = base64.b64encode(tsne_image).decode()
        return f'data:image/png;base64,{tsne_encoded}'
    elif triggered_id == 'umap-button' and umap_clicks:
        umap_image = open('fig_umap.png', 'rb').read()
        umap_encoded = base64.b64encode(umap_image).decode()
        return f'data:image/png;base64,{umap_encoded}'
    elif triggered_id == 'spectral-button' and spectral_clicks:
        spectral_image = open('fig_spectral.png', 'rb').read()
        spectral_encoded = base64.b64encode(spectral_image).decode()
        return f'data:image/png;base64,{spectral_encoded}'
    elif triggered_id == 'random-tree-button' and random_tree_clicks:
        random_tree_image = open('fig_rt.png', 'rb').read()
        random_tree_encoded = base64.b64encode(random_tree_image).decode()
        return f'data:image/png;base64,{random_tree_encoded}'
    elif triggered_id == 'mds-button' and mds_clicks:
        mds_image = open('fig_mds.png', 'rb').read()
        mds_encoded = base64.b64encode(mds_image).decode()
        return f'data:image/png;base64,{mds_encoded}'
    elif triggered_id == 'ltsa_lle-button' and ltsa_lle_clicks:
        ltsa_lle_image = open('fig_ltsa-lle.png', 'rb').read()
        ltsa_lle_encoded = base64.b64encode(ltsa_lle_image).decode()
        return f'data:image/png;base64,{ltsa_lle_encoded}'
    elif triggered_id == 'isomap-button' and isomap_clicks:
        isoamp_image = open('fig_isomap.png', 'rb').read()
        isomap_encoded = base64.b64encode(isoamp_image).decode()
        return f'data:image/png;base64,{isomap_encoded}'
    elif triggered_id == 'lda-button' and lda_clicks:
        lda_image = open('fig_lda.png', 'rb').read()
        lda_encoded = base64.b64encode(lda_image).decode()
        return f'data:image/png;base64,{lda_encoded}'
    elif triggered_id == 'tsvd-button' and tsvd_clicks:
        tsvd_image = open('fig_svd.png', 'rb').read()
        tsvd_encoded = base64.b64encode(tsvd_image).decode()
        return f'data:image/png;base64,{tsvd_encoded}'
    elif triggered_id == 'srp-button' and srp_clicks:
        srp_image = open('fig_srp.png', 'rb').read()
        srp_encoded = base64.b64encode(srp_image).decode()
        return f'data:image/png;base64,{srp_encoded}'
    elif triggered_id == 'pca-button' and pca_clicks:
        pca_image = open('fig_pca.png', 'rb').read()
        pca_encoded = base64.b64encode(pca_image).decode()
        return f'data:image/png;base64,{pca_encoded}'
    elif triggered_id == 'kpca-button' and kpca_clicks:
        kpca_image = open('fig_kpca.png', 'rb').read()
        kpca_encoded = base64.b64encode(kpca_image).decode()
        return f'data:image/png;base64,{kpca_encoded}'
    elif triggered_id == 'ptsne-button' and ptsne_clicks:
        ptsne_image = open('fig_ptsne.png', 'rb').read()
        ptsne_encoded = base64.b64encode(ptsne_image).decode()
        return f'data:image/png;base64,{ptsne_encoded}'
    else:
        return ''

if __name__ == '__main__':
    app.run_server(debug=True, port=8053)