# All buttons are arranged - all embedding plots are depicted correctly

import dash
from dash import dcc, Input, Output, dash_table, html
from dash.dependencies import Input, Output, State
from dash import callback_context
import pandas as pd
import plotly.express as px
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import base64
import io
import gc

# download from scikit-learn!!!
from sklearn.manifold import TSNE as SklearnTSNE
# from umap import UMAP
# import UMAP
from umap.umap_ import UMAP
from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import MDS
# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from openTSNE import TSNE as PTSNE

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Embedding of descriptors of chemical compounds and interactive plots"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.A('Select CSV File', style={'color': 'white', 'background-color': 'black'})
        ]),
        style={'margin-bottom': '20px'},  # Add margin to create space between input line and buttons
        multiple=False
    ),
    html.Div([
        html.Button('TSNE', id='tsne_plot', style={'display': 'block'}),
        html.Button('UMAP', id='umap_plot', style={'display': 'block', 'margin-top': '5px'}),
        html.Button('Spectral', id='spectral_plot', style={'margin-top': '5px'}),
        html.Button('Random Tree', id='random_tree_plot', style={'margin-top': '5px'}),
        html.Button('MDS', id='mds_plot', style={'margin-top': '5px'}),
        html.Button('LTSA LLE', id='ltsa_lle_plot', style={'margin-top': '5px'}),
        html.Button('Isomap', id='isomap_plot', style={'margin-top': '5px'}),
        html.Button('LDA', id='lda_plot', style={'margin-top': '5px'}),
        html.Button('Trancated SVD', id='tsvd_plot', style={'margin-top': '5px'}),
        html.Button('SRP', id='srp_plot', style={'margin-top': '5px'}),
        html.Button('PCA', id='pca_plot', style={'margin-top': '5px'}),
        html.Button('Kernel PCA', id='kpca_plot', style={'margin-top': '5px'}),
        html.Button('Parametric TSNE', id='ptsne_plot', style={'margin-top': '5px'}),
    ],
        style={'display': 'flex', 'flex-direction': 'column', 'align-items': 'flex-start'}),
    html.Div(
        id='plot-container',
        style={'text-align': 'center', 'margin': 'auto', 'width': '50%', 'margin-top': '-375px'}
    )
])

def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')

    try:
        dataset = pd.read_csv(io.StringIO(decoded))
        if 'SMILES' in dataset.columns and 'target_variable' in dataset.columns:
            smiles = dataset['SMILES']
            targets = dataset['target_variable']

            molecules = [Chem.MolFromSmiles(smi) for smi in smiles]
            descriptors = [AllChem.GetMorganFingerprintAsBitVect(
                mol,
                8,
                useFeatures=True,
                nBits=4096
            ) for mol in molecules]

            X = np.array(descriptors)
            y = np.array(targets)

            return X, y, dataset

        else:
            return None
    except Exception as e:
        print(e)
        return html.Div(['There was an error processing the uploaded data.'])

def plot_scatter(df, x_col, y_col, color_col, hover_name, color_continuous_scale, title):
    fig = px.scatter(df,
        color=color_col,
        x=x_col,
        y=y_col,
        template='plotly_dark',
        width=600,
        height=400,
        color_continuous_scale=color_continuous_scale,
        hover_name=hover_name)
    fig.update_layout(title_text=title)

    fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn', default_height=500, default_width=700)
    with open(f'{title}.html', 'w') as file:
        file.write(fig_html)

    return fig

@app.callback(
    Output('plot-container', 'children'),
    Input('tsne_plot', 'n_clicks'),
    Input('umap_plot', 'n_clicks'),
    Input('spectral_plot', 'n_clicks'),
    Input('random_tree_plot', 'n_clicks'),
    Input('mds_plot', 'n_clicks'),
    Input('ltsa_lle_plot', 'n_clicks'),
    Input('isomap_plot', 'n_clicks'),
    Input('lda_plot', 'n_clicks'),
    Input('tsvd_plot', 'n_clicks'),
    Input('srp_plot', 'n_clicks'),
    Input('pca_plot', 'n_clicks'),
    Input('kpca_plot', 'n_clicks'),
    Input('ptsne_plot', 'n_clicks'),
    Input('upload-data', 'contents')
)

def update_plot(tsne_clicks, umap_clicks, spectral_clicks, random_clicks, mds_clicks, ltsa_lle_clicks,
                isomap_clicks, lda_clicks, tsvd_clicks, srp_clicks, pca_clicks, kpca_clicks, ptsne_clicks,
                contents):

    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if contents is not None:
        if isinstance(contents, list):
            contents = contents[0]

        data = parse_contents(contents)

        if data is not None:
            X, y, dataset = data
            fig = None

            if triggered_id == 'tsne_plot':
                tsne = SklearnTSNE(n_components=2,
                    perplexity=10,
                    early_exaggeration=12,
                    n_iter=1000,
                    n_jobs=2,
                    random_state=0)
                tsne_data = tsne.fit_transform(X)
                df_tsne = pd.DataFrame(data=tsne_data, columns=['Dim 1', 'Dim 2'])
                df_tsne['y'] = y  # Add the y variable as a column to the DataFrame
                fig = plot_scatter(df_tsne, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'TSNE')

            elif triggered_id == 'umap_plot':
                umap = UMAP(n_components=2,
                    n_neighbors=30,
                    min_dist=0.1,
                    metric='euclidean',
                    spread=0.99,
                    # learning_rate=500,
                    init='pca',
                    random_state=0)
                umap_data = umap.fit_transform(X)
                df_umap = pd.DataFrame(data=umap_data, columns=['Dim 1', 'Dim 2'])
                df_umap['y'] = y
                fig = plot_scatter(df_umap, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'UMAP')

            elif triggered_id == 'spectral_plot':
                spectral = SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
                spectral_data = spectral.fit_transform(X)
                df_spectral = pd.DataFrame(data=spectral_data, columns=['Dim 1', 'Dim 2'])
                df_spectral['y'] = y
                fig = plot_scatter(df_spectral, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'Spectral')

            elif triggered_id == 'random_tree_plot':
                pipeline = make_pipeline(
                    RandomTreesEmbedding(n_estimators=200, max_depth=2, random_state=0),
                    TruncatedSVD(n_components=2))
                pipeline_data = pipeline.fit_transform(X)
                df_pipeline = pd.DataFrame(data=pipeline_data, columns=['Dim 1', 'Dim 2'])
                df_pipeline['y'] = y
                fig = plot_scatter(df_pipeline, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                   'RdBu_r', 'Random Tree')

            elif triggered_id == 'mds_plot':
                mds = MDS(n_components=2, n_init=4, max_iter=300, n_jobs=2, random_state=0)
                mds_data = mds.fit_transform(X)
                df_mds = pd.DataFrame(data=mds_data, columns=['Dim 1', 'Dim 2'])
                df_mds['y'] = y
                fig = plot_scatter(df_mds, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'MDS')

            elif triggered_id == 'ltsa_lle_plot':
                lle = LocallyLinearEmbedding(n_components=2,
                    n_neighbors=30,
                    method="modified",  # the best parameter among others!!!
                    neighbors_algorithm='ball_tree',
                    random_state=0)
                lle_data = lle.fit_transform(X)
                df_lle = pd.DataFrame(data=lle_data, columns=['Dim 1', 'Dim 2'])
                df_lle['y'] = y
                fig = plot_scatter(df_lle, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'LTSA LLE')

            elif triggered_id == 'isomap_plot':
                isomap = Isomap(n_neighbors=30, n_components=2)
                isomap_data = isomap.fit_transform(X)
                df_isomap = pd.DataFrame(data=isomap_data, columns=['Dim 1', 'Dim 2'])
                df_isomap['y'] = y
                fig = plot_scatter(df_isomap, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                   'RdBu_r', 'Isomap')

            elif triggered_id == 'lda_plot':
                lda = LinearDiscriminantAnalysis(n_components=2)
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y)
                lda_data = lda.fit_transform(X, y_encoded)
                df_lda = pd.DataFrame(data=lda_data, columns=['Dim 1', 'Dim 2'])
                df_lda['y'] = y
                fig = plot_scatter(df_lda, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'LDA')

            elif triggered_id == 'tsvd_plot':
                tsvd = TruncatedSVD(n_components=2)
                tsvd_data = tsvd.fit_transform(X)
                df_tsvd = pd.DataFrame(data=tsvd_data, columns=['Dim 1', 'Dim 2'])
                df_tsvd['y'] = y
                fig = plot_scatter(df_tsvd, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                   'RdBu_r', 'TSVD')

            elif triggered_id == 'srp_plot':
                srp = SparseRandomProjection(n_components=2,
                    density=0.3,
                    eps=0.1, random_state=0,
                    dense_output=False)
                srp_data = srp.fit_transform(X)
                df_srp = pd.DataFrame(data=srp_data, columns=['Dim 1', 'Dim 2'])
                df_srp['y'] = y
                fig = plot_scatter(df_srp, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'SPR')

            elif triggered_id == 'pca_plot':
                pca = PCA(n_components=2,
                    whiten=True,
                    svd_solver='full',
                    random_state=0,
                    # copy='True',
                    iterated_power=500)
                pca_data = pca.fit_transform(X)
                df_pca = pd.DataFrame(data=pca_data, columns=['Dim 1', 'Dim 2'])
                df_pca['y'] = y
                fig = plot_scatter(df_pca, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'PCA')

            elif triggered_id == 'kpca_plot':
                kpca = KernelPCA(n_components=2,
                    kernel='poly',  # high and middle values
                    gamma=0.3,
                    degree=2,
                    coef0=0.3,
                    eigen_solver='auto',
                    ## kernel='rbf',    # low values
                    ## gamma=0.03,
                    # kernel='sigmoid', # high and low values
                    # gamma=0.04,
                    # coef0=5.0
                    )
                kpca_data = kpca.fit_transform(X)
                df_kpca = pd.DataFrame(data=kpca_data, columns=['Dim 1', 'Dim 2'])
                df_kpca['y'] = y
                fig = plot_scatter(df_kpca, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                  'RdBu_r', 'KPCA')

            elif triggered_id == 'ptsne_plot':
                ptsne = PTSNE(n_components=2,
                    initialization='pca',
                    perplexity=7,
                    random_state=0,
                    early_exaggeration=5,
                    metric='euclidean',
                    verbose=False,
                    theta=0.2)
                ptsne_data = ptsne.fit(X)
                df_ptsne = pd.DataFrame(data=ptsne_data, columns=['Dim 1', 'Dim 2'])
                df_ptsne['y'] = y
                fig = plot_scatter(df_ptsne, 'Dim 1', 'Dim 2', 'y', dataset['Name'],
                                                'RdBu_r', 'PTSNE')

            if fig is not None:
                return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run_server(debug=True, port=8054)