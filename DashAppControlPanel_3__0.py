import dash
from dash import html
import webbrowser
import about

# Initialize the Dash app for the floating window
app = dash.Dash(__name__)

# Define the layout of the floating window
floating_layout = html.Div([
    # Buttons for each app, including the About button on the left
    html.Div([
        html.Button("About", id="about-btn"),  # Add About button here
        html.Button("App 1", id="app1-btn"),
        html.Button("App 2", id="app2-btn"),
        html.Button("App 3", id="app3-btn"),
        html.Button("App 4", id="app4-btn"),
        # Image at the top right
        html.Div([
            html.Img(src='/assets/UrbanUniversity.png', style={'height': 'auto', 'width': '160px'})
            # Adjust this width as needed
        ], style={'position': 'absolute', 'top': '0px', 'right': '10px', 'z-index': '1000'})
        # Positioned at top right with z-index

    ], style={'display': 'flex', 'justify-content': 'flex-start', 'margin-bottom': '20px', 'gap': '5px'}),

    # About information
    about.layout
])

# Set the layout for the floating window
app.layout = floating_layout

# Callbacks to open each app in a new tab when the corresponding button is clicked
@app.callback(
    dash.dependencies.Output("app1-btn", "n_clicks"),
    [dash.dependencies.Input("app1-btn", "n_clicks")]
)
def open_app1(n_clicks):
    if n_clicks:
        webbrowser.open_new_tab("http://127.0.0.1:8051")  # Open App 1 in a new tab
        return None

@app.callback(
    dash.dependencies.Output("app2-btn", "n_clicks"),
    [dash.dependencies.Input("app2-btn", "n_clicks")]
)
def open_app2(n_clicks):
    if n_clicks:
        webbrowser.open_new_tab("http://127.0.0.1:8052")  # Open App 2 in a new tab
        return None

@app.callback(
    dash.dependencies.Output("app3-btn", "n_clicks"),
    [dash.dependencies.Input("app3-btn", "n_clicks")]
)
def open_app3(n_clicks):
    if n_clicks:
        webbrowser.open_new_tab("http://127.0.0.1:8053")  # Open App 3 in a new tab
        return None

@app.callback(
    dash.dependencies.Output("app4-btn", "n_clicks"),
    [dash.dependencies.Input("app4-btn", "n_clicks")]
)
def open_app4(n_clicks):
    if n_clicks:
        webbrowser.open_new_tab("http://127.0.0.1:8054")  # Open App 4 in a new tab
        return None

# Run the floating window on a specific port
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
