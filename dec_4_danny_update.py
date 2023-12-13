# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, State, ctx
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


# Put some stuff here to mak layout less busy.
row_1_style = {"height":"70vh"}
container_style = {"position":"relative"}
absolute_object_style = {"position":"absolute", "z-index":"10000", "bottom":"25%", "left":"5%"}
absolute_object_style_2 = {"position":"absolute", "z-index":"10000", "bottom":"35%", "left":"5%"}

# You can replace this with a less.. heavy version of the csv.
df = pd.read_csv("V-Dem-CY-Full+Others-v13.csv")

available_variables = ['v2x_polyarchy', 'v2x_suffr', 'v2xel_frefair', 'v2x_freexp_altinf', 'v2x_frassoc_thick', 'v2x_elecoff']
variables_names = ['Democracy', 'Suffrage', 'free and fair', 'free', 'frassoq', 'election']

metrics_dict=dict(zip(variables_names, available_variables))
metrics_inverse = {v: k for k, v in metrics_dict.items()} # Can remove if not used.

metrics_list_dicts = [{"label": x, "value": y} for x, y in zip(variables_names, available_variables)]

df = df[["year",  # The desired columns from the csv.
         "country_text_id", 
         "country_name", 
         "v2eltrnout",
         "v2exbribe",
         "v2exnamhos",
         "v2exnamhog",
         'v2x_polyarchy', 
         'v2x_suffr', 
         'v2xel_frefair', 
         'v2x_freexp_altinf', 
         'v2x_frassoc_thick', 
         'v2x_elecoff'
         ]]

df = df[df["year"].isin(range(1980, 2023 + 1))] # Desired year range.
df = df.sort_values(by="year", ascending=True)



df_for_dots = df.copy()

country_list = list(df_for_dots["country_name"].unique())

# I am calculating for every variable, but I guess we just need the one main one?
for metric in available_variables:
    df_for_dots[f"trend_{metric}"] = 0
    
    for country in country_list:
        country_df = df_for_dots[(df_for_dots["country_name"] == country) & df_for_dots[metric].notna()]

        if not country_df.empty:
            max_year = country_df['year'].max()
            min_year = country_df['year'].min()

            last_val = country_df.loc[country_df['year'] == max_year, metric].values[0]
            init_val = country_df.loc[country_df['year'] == min_year, metric].values[0]
            difference = round((last_val - init_val), 3)
            
            df_for_dots.loc[df_for_dots["country_name"] == country, f"trend_{metric}"] = difference



#%% so we don"t load the csv file again by accident
# Here begins the dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    # The Choropleth Map Row
    
    dbc.Row([
        # Choropleth Map
        dbc.Col([
            dcc.Graph(id="choropleth", style={"height":"100%", "width": "100%"}),
                
            # Region Selector
            html.Div([
                html.P("Toggle Colorblind Mode"),
                daq.BooleanSwitch(id='color_switch', on=False),
                html.P("Select a Metric", style={"margin-bottom":0}),
                # Metric Selector
                dcc.Dropdown(
                    id="Democracy metric", 
                    options=list(metrics_dict.keys()),
                    value="Democracy",
                    clearable=False,
                    multi=False,
                ),
                html.P("Select a Region", style={"margin-bottom":0}),
                dcc.Dropdown(
                    id="regions", 
                    options=["World", "Europe", "Asia", "Africa", "North America", "South America"],
                    value="World",
                    clearable=False
                ),
            ], style=absolute_object_style)
            
            
            
        ], width=9, style=container_style),
        dbc.Col([
            


            dbc.Tabs([
                dbc.Tab([
                    dcc.Dropdown(
                        id="multi_compare", 
                        options=[{"label": country, "value": country} for country in df["country_name"].drop_duplicates()], # Removes duplicate country entries
                        value=["Denmark", "South Korea", "Hungary"], multi=True), # Multi allows for multiple selections.
                        dcc.Graph(id="compare_graph")
                ], label="Comparison"),
                dbc.Tab(
                    dcc.Graph(id="area_chart"), label="Area Graph", disabled=False),
            ]),
            
            ], width=3),
            

            
    ], style = row_1_style),
    

    
    
    # Click country graph & most democratic states by year graph
    
    dbc.Row([ 
        dbc.Col(
            dcc.Graph(id="select_country_graph", style={"height":"100%"}), 
            style={"height":"30vh"}, width=6
        ),
        dbc.Col([
           dcc.Dropdown(
                id="select_year",
                options=[{"label": x, "value": x} for x in df["year"].unique()],
                value=2022,
                clearable=False,
                style={"position":"absolute", "z-index":"10000", "top":"-2%", "left":"36%", "width":100}
            ),
            dcc.Graph(id="min_max_graph", style={"height":"100%"}) 
        ], style={"height":"30vh", "position":"relative"}, width=6)
    ]),
    
    
    
    
], fluid=True, style = {"height":"100vh"}) # Sets the dbc container to fill the entire page



@app.callback(
    Output("select_country_graph", "figure"),
    Input("Democracy metric", "value"),
    Input("choropleth", "clickData"),
    )

def update_select_country(selected_metric, selected_country):
    
    selected_column = metrics_dict[selected_metric]

    if selected_country is not None:
        country_iso = selected_country["points"][0]["location"]
        fig_selected_country = px.line(
            df[df["country_text_id"] == country_iso].groupby("year")[selected_column].mean().reset_index(),
            x="year",  
            y=[selected_column],    
            #title=f"{legend_title}: {country_name}",
            markers=True,
        )

    
        
        
    else:
        country_iso = "TUR"
        fig_selected_country = px.line(
            df[df["country_text_id"] == country_iso].groupby("year")[selected_column].mean().reset_index(),
            x="year",  
            y=selected_column, 
            #title=f"{legend_title}: {df.loc[df['country_text_id'] == country_iso, 'country_name'].iloc[0]}",
            markers=True,
        )
        
    fig_selected_country.update_traces(hovertemplate=None)
    #fig_selected_country.update_xaxes(rangeslider_visible=True)
    fig_selected_country.update_layout(
        hovermode='x unified',
        xaxis_title="Year", 
        yaxis_title=selected_metric,
        margin=dict(l=0, r=0, t=10, b=10),
    ),

    fig_selected_country.update_yaxes(range=[df[selected_column].min(), df[selected_column].max() ])        
    fig_selected_country.add_annotation(text=f"{selected_metric}: {df.loc[df['country_text_id'] == country_iso, 'country_name'].iloc[0]}",
                  xref="paper", yref="paper",
                  x=0.5, y=1, showarrow=False)
    
    return fig_selected_country


@app.callback(
    Output("compare_graph", "figure"),
    Input("Democracy metric", "value"),
    Input("multi_compare", "value"),
)

def update_comparison(selected_metric, compare):
    
    selected_column = metrics_dict[selected_metric]
    legend_title = selected_metric

    
    # Comparison tool
    compare_df = df[df["country_name"].isin(compare)]  
    compare_df = compare_df.groupby(["country_name", "year"])[selected_column].mean().reset_index()
    
    fig_comparison = px.line(
        compare_df,
        x="year",
        y=selected_column,
        color="country_name",
        markers=True,
    )
    

    
    fig_comparison.update_traces(hovertemplate=None)
    fig_comparison.update_layout(hovermode='x unified')
    fig_comparison.update_xaxes(rangeslider_visible=False)
    
    fig_comparison.update_layout(
        xaxis_title="Year", 
        yaxis_title=legend_title,
        margin=dict(l=0, r=0, t=20, b=0),
        legend_orientation='v',
    )
    
    
    return fig_comparison



        
@app.callback(
    Output("choropleth", "figure"),
    Input("Democracy metric", "value"),
    Input("regions", "value"),
    Input("choropleth", "clickData"),
    Input("color_switch", "on"),
    State("choropleth", "figure")
)


def update_choropleth(selected_metric, selected_region, clicked, switch, old_fig):
    
    if ctx.triggered_id == 'choropleth':
        old_fig["data"][2]["locations"] = [clicked["points"][0]["location"]]
        return old_fig
    
    if ctx.triggered_id == 'regions':
        old_fig["layout"]["geo"]["scope"] = selected_region.lower()
        
        if selected_region != "World":
            old_fig["layout"]["geo"]["lataxis"] = None
        else:
            old_fig["layout"]["geo"]["lataxis"] = {'range': [-59, 100]}
        return old_fig

    selected_column = metrics_dict[selected_metric]
    legend_title = selected_metric
    
    # Main choropleth graph
    choro_fig = px.choropleth(
        df[["country_text_id", "country_name", "year", selected_column]],
        labels={selected_column: legend_title},
        locations="country_text_id",
        locationmode="ISO-3",
        color=selected_column,
        hover_name="country_name",
        color_continuous_scale="gray_r" if switch else "Blues", # Just toying with some different types.
        scope=selected_region.lower(),
        animation_frame="year",

    )
    #choro_fig.update_geos(projection_scale=3.5, center=dict(lon=-95, lat=37), visible=False)
    

    choro_fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>%{z}"
    )


    choro_fig.update_layout(
        dragmode=False,
        coloraxis_colorbar=dict(x = 0.8, y = -0.10, len=0.5, thickness=10, title="", orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        
        
        geo = dict(
            resolution=110,
            projection_type="equirectangular",
            coastlinecolor="Black",
            landcolor="yellow" if switch else "black",
            oceancolor="white",
            showcoastlines=True,
            showland=True,
            showocean=True,
            showlakes=False,
            showframe=False,
        ),
        
        annotations = [dict(
            x=0.04,
            y=0.10,
            text='Source: <a href="https://v-dem.net/data/the-v-dem-dataset/", target="_blank">The V-Dem Dataset</a>',
            showarrow = False
        )],
        
        updatemenus = [dict(y=0.1)],
        sliders = [dict(y=0.1, len=0.4)]
    )
    
    if selected_region == "World":
        choro_fig.update_geos(lataxis_range=[-59, 100]) # REMOVES ANTARCTICA!
    

     
    scatter_icons = go.Figure(
        go.Scattergeo(
            locations=df_for_dots["country_text_id"],
            locationmode="ISO-3",
            marker=dict(
                color=df_for_dots["trend_v2x_polyarchy"],
                colorscale=[[0, 'red'], [0.5, 'blue'], [1, 'green']],
                size=5,
                symbol=df_for_dots["trend_v2x_polyarchy"].apply(lambda x: "triangle-up-open" if x > 0 else "triangle-down-open"),
                opacity=1,
            ),
            
            hoverinfo='skip',

        )
    )
    

    scatter_icons.update_geos(
        visible=False
    )

    scatter_icons.update_layout(
        dragmode=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    
    # Adds click functionality of highlighting a country.
    # If a country is clicked, this layer is added to the model.
    # This choropleth has only one location (selected) and a different color scale.
    
    new_fig = go.Choropleth(
        locations=[clicked["points"][0]["location"]] if clicked else ["USA"], 
        z=[1], # So that country is maximum purple. Or whichever other color.
        locationmode="ISO-3",
        colorscale=[[0, "purple"]],
        showscale=False,  # Hides the color scale.
        hoverinfo='skip',
    )
  
    choro_fig.add_traces(scatter_icons.data)
    
    choro_fig.add_traces(new_fig)

    return choro_fig
    



@app.callback(
    Output('min_max_graph', 'figure'),
    Input("select_year", "value"),
    Input("Democracy metric", "value"),
    State("min_max_graph", "figure")
)
def update_min_max_graph(year, selected_metric, old_fig):
    column = metrics_dict[selected_metric]
    shortened_df = df[["year", "country_name", column]]
    selected_df = shortened_df.loc[df["year"] == year].sort_values(by=column, ascending=True)
    min_max_df = pd.concat([selected_df.head(5), selected_df.tail(5)], ignore_index=True, axis=0)
    
    fig_combined = px.bar(
        min_max_df,
        x=column,
        y='country_name',
        title=f"Most and least democratic states in {year}",
        color=column,
        color_continuous_scale='Blues'
    )
    
    fig_combined.update_layout(
        xaxis_title="Democracy score",
        yaxis_title="State",
        coloraxis_showscale=False,
        template='plotly_white',
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig_combined.update_traces(
        hovertemplate="Democracy score: %{x}"
    )
    
    return fig_combined

@app.callback(
    Output('area_chart', 'figure'),
    Input("Democracy metric", "value"),
    Input("choropleth", "clickData")
)

def update_area_chart(selected_metric, clicked):
    
    if not clicked:
        clicked = "MMR"
    else:
        clicked = clicked["points"][0]["location"]

    grouped_df = df[df["country_text_id"] == clicked]
    
    grouped_df = grouped_df[grouped_df["year"].between(1980, 2022)].groupby("year").mean().reset_index()
    
    
    plot = go.Figure()
    n_metrics = 0
    for name, value in metrics_dict.items():
        n_metrics += 1
        df[value] = MinMaxScaler().fit_transform(df[[value]])
        plot.add_trace(go.Scatter( 
            name = name, 
            x = grouped_df["year"],
            y = grouped_df[value], 
            stackgroup='one',
       ))
    plot.update_yaxes(range=(0, n_metrics+1))
    plot.update_layout(
        margin=dict(l=0, r=0, t=10, b=0)
        )
  
    return plot



app.run_server(debug=True, use_reloader=False) # Can also set to true, but doesn"t work for me for some reason.

