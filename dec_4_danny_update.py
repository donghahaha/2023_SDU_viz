#%% Preparing Data
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# =============================================================================
# STYLE SECTION.
# =============================================================================
row_1_style = {"height":"70vh"}
container_style = {"position":"relative"}
absolute_object_style = {"position":"absolute", "zIndex":"10000", "bottom":"25%", "left":"5%"}
absolute_object_style_2 = {"position":"absolute", "zIndex":"10000", "bottom":"35%", "left":"5%"}
abs_obj_style_3 = {
        "position":"absolute", 
        "zIndex":"10000", 
        "top":"0%", 
        "left":"5%", 
        # "background": "linear-gradient(to left, blue, red)",
        # "-webkit-background-clip": "text",
        # "color": "transparent"
}

abs_obj_explanations = {
        "position":"absolute", 
        "zIndex":"10000", 
        "top":"50%", 
        "left":"18%",
        #"background":"black",
        #"color":"white",
        "padding":"5px",
        "width":"10%",
        "fontSize":"15px"
}

# =============================================================================
# CSV COLUMN SELECTION, METRICS & MORE
# =============================================================================

available_variables = ['v2x_polyarchy',
                       'v2x_clphy',
                       'v2smonper',
                       'v2x_gender',
                       'v2smgovdom',
                       #'v2x_regime_amb',
                       'v2xeg_eqaccess',
                       'v2x_suffr',
                       'v2x_freexp_altinf',
                       'v2xel_frefair', 
                       'v2x_frassoc_thick', 
]

#'v2exl_legitlead', # cult of personality
#'v2x_freexp_altinf', 
# 'v2mecenefm', # Traditional media censorship
# 'v2mecenefi' # internet censorship
# 'v2smgovfilprc' # Internet filtering in practice

variables_names = ['Electoral Democracy',
                   'Freedom from Violence',
                   'Online Media Pluralism',
                   'Women\'s Pol. Rights',
                   'Gov. Fake News',
                   #'Type of Regime',
                   'Equal Access',
                   'Legal Suffrage', 
                   'Freedom of Expression',
                   'Clean Elections', 
                   'Freedom of Association.', 
]

metric_descriptions = [
    "Question: To what extent is the ideal of electoral democracy in its fullest sense achieved?",

    "Question: To what extent is physical integrity respected?\nClarification: Physical integrity is understood as freedom from political killings and torture by the government.",

    "Question: Do the major domestic online media outlets represent a wide range of political perspectives?",

    "Question: How politically empowered are women?",
    "Question: How often do the government and its agents use social media to disseminate\n\
misleading viewpoints or false information to influence its own population?",

#     "Question: How can the political regime overall be classified considering the competitiveness of\n\
# access to power (polyarchy) as well as liberal principles?",
    "Question: How equal is access to power?",
    "Question: What share of adult citizens as defined by statute has the legal right to vote in \
national elections?",
    "Question: To what extent does government respect [expression for press media, academicia, and private persons]?",
    "Question: To what extent are elections free and fair?",
    "Question: To what extent are parties, including opposition parties, allowed\n\
to form and to participate in elections, and to what extent are civil society \
            \norganizations able to form and to operate freely?",
]

descriptions_dict = dict(zip(variables_names, metric_descriptions))

metrics_dict=dict(zip(variables_names, available_variables))
metrics_inverse = {v: k for k, v in metrics_dict.items()} # Can remove if not used.

metrics_list_dicts = [{"label": x, "value": y} for x, y in zip(variables_names, available_variables)]

# =============================================================================
# LOADING CSV
# =============================================================================

identifying_columns = ["year", "country_text_id", "country_name", "e_regiongeo"]

df_cols = identifying_columns + available_variables
df = pd.read_csv("/Users/ejvindbrandt/Documents/Uni/SDU/visualization/vis_exam/V-Dem-CY-Full+Others-v13.csv", usecols=df_cols)

df = df[df["year"].isin(range(2000, 2022 + 1))] # Desired year range.
df = df.sort_values(by="year", ascending=True)


# =============================================================================
# LOADING CSV
# =============================================================================


# This scales the columns into a range of [0, 1] to standardize. Might be a better way?
# The problem is that some metrics are not already standardized and instead have a [-4, 4] interval.
def scale_column(column, lower_bound, upper_bound):
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val) * (upper_bound - lower_bound) + lower_bound
    return scaled_column


for column_name in available_variables:
    df[column_name] = scale_column(df[column_name], 0, 1)

df_for_dots = df.copy()
df_for_dots = df_for_dots[df_for_dots["year"].isin([2022, 2000])]
country_list = list(df_for_dots["country_name"].unique())
for metric in available_variables:
    
    df_for_dots[f"trend_{metric}"] = 0
    
    for country in country_list:
        country_df = df_for_dots[df_for_dots["country_name"] == country].fillna(0)
        
        max_year = country_df['year'].max()
        min_year = country_df['year'].min()

        last_val = country_df.loc[country_df['year'] == max_year, metric].values[0]
        init_val = country_df.loc[country_df['year'] == min_year, metric].values[0]
        difference = last_val - init_val

        df_for_dots.loc[df_for_dots["country_name"] == country, f"trend_{metric}"] = difference

df_for_dots.sort_values(by="country_name", inplace=True)
df_for_dots.reset_index(drop=True, inplace=True)

dropdown_options = [{"label": country, "value": country} for country in df["country_name"].drop_duplicates()]


eu_df = df[df["e_regiongeo"].isin(range(1, 5))].copy()
eu_df["e_regiongeo"] = "Europe"

afr_df = df[df["e_regiongeo"].isin(range(5, 10))].copy()
afr_df["e_regiongeo"] = "Africa"

asia_df = df[df["e_regiongeo"].isin(range(10, 16))].copy()
asia_df["e_regiongeo"] = "Asia"

amer_df = df[df["e_regiongeo"].isin(range(16, 20))].copy()
amer_df["e_regiongeo"] = "Americas"



#%% Dash App

# =============================================================================
# DASH APP HTML LAYOUT
# =============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([

    
# First row items.

    dbc.Row([
        
        # Choropleth Map
        dbc.Col([
            html.H1(id="headline", style=abs_obj_style_3),
            html.P(id="explanations", style=abs_obj_explanations),
            dcc.Graph(id="choropleth", style={
                      "height": "100%", "width": "100%"}),

            # Region Selector
            html.Div([
                html.P("Toggle Colorblind Mode"),
                daq.BooleanSwitch(id='color_switch', on=False),
                html.P("Select a Metric", style={"marginBottom": 0}),
                # Metric Selector
                dcc.Dropdown(
                    id="Democracy metric",
                    options=list(metrics_dict.keys()),
                    value=list(metrics_dict.keys())[0],
                    clearable=False,
                    multi=False,
                ),
                html.P("Select a Region", style={"marginBottom": 0}),
                dcc.Dropdown(
                    id="regions",
                    options=["World", "Europe", "Asia", "Africa",
                             "North America", "South America"],
                    value="World",
                    clearable=False
                ),
            ], style=absolute_object_style)

        ], width=9, style=container_style),
    
        dbc.Col([
            dcc.Graph(id="select_country_graph", style={"height": "35vh", "textAlign": "left"}),
            dcc.Graph(id="area_chart", style={"height": "35vh", "textAlign": "left"}), 
        ], width=3),
    ], style=row_1_style),

    html.Hr(),

# Second row items.
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id="multi_compare",
                options=dropdown_options,
                value=["Denmark", "South Korea", "Hungary"], 
                multi=True,
                style={"margin":"0", "padding":"0"}
                ),  # Multi allows for multiple selections.
                dbc.Tabs([
                    dbc.Tab(
                       dcc.Graph(id="compare_graph", style={"height": "22vh", "margin":"0", "padding":"0"}), 
                    label = "Single variable over time"),
                    dbc.Tab(
                       dcc.Graph(id="radar-chart", style={"height": "22vh", "margin":"-10", "padding":"0"}), 
                    label = "All varibles for a single year")
                ])
            
        ], width=4, style={"height": "100%"}),
        
        dbc.Col(
            
            dbc.Tabs([
                dbc.Tab(
                   dcc.Graph(id="global_trend_1", style={"height": "22vh"}), 
                label = "Line Graph"),
                dbc.Tab(
                   dcc.Graph(id="global_trend_2", style={"height": "22vh"}),
                label = "Area Graph")
            ]),
            
            width=4, style={"height": "100%"}),
        dbc.Col([
            dcc.Dropdown(
                id="select_year",
                options=[{"label": x, "value": x}
                         for x in df["year"].unique()],
                value=2022,
                clearable=False,
                style={"position": "absolute", "zIndex": "10000",
                       "top": "-3%", "right": "10%", "width": 100}
            ),
            dcc.Graph(id="min_max_graph", style={"height": "100%"}),
        ], style={"height": "100%", "position": "relative"}, width=4)
    ], style={"height": "25vh"}),



], fluid=True, style={"height": "100vh"})  # Sets the dbc container to fill the entire page


# =============================================================================
# DASH APP CALLBACKS SECTION
# =============================================================================


@app.callback(
    Output("select_country_graph", "figure"),
    Input("Democracy metric", "value"),
    Input("choropleth", "clickData"),
)
def update_select_country(selected_metric, selected_country):
    selected_column = metrics_dict[selected_metric]

    if selected_country is None:
        
        country_iso = "TUR"
        country_name = "Turkey"
    else: 
        country_iso = selected_country["points"][0]["location"]
        country_name = selected_country["points"][0]["hovertext"]
    
    wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_{country_name}'>ㆍ</a>" for year in range(2000,2023)]

    df_ = df[df["country_text_id"] == country_iso].groupby("year")[selected_column].mean().reset_index()

    fig_selected_country = px.line(
        df_,
        x="year",  
        y=[selected_column],
        markers=True
    )

    for year in range(2000,2023):
        fig_selected_country.add_annotation(
                                x=year,  
                                y=df_.iloc[year-2000][f"{selected_column}"],
                                showarrow=False,
                                text=wikipedia_link[year-2000],
                                xanchor='auto',
                                yanchor='auto')
    

    fig_selected_country.update_traces(hovertemplate=None, showlegend=False)
    fig_selected_country.update_layout(
        hovermode='x unified',
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis=dict(tickmode='linear')
    ),

    fig_selected_country.update_yaxes(
        range=[df[selected_column].min(), 
        df[selected_column].max()],
        autorange=True,
    )
    
    fig_selected_country.add_annotation(text=f"{selected_metric}: {df.loc[df['country_text_id'] == country_iso, 'country_name'].iloc[0]}",
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.1, showarrow=False)

    return fig_selected_country


@app.callback(
    Output("compare_graph", "figure"),
    Input("Democracy metric", "value"),
    Input("multi_compare", "value"),
    Input("choropleth", "selectedData")
)
def update_comparison(selected_metric, compare, box_select):
    selected_column = metrics_dict[selected_metric]


    if ctx.triggered_id == "choropleth":
        country_list = []
        for points in box_select["points"]:
            if "hovertext" in points:
                country_list.append(points["hovertext"])
        compare = country_list

    # Comparison tool
    compare_df = df[df["country_name"].isin(compare)]
    compare_df = compare_df.groupby(["country_name", "year"])[
        selected_column].mean().reset_index()

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
        xaxis_title="",
        yaxis_title="",
        margin=dict(l=0, r=0, t=20, b=0),
        legend_orientation='v',
        legend_title_text='',
        showlegend=False,
        #xaxis=dict(tickmode='linear')
    )
    
    fig_comparison.update_yaxes(
        autorange=True,
    )

    return fig_comparison


@app.callback(
    Output('radar-chart', 'figure'),
    Input('multi_compare', 'value'),
    Input('select_year', 'value'),
)
def update_radar_chart(selected_countries, selected_year):
    
    fig = go.Figure()
    
    for country in selected_countries:        
        selected_df = df.loc[(df['country_name'] == country) & (df['year'] == selected_year)]
        r_data = selected_df[metrics_dict.values()].values.flatten()
        fig.add_trace(go.Scatterpolar(
              r=r_data,
              theta=list(metrics_dict.keys()),
              fill='toself',
              name= country
              ))

        fig.update_layout(
          polar=dict(
            radialaxis=dict(
              visible=True
            )),
          showlegend=True,
          margin=dict(l=0, r=0, t=10, b=15) # sizes the plot
        )

    return fig


@app.callback(
    Output("choropleth", "figure"),
    Input("Democracy metric", "value"),
    Input("regions", "value"),
    Input("choropleth", "clickData"),
    Input("color_switch", "on"),
    State("choropleth", "figure"),
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
    first_layer_fig_choropleth = px.choropleth(
        df[["country_text_id", "country_name", "year", selected_column]],
        labels={selected_column: legend_title},
        locations="country_text_id",
        locationmode="ISO-3",
        color=selected_column,
        hover_name="country_name",
        # Just toying with some different types.
        color_continuous_scale="greys" if switch else "RdBu",
        #color_continuous_scale=["rgb(222, 50, 32)", "rgb(0, 90, 181)"] if switch else "RdBu",
        scope=selected_region.lower(),
        animation_frame="year",
    )


    first_layer_fig_choropleth.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>%{z}"
    )

    first_layer_fig_choropleth.update_layout(
        dragmode=False,
        coloraxis_colorbar=dict(x=0.8, y=-0.10, len=0.5,
                                thickness=10, title="", orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),


        geo=dict(
            resolution=110,
            projection_type="equirectangular",
            landcolor="yellow" if switch else "black",
            showlakes=False,
            showframe=False,
            showcountries=False,

        ),

        annotations=[dict(
            x=0.04,
            y=0.10,
            text='Source: <a href="https://v-dem.net/data/the-v-dem-dataset/", target="_blank">The V-Dem Dataset</a>',
            showarrow=False
        ),
            ],

        updatemenus=[dict(y=0.1)],
        sliders=[dict(y=0.1, len=0.4)]
    )

    if selected_region == "World":
        first_layer_fig_choropleth.update_geos(lataxis_range=[-59, 100])  # REMOVES ANTARCTICA!

    second_layer_fig_markers = go.Figure(
        go.Scattergeo(
            locations=df_for_dots["country_text_id"],
            locationmode="ISO-3",
            marker=dict(
                color=df_for_dots[f"trend_{selected_column}"],
                colorscale=[[0, 'red'], [0.5, "black"], [1, 'blue']],
                size = np.minimum(abs(df_for_dots[f"trend_{selected_column}"]) * 40, 10),
                symbol=df_for_dots[f"trend_{selected_column}"].apply(
                    lambda x: "triangle-up" if x > 0 else "triangle-down"),
                opacity=1,
                line=dict(color='black', width=1)
            ),

            hoverinfo='skip',

        )
    )
    
    

    second_layer_fig_markers.update_geos(
        visible=False
    )

    second_layer_fig_markers.update_layout(
        dragmode=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # Adds click functionality of highlighting a country.
    # If a country is clicked, this layer is added to the model.
    # This choropleth has only one location (selected) and a different color scale.

    third_layer_fig_selection = go.Choropleth(
        locations=[clicked["points"][0]["location"]] if clicked else [""],
        z=[1],
        locationmode="ISO-3",
        #colorscale=[[0, "black" if switch else "yellow"]], 
        showscale=False,  # Hides the color scale.
        hoverinfo='skip',
    )


    first_layer_fig_choropleth.add_traces(second_layer_fig_markers.data)

    first_layer_fig_choropleth.add_traces(third_layer_fig_selection)
    

    second_layer_fig_markers.update_geos(
        visible=False
    )

    second_layer_fig_markers.update_layout(
        dragmode=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    

    return first_layer_fig_choropleth


@app.callback(
    Output('min_max_graph', 'figure'),
    Input("select_year", "value"),
    Input("Democracy metric", "value"),
    State("min_max_graph", "figure")
)
def update_min_max_graph(year, selected_metric, old_fig):
    column = metrics_dict[selected_metric]
    shortened_df = df[["year", "country_name", column]]
    selected_df = shortened_df.loc[df["year"] ==
                                   year].sort_values(by=column, ascending=True)
    min_max_df = pd.concat(
        [selected_df.head(5), selected_df.tail(5)], ignore_index=True, axis=0)

    fig_combined = px.bar(
        min_max_df,
        x=column,
        y='country_name',
        title=f"Best and Worst in {selected_metric}",
        color=column,
        color_continuous_scale='RdBu'
    )

    fig_combined.update_layout(
        xaxis_title=f"{selected_metric}",
        yaxis_title="State",
        coloraxis_showscale=False,
        template='plotly_white',
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig_combined.update_traces(
        hovertemplate="%{x}"
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

    grouped_df = grouped_df[grouped_df["year"].between(
        2000, 2022)].groupby("year").mean().reset_index()

    plot = go.Figure()
    n_metrics = 0
    for name, value in metrics_dict.items():
        n_metrics += 1
        df[value] = MinMaxScaler().fit_transform(df[[value]])
        plot.add_trace(go.Scatter(
            name=name,
            x=grouped_df["year"],
            y=grouped_df[value],
            stackgroup='one',
        ))
    plot.update_yaxes(range=(0, n_metrics+1), autorange=True,)
    plot.update_layout(
        hovermode='x unified',
        margin=dict(l=0, r=0, t=10, b=0),
        legend_orientation='h',
    )

    return plot


@app.callback(
    Output('global_trend_1', 'figure'),
    Input("Democracy metric", "value"),
)

def update_global_trends_line(selected_metric):
    column = metrics_dict[selected_metric]
    df_sum = df.groupby('year')[column].mean().reset_index()

    # Plot using px.line
    global_fig = px.line(df_sum, x='year', y=column, title='Global Trends')
    global_fig.update_layout(
        xaxis_title="",
        yaxis_title=f"{selected_metric}",
        coloraxis_showscale=False,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0),
    )
    
   
    return global_fig

@app.callback(
    Output('global_trend_2', 'figure'),
    Input("Democracy metric", "value"),
)

def update_global_trends_area(selected_metric):
    
    column = metrics_dict[selected_metric]
    eu_mean = eu_df.groupby("year")[column].mean().reset_index()
    afr_mean = afr_df.groupby("year")[column].mean().reset_index()
    asia_mean = asia_df.groupby("year")[column].mean().reset_index()
    amer_mean = amer_df.groupby("year")[column].mean().reset_index()
        
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eu_mean["year"], y=eu_mean[column],
        hoverinfo='x+y',
        mode='lines',
        stackgroup='one', # define stack group
        name="Europe"
    ))
    fig.add_trace(go.Scatter(
        x=afr_mean["year"], y=afr_mean[column],
        hoverinfo='x+y',
        mode='lines',
        stackgroup='one',
        name="Africa"
    ))
    fig.add_trace(go.Scatter(
        x=asia_mean["year"], y=asia_mean[column],
        hoverinfo='x+y',
        mode='lines',
        stackgroup='one',
        name="Asia"
    ))
    fig.add_trace(go.Scatter(
        x=amer_mean["year"], y=amer_mean[column],
        hoverinfo='x+y',
        mode='lines',
        stackgroup='one',
        name="Americas"
    ))
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title=f"{selected_metric}",
        coloraxis_showscale=False,
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig



@app.callback(
    Output("headline", 'children'),
    Input("Democracy metric", "value"),
    Input("regions", "value"),
)

def update_headline(selected_metric, region):
    if region.lower() == "world":
        region = "the world"
    title = f"{selected_metric.upper()} IN {region.upper()}"
    return title


@app.callback(
    Output("explanations", 'children'),
    Input("Democracy metric", "value"),
)

def update_metric_text(selected_metric):
    text = descriptions_dict[selected_metric]
    
    return text

app.run_server(debug=True, use_reloader=False) # Can also set to true, but doesn"t work for me for some reason.