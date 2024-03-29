# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:20:33 2024

@author: caspe
"""

# -*- coding: utf-8 -*-


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

available_variables = [
                        'v2x_polyarchy',
                       'v2xel_frefair', 
                       'v2x_freexp_altinf',
                       'v2x_elecoff',
                       'v2x_suffr',
                       'v2x_frassoc_thick',
                       'v2x_clphy',
                       'v2smgovdom',
]

variables_names = [
                   'Electoral Democracy',
                   'Free and fair elections', 
                   'Freedom of Expression',
                   'Elected officials',
                   'De Jure Suffrage',
                   'Freedom of association',
                   'Freedom from Violence',
                   'Gov. Fake News',
]

metric_descriptions = [
    "Question: To what extent is the ideal of electoral democracy in its fullest sense achieved?",
    "Question: To what extent are elections free and fair?",
    "Question: To what extent does government respect expression for press media, academicia, and private persons?",
    "Question: Is the chief executive and legislature appointed through popular elections?",
    'Question: What share of adult citizens as defined by statute has the legal right to vote in national elections?',
    'Question: To what extent are parties, including opposition parties, allowed to form and to participate in elections, and to what extent are civil society organizations able to form and to operate freely?',
    "Question: To what extent is physical integrity respected?\nClarification: Physical integrity is understood as freedom from political killings and torture by the government.",
    'Question: How often do the government and its agents use social media to disseminate misleading viewpoints or false information to influence its own population?'
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
df = pd.read_csv("V-Dem-CY-Full+Others-v13.csv", usecols=df_cols)

df = df[df["year"].isin(range(2000, 2022 + 1))] # Desired year range.
df = df.sort_values(by="year", ascending=True)


# =============================================================================
# LOADING CSV
# =============================================================================


# This scales the columns into a range of [0, 1] to standardize. Might be a better way?
# The problem is that some metrics are not already standardized and instead have a [-4, 4] interval.
def ordinalize_column(df_name, column_name, lower_bound, upper_bound):
    min_val = df_name[column_name].min()
    max_val = df_name[column_name].max()
    scaled_column = (df_name[column_name] - min_val) / (max_val - min_val) * (upper_bound - lower_bound) + lower_bound
    return scaled_column

df['v2smgovdom'] = ordinalize_column(df,'v2smgovdom', 0, 1)

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

n_amer_df = df[df["e_regiongeo"].isin([16, 17, 19])].copy()
n_amer_df["e_regiongeo"] = "North America"

s_amer_df = df[df["e_regiongeo"].isin([18])].copy()
s_amer_df["e_regiongeo"] = "South America"


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
                html.P("Select a Metric", style={"marginBottom": 0, "font-size": "25px"}),
                # Metric Selector
                dcc.Dropdown(
                    id="Democracy metric",
                    options=list(metrics_dict.keys()),
                    value=list(metrics_dict.keys())[0],
                    clearable=False,
                    multi=False,
                ),
                html.P("Select a Region", style={"marginBottom": 0, "font-size": "25px"}),
                dcc.Dropdown(
                    id="regions",
                    options=["World", "Europe", "Asia", "Africa",
                             "North America", "South America"],
                    value="World",
                    clearable=False
                ),
                html.P("Toggle Colorblind Mode", style={"marginBottom":0}),
                daq.BooleanSwitch(id='color_switch', on=False),
            ], style=absolute_object_style)

        ], width=9, style=container_style),
    
        dbc.Col([
            dcc.Graph(id="select_country_graph", 
                      style={"height": "30vh", "textAlign": "left", "marginTop":"5vh"}
                      ),
            dcc.Graph(id="area_chart", 
                      style={"height": "30vh", "textAlign": "left"}
                      ), 
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
                       dcc.Graph(id="compare_graph", style={"height": "20vh", "margin":"0", "padding":"0"}), 
                    label = "Line (Multiple Years)"),
                    dbc.Tab([
                       dcc.Graph(id="radar-chart", style={"height": "20vh", "margin":"-10", "padding":"0"}),
                       dcc.Dropdown(
                           id="select_year_radar",
                           options=[{"label": x, "value": x}
                                    for x in df["year"].unique()],
                           value=2022,
                           clearable=False,
                           style={"position": "absolute", "zIndex": "10000",
                                  "top": "0", "left": "0%", "width": 100}
                       ),
                    ], label = "Radar (Single Year)", style={"position":"relative"})
                ])
            
        ], width=4, style={"height": "100%"}),
        
        dbc.Col(
            
            dbc.Tabs([
                dbc.Tab(
                   dcc.Graph(id="global_trend_1", style={"height": "22vh"}), 
                label = "Line (World Trends)"),
                dbc.Tab(
                   dcc.Graph(id="global_trend_2", style={"height": "22vh"}),
                label = "Area (Regions Trends)")
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

    if country_name == "Burma/Myanmar":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_Myanmar'>ㆍ</a>" for year in range(2000,2023)]
    elif country_name == "United States of America":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_the_United_States'>ㆍ</a>" for year in range(2000,2023)]
    elif country_name == "United Kingdom":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_the_United_Kingdom'>ㆍ</a>" for year in range(2000,2023)]
    elif country_name == "Democratic Republic of the Congo":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_the_Democratic_Republic_of_the_Congo'>ㆍ</a>" for year in range(2000,2023)]
    elif country_name == "Central African Republic":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_the_Central_African_Republic'>ㆍ</a>" for year in range(2000,2023)]
    elif country_name == "Dominican Republic":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_the_Dominican_Republic'>ㆍ</a>" for year in range(2000,2023)]
    elif country_name == "Timor-Leste":
        wikipedia_link = [f"<a href='http://en.wikipedia.org/wiki/{year}_in_East_Timor'>ㆍ</a>" for year in range(2000,2023)]
    else:
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
        margin=dict(l=0, r=0, t=30, b=10),
        #xaxis=dict(tickmode='linear'),
        title=f"Development in {selected_metric}: {df.loc[df['country_text_id'] == country_iso, 'country_name'].iloc[0]}<br><sup>Click the dot to find out what happened in {country_name}</sup>"
    ),

    fig_selected_country.update_yaxes(
        range=[df[selected_column].min(), 
        df[selected_column].max()],
        autorange=True,
    )
    
    # fig_selected_country.add_annotation(text=f"{selected_metric}: {df.loc[df['country_text_id'] == country_iso, 'country_name'].iloc[0]}",
    #                                     xref="paper", yref="paper",
    #                                     x=0.5, y=0.1, showarrow=False)

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
    Input('select_year_radar', 'value'),
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
          legend_orientation='h',
          margin=dict(l=0, r=0, t=15, b=15) # sizes the plot
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
    
    if switch:
        colorscale = [[0, 'purple'], [0.5, "black"], [1, 'green']]
    else:
        colorscale = [[0, 'red'], [0.5, "black"], [1, 'blue']]
    second_layer_fig_markers = go.Figure(
        go.Scattergeo(
            locations=df_for_dots["country_text_id"],
            locationmode="ISO-3",
            marker=dict(
                color=df_for_dots[f"trend_{selected_column}"],
                colorscale=colorscale,
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
    State("min_max_graph", "figure"),
    Input("regions", "value"),
    Input("choropleth", "selectedData")
)
def update_min_max_graph(year, selected_metric, old_fig, region, box_select):
    column = metrics_dict[selected_metric]
    
    if ctx.triggered_id == "choropleth":
        country_list = []
        for points in box_select["points"]:
            if "hovertext" in points:
                country_list.append(points["hovertext"])
        
        shortened_df = df[df["country_name"].isin(country_list)]
        shortened_df = shortened_df[["year", "country_name", column]]
        
        selected_df = shortened_df.loc[df["year"] ==
                                       year].sort_values(by=column, ascending=True)
        
        number = int(min(5, np.floor(len(selected_df) / 2)))
        
        min_max_df = pd.concat(
            [selected_df.head(number), selected_df.tail(number)], ignore_index=True, axis=0)

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
    
    
    if region == "World":
        shortened_df = df[["year", "country_name", column]]
        
    elif region == "Asia":
        shortened_df = asia_df[["year", "country_name", column]]
    
    elif region == "Europe":
        shortened_df = eu_df[["year", "country_name", column]]
    elif region == "Africa":
        shortened_df = afr_df[["year", "country_name", column]]
    elif region == "North America":
        shortened_df = n_amer_df[["year", "country_name", column]]
    else: 
        shortened_df = s_amer_df[["year", "country_name", column]]
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
        clicked = "TUR"
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
        margin=dict(l=0, r=0, t=30, b=0),
        legend_orientation='h',
        title=f"All Metrics: {df.loc[df['country_text_id'] == clicked, 'country_name'].iloc[0]}"
    )

    return plot


@app.callback(
    Output('global_trend_1', 'figure'),
    Input("Democracy metric", "value"),
    Input("regions", "value"),
    Input("choropleth", "selectedData")
)

def update_global_trends_line(selected_metric, region, box_select):
    
    if ctx.triggered_id == "choropleth":
        country_list = []
        for points in box_select["points"]:
            if "hovertext" in points:
                country_list.append(points["hovertext"])
        column = metrics_dict[selected_metric]
        df_data = df[df["country_name"].isin(country_list)]
        df_sum = df_data.groupby('year')[column].mean().reset_index()
        global_fig = px.line(df_sum, x='year', y=column, title=f'Selected Countries Trends in {selected_metric}')
        global_fig.update_layout(
            xaxis_title="",
            yaxis_title=f"{selected_metric}",
            coloraxis_showscale=False,
            template='plotly_white',
            hovermode='x unified',
            margin=dict(l=0, r=0, t=30, b=0),
        )
        return global_fig
    
    column = metrics_dict[selected_metric]
    if region == "World":
        df_sum = df.groupby('year')[column].mean().reset_index()
    elif region == "Asia":
        df_sum = asia_df.groupby('year')[column].mean().reset_index()
        
    elif region == "North America":
        df_sum = n_amer_df.groupby('year')[column].mean().reset_index()
        
    elif region == "South America":
        df_sum = s_amer_df.groupby('year')[column].mean().reset_index()
    
    elif region == "Africa":
        df_sum = afr_df.groupby('year')[column].mean().reset_index()
    else:
        df_sum = eu_df.groupby('year')[column].mean().reset_index()
    
    
    
    # Plot using px.line
    global_fig = px.line(df_sum, x='year', y=column, title=f'{region} Trends in {selected_metric}')
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
    n_amer_mean = n_amer_df.groupby("year")[column].mean().reset_index()
    s_amer_mean = s_amer_df.groupby("year")[column].mean().reset_index()  
    
    
    
    
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
        x=n_amer_mean["year"], y=n_amer_mean[column],
        hoverinfo='x+y',
        mode='lines',
        stackgroup='one',
        name="North America"
    ))
    
    fig.add_trace(go.Scatter(
        x=s_amer_mean["year"], y=s_amer_mean[column],
        hoverinfo='x+y',
        mode='lines',
        stackgroup='one',
        name="South America"
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