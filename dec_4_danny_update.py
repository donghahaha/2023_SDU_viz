# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:34:36 2023

@author: caspe
"""
""" THIS VERSION INCLUDES:
        SOME RESTRUCTURING (WIP, FEEDBACK THX)
        TIME RANGE SLIDER
        COMPARISON TOOL REWORK (MULTI SELECT)
        REGION SELECT
        COULD NOT MAKE ICONS ON MAP WORK YET BUT MAYBE POsSIBLE TO COMBINE CHOROPLETH WITH SCATTER
            (BOTTOM PAGE)
        SOME OTHER STUFF. 

"""

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go


# Put some stuff here to mak layout less busy.
title = "INTERACTIVE MAP OF V-DEM DEMOCRACY INDEX BY COUNTRY"
title_style = {"text_align": "left"}

# You can replace this with a less.. heavy version of the csv.
df = pd.read_csv("V-Dem-CY-Full+Others-v13.csv")
df = df[["year",  # The desired columns from the csv.
         "country_text_id", 
         "country_name", 
         "v2eltrnout",
         "v2exbribe",
         "v2x_polyarchy",
         "v2exnamhos",
         "v2exnamhog"
         ]]

df = df[df["year"].isin(range(2000, 2023 + 1))] # Desired year range.
# df = df.sort_values(by="year", ascending=False) # Sorting by latest year first.

#%% so we don"t load the csv file again by accident
# Here begins the dash app
app = Dash(__name__, external_stylesheets = [dbc.themes.BOOTSTRAP])


app.layout = dbc.Container([
    
    # Main Title
    dbc.Row(
        html.H1(title),
        ),
    
    
    html.Br(),
    
    # The Choropleth Map Row
    dbc.Row([
        dbc.Col([
            html.P("Select a Metric"),
            # Metric Selector
            dcc.Dropdown(
                id="Democracy metric", 
                options=["Democracy Score", "Corruption Score", "Turnout"],
                value="Democracy Score",
                clearable=False
            ),
            html.Br(),
            html.Br(),
            # Region Selector
            html.P("Select a Region"),
            dcc.Dropdown(
                id="regions", 
                options=["World", "Europe", "Asia", "Africa", "North America", "South America"],
                value="World",
                clearable=False
            ),
            
            html.Br(),
            html.Br(),
            
            html.Div(id="test"),
            
            html.Br(),
            html.Br(),
            
            dcc.Slider(0, 1, value=1, id="value_slider")
            
            ], width=2),
            
           

        # Choropleth Map
        dbc.Col(html.Div(dcc.Graph(id="choropleth")), width=10),
        
        
            
    ]),
    
    # Click country graph & other 
    dbc.Row([
        dbc.Col(html.Div(dcc.Graph(id="select_country_graph"))),
        dbc.Col(html.Div(id="test_thingy"))
    ]),
    
    
    
    
    # Comparison Tool
    dbc.Row([
        html.P("Comparison Tool"),
        dcc.Dropdown(
        id="multi_compare", 
        options=[{"label": country, "value": country} for country in df["country_name"].drop_duplicates()], # Removes duplicate country entries
        value=["Denmark", "South Korea"], multi=True), # Multi allows for multiple selections.
        ]),
    
    # Compare Graph
    dbc.Row([
        dbc.Col(
            dcc.Graph(id="compare_graph"), width=10
        ),
        
    ]),
    
])



@app.callback(
    Output("select_country_graph", "figure"),
    Output("test_thingy", "children"),
    Input("Democracy metric", "value"),
    Input("choropleth", "clickData"),
    )

def update_select_country(selected_metric, selected_country):
    if selected_metric == "Turnout":
        selected_column = "v2eltrnout"
        legend_title = "Turnout"
    elif selected_metric == "Corruption Score":
        selected_column = "v2exbribe"
        legend_title = "Lack of Corruption"
    elif selected_metric == "Democracy Score":
        selected_column = "v2x_polyarchy"
        legend_title = "Democracy Score"
    

    if selected_country is not None:
        country_iso = selected_country["points"][0]["location"]
        country_name = df.loc[df["country_text_id"] == country_iso, "country_name"].iloc[0]
        fig_selected_country = px.line(
            df[df["country_text_id"] == country_iso].groupby("year")[selected_column].mean().reset_index(),
            x="year",  
            y=selected_column,    
            title=f"{legend_title}: {country_name}",
            markers=True,
        )
        fig_selected_country.update_traces(hovertemplate=None)
        fig_selected_country.update_layout(hovermode='x unified')
        fig_selected_country.update_xaxes(rangeslider_visible=True)
        fig_selected_country.update_layout(
            xaxis_title="Year", 
            yaxis_title=legend_title
        )    
        
        
    else:
        country_iso = "TUR"
        fig_selected_country = px.line(
            df[df["country_text_id"] == country_iso].groupby("year")[selected_column].mean().reset_index(),
            x="year",  
            y=selected_column, 
            title=f"{legend_title}: {df.loc[df['country_text_id'] == country_iso, 'country_name'].iloc[0]}",
            markers=True,
        )
        fig_selected_country.update_traces(hovertemplate=None)
        fig_selected_country.update_layout(hovermode='x unified')
        fig_selected_country.update_xaxes(rangeslider_visible=True)
        fig_selected_country.update_layout(
            xaxis_title="Year", 
            yaxis_title=legend_title
        )    
    

    fig_selected_country.update_yaxes(range=[df[selected_column].min(), df[selected_column].max()])        
        
    # This is just calculating the difference between first recorded year and last.
    drop_nan = df.loc[df[selected_column].notna()]
    last_val = drop_nan.loc[(df["country_text_id"] == country_iso), ["year", selected_column]].sort_values("year", ascending=False).iloc[0][selected_column]
    init_val = drop_nan.loc[(df["country_text_id"] == country_iso), ["year", selected_column]].sort_values("year").iloc[0][selected_column]
    difference = round((last_val - init_val), 3)

    
    return fig_selected_country, difference


@app.callback(
    Output("compare_graph", "figure"),
    Input("Democracy metric", "value"),
    Input("multi_compare", "value"),
)

def update_comparison(selected_metric, compare):
    
    if selected_metric == "Democracy Score":
        selected_column = "v2x_polyarchy"
        legend_title = "Democracy Score"
    elif selected_metric == "Corruption Score":
        selected_column = "v2exbribe"
        legend_title = "Lack of Corruption"
    else:
        selected_column = "v2eltrnout"
        legend_title = "Turnout"
    
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
    fig_comparison.update_xaxes(rangeslider_visible=True)
    
    fig_comparison.update_layout(
        xaxis_title="Year", 
        yaxis_title=legend_title
    )
    
    
    return fig_comparison


@app.callback(
    Output("value_slider", "min"),
    Output("value_slider", "max"),
    Input("Democracy metric", "value"),
)

def update_value_slider(metric):
    if metric == "Democracy Score":
        return 0, 1
    elif metric == "Corruption Score":
        return -4, 4
    else:
        return 0, 100
        
@app.callback(
    Output("choropleth", "figure"),
    Input("Democracy metric", "value"),
    Input("regions", "value"),
    Input("value_slider", "value"),
    Input("choropleth", "clickData")
)

def update_choropleth(selected_metric, selected_region, arbitrary_limit, clicked):
    if selected_metric == "Turnout":
        selected_column = "v2eltrnout"
        legend_title = "Turnout"
    elif selected_metric == "Corruption Score":
        selected_column = "v2exbribe"
        legend_title = "Lack of Corruption"
    elif selected_metric == "Democracy Score":
        selected_column = "v2x_polyarchy"
        legend_title = "Democracy Score"

    # Main choropleth graph
    choro_fig = px.choropleth(
        df,
        labels={selected_column: legend_title},
        locations="country_text_id",
        locationmode="ISO-3",
        color=selected_column,
        hover_name="country_name",
        hover_data={"v2exnamhos": True, "v2exnamhog": True, "year": True},
        color_continuous_scale='Blues',
        scope=selected_region.lower(),
        animation_frame="year"
    )
    

    choro_fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>"
    )

    choro_fig.update_geos(
        coastlinecolor="Black",
        landcolor="white",
        countrycolor="white",
        oceancolor="LightBlue",
        showcoastlines=True,
        showland=True,
        showcountries=True,
        showocean=True,
        showlakes=False,
        showframe=True,
    )

    choro_fig.update_layout(
        dragmode=False,
        coloraxis_colorbar=dict(len=0.7, thickness=20, title=""),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )

    # This user thingy doesn't work properly. 
    df["user_limit"] = df[selected_column].ge(arbitrary_limit)
    scatter_icons = go.Figure(
        go.Scattergeo(
            locations=df["country_text_id"],
            locationmode="ISO-3",
            text=df["country_name"],
            marker=dict(
                color=df["user_limit"].map({True: "white", False: "black"}),
                size=5,
                symbol=df["user_limit"].map({True: "circle", False: "square"}),
                opacity=1,
            ),
            mode="markers",
        )
    )
    
    scatter_icons.update_traces(
        hovertemplate="<b>%{text}</b><br>"
    )


    scatter_icons.update_geos(
        projection_type="natural earth", 
        visible=False
    )

    scatter_icons.update_layout(
        dragmode=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )
    
    # Adds click functionality of highlighting a country.
    # If a country is clicked, this layer is added to the model.
    # This choropleth has only one location (selected) and a different color scale.
    if clicked:
        new_fig = go.Choropleth(
            locations=[clicked["points"][0]["location"]], 
            z=[1], # So that country is maximum purple. Or whichever other color.
            locationmode="ISO-3",
            colorscale=[[0, "purple"]],
            showscale=False  # Hides the color scale.
        )
      
        choro_fig.add_traces(scatter_icons.data)
        choro_fig.add_traces(new_fig)
        
        return choro_fig
    
    
    # Combine choropleth and scatter plot
    
    choro_fig.add_traces(scatter_icons.data)

    return choro_fig

# This callback is just for testing the hover. 
# I was thinking of implementing a highlight, but that seemed a little hard.
@app.callback(
    Output("test", "children"),
    Input("choropleth", "hoverData"),
)

def do_something(hovering):
    if not hovering:
        return no_update
    elif hovering and "points" in hovering and hovering["points"]:
        if "hovertext" in hovering["points"][0]:
            iso = hovering["points"][0]["hovertext"]
            return iso
        elif "text" in hovering["points"][0]:
            iso = hovering["points"][0]["text"]
            return iso
    else:
        iso = hovering["points"][0]["hovertext"]

        return iso

    
app.run_server(debug=False) # Can also set to true, but doesn"t work for me for some reason.