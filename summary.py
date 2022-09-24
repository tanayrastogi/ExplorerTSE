"""
The script is to make the summary plots using Scenario Traffic States and RMSE result from GA. 
"""
import pandas as pd
import plotly.graph_objects as go 
import json

def plot_fundamental_diagram(data: pd.DataFrame, 
                            X_data:str, X_title:str,
                            Y_data:str, Y_title:str,
                            error_type:str, 
                            plotTitle:str,
                            hightIDX:int,
                            scaleColor="Reds",
                            )->go.Figure():
    """
    Plots FD scatter plot with marker colors as the error
    """
    expName = data["Scenario"] +"-"+ data["ProbeID"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                            x=data[X_data], y=data[Y_data],
                            mode="markers",
                            marker=dict(size=12,
                                        color=data[error_type],
                                        line=dict(color="Black", width=1),
                                        showscale=True,
                                        cmin=0,
                                        cmax=data[error_type].max(),
                                        colorbar=dict(title=error_type),
                                        colorscale=scaleColor),
                            customdata = [(nm, error) for nm, error in  zip(expName, data[error_type].round(3))],
                            hovertemplate='<b>%{customdata[0]}</b><br><br> X: %{x:.3f} <br> Y: %{y:.3f} <br> Error: %{customdata[1]} <extra></extra>',
                            showlegend=False
                        ))
    
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[data.loc[hightIDX, X_data]],
            y=[data.loc[hightIDX, Y_data]],
            marker_symbol="x",
            marker_line_color="midnightblue", marker_color="purple",
            marker_line_width=2, marker_size=15,
            customdata = [data.loc[hightIDX, error_type].round(3)],
            hovertemplate='X: %{x:.3f} <br> Y: %{y:.3f} <br> Error: %{customdata} <extra></extra>',
            showlegend=False
        ))




    fig.add_vline(x=data[X_data].max()/2,
                line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=data[Y_data].max()/2,
                line_width=1, line_dash="dash", line_color="black")

    fig.update_xaxes(title_text=X_title,
                    showline=True, linewidth=2, linecolor='black', mirror=True)

    fig.update_yaxes(title_text=Y_title,
                    showline=True, linewidth=2, linecolor='black', mirror=True)

    fig.update_layout(title=plotTitle, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
    return fig


def load_data(type, fpath)->None:
    if type=="plotly":    
        with open(fpath, 'r') as f:
            v = json.loads(f.read())
        return go.Figure(data=v['data'], layout=v['layout'])
    elif type=="dataframe":
        return pd.read_pickle(fpath)
    elif type=="json":
        with open(fpath, 'r') as fp:
            return json.load(fp)
    else:
        raise TypeError("Datatype is not available to load")





if __name__ == "__main__":
        

    GA_RES_CSV = "ResultsNew.csv"
    TRAFFIC_STATES = "SimulationDataNew/ScenarioData.csv"


    ### READ CSV ###
    result = pd.read_csv(GA_RES_CSV, sep=";", index_col=0)
    result["fullnames"] = result.apply(lambda x: x["Scenario"]+"_"+x["ProbeID"]+"_"+x["Timestamp"], axis=1)
    result.set_index("fullnames", drop=True, inplace=True)

    states = pd.read_csv(TRAFFIC_STATES, sep=";", index_col=0)

    ### MERGE ###
    data = states.join(result)
    data["RMSE-plot"] = data["RMSE Error - Masked"]*20
    data["MAE-plot"] = data["MAE Error - Masked"]*20



    ### PLOT ###
    ERROR_TYPE = "MAE-plot"
    X_DATA = "Car Density"
    X_AXIS = "Traffic Density [veh/m]"
    Y_DATA = "Car Flow"
    Y_AXIS = "Traffic Flow [veh/s]"
    TITLE  = "MAE for Flow-vs-Density diagram"

    expName = data["Scenario"] +"-"+ data["ProbeID"] 

    fig = go.Figure()
    fig.add_trace(go.Scatter(
                            x=data[X_DATA], y=data[Y_DATA],
                            mode="markers",
                            marker=dict(size=12,
                                        color=data[ERROR_TYPE],
                                        line=dict(color="Black", width=1),
                                        showscale=True,
                                        cmin=0,
                                        cmax=3.5,
                                        colorbar=dict(title="MAE"),
                                        colorscale="Reds"),
                            customdata = [(nm, error) for nm, error in  zip(expName, data[ERROR_TYPE].round(3))],
                            hovertemplate='<b>%{customdata[0]}</b><br><br> X: %{x:.3f} <br> Y: %{y:.3f} <br> Error: %{customdata[1]} <extra></extra>'
                        ))

    fig.add_vline(x=0.25,
                line_width=1, line_dash="dash", line_color="black")
    fig.add_hline(y=data[Y_DATA].max()/2,
                line_width=1, line_dash="dash", line_color="black")
    fig.update_xaxes(title_text=X_AXIS, range=[0, 0.5],
                    showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text=Y_AXIS,
                    showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(margin={"r":20,"t":40,"l":20,"b":10})
    fig.update_layout(title=TITLE, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
    fig.show()



    # #####################################################
    # X_DATA = "Car Speed"
    # X_AXIS = "Traffic Speed [m/s]"
    # Y_AXIS = "RMSE [veh]"

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(
    #                         x=data[X_DATA], y=data[ERROR_TYPE],
    #                         mode="markers"))

    # fig.update_xaxes(title_text=X_AXIS, 
    #                  showline=True, linewidth=2, linecolor='black', mirror=True,
    #                  showgrid=True, gridwidth=1, gridcolor='LightGrey')
    # fig.update_yaxes(title_text=Y_AXIS,
    #                  showline=True, linewidth=2, linecolor='black', mirror=True,
    #                  showgrid=True, gridwidth=1, gridcolor='LightGrey')
    # fig.update_layout(margin={"r":20,"t":40,"l":20,"b":10})
    # fig.update_layout(title=TITLE, title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')
    # fig.show()



    #####################################################
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Scatter(
    #                         x=data["Car Density"], y=data["Car Flow"],
    #                         name="Flow vs Density",
    #                         mode="markers",
    #                         marker=dict(size=12,
    #                                     opacity=0.9,
    #                                     color=data["RMSE-plot"],
    #                                     line=dict(color="Black", width=1),
    #                                     showscale=True,
    #                                     cmin=0,
    #                                     cmax=round(data["RMSE-plot"].max(), 1),
    #                                     colorbar=dict(title="RMSE"),
    #                                     colorscale="Reds"),
    #                         customdata = [(nm, error) for nm, error in  zip(expName, data["RMSE-plot"].round(3))],
    #                         hovertemplate='<b>%{customdata[0]}</b><br><br> Density: %{x:.3f} <br> Flow: %{y:.3f} <br> Error: %{customdata[1]} <extra></extra>'
    #                     ), secondary_y=False)

    # fig.add_trace(go.Scatter(
    #                         x=data["Car Density"], y=data["Car Speed"],
    #                         name="Speed vs Density",
    #                         mode="markers",
    #                         marker=dict(size=12,
    #                                     opacity=0.9,
    #                                     color=data["RMSE-plot"],
    #                                     line=dict(color="Black", width=1),
    #                                     showscale=True,
    #                                     cmin=0,
    #                                     cmax=round(data["RMSE-plot"].max(), 1),
    #                                     colorbar=dict(title="RMSE"),
    #                                     colorscale="Blues"),
    #                         customdata = [(nm, error) for nm, error in  zip(expName, data["RMSE-plot"].round(3))],
    #                         hovertemplate='<b>%{customdata[0]}</b><br><br> Density: %{x:.3f} <br> Speed: %{y:.3f} <br> Error: %{customdata[1]} <extra></extra>'
    #                     ), secondary_y=True)

    # fig.update_xaxes(title_text="Traffic Density [veh/m]", range=[0, 0.5],
    #                  showline=True, linewidth=2, linecolor='black', mirror=True)
    # fig.update_yaxes(title_text="Traffic Flow [veh/s]", range=[0, 1.0],
    #                  secondary_y=False, 
    #                  showline=True, linewidth=2, linecolor='black', mirror=True)
    # fig.update_yaxes(title_text="Traffic Speed [m/s]", range=[0, 12.0],
    #                  secondary_y=True,
    #                  showline=True, linewidth=2, linecolor='black', mirror=True)
    # fig.update_layout(margin={"r":20,"t":40,"l":20,"b":10}, 
    #                  legend=dict(yanchor="top",
    #                              y=0.99,
    #                              xanchor="left",
    #                              x=0.5))
    # fig.data[0].marker.colorbar.x=-0.3
    # fig.data[1].marker.colorbar.x=1.1
    # fig.update_layout(title="RMSE PLOTS", title_x=0.5, plot_bgcolor='rgba(0,0,0,0)')

    # fig.add_vline(x=0.25,
    #               line_width=1, line_dash="dash", line_color="black")
    # fig.add_hline(y=data["Car Flow"].max()/2,
    #               line_width=1, line_dash="dash", line_color="black")
    # fig.show()



    #########################################################
