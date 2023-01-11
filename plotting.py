import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.figure_factory as ff



def save_data(data, fpath)-> None:
    import pandas as pd
    import plotly
    import plotly.graph_objects as go
    import json

    if type(data) == pd.DataFrame:
        data.to_pickle(fpath)
    elif type(data) == go.Figure:
        plotly.io.write_json(data, fpath, pretty=True)
    elif type(data) == dict:
        with open(fpath, 'w') as fp:
            json.dump(data, fp)
    else:
        raise TypeError("Save is not supported for {} datatype".format(type(data)))

def load_data(type, fpath)->None:
    import pandas as pd
    import plotly
    import plotly.graph_objects as go
    import json

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





def unique_interval_values(intervals)->list:
    """
    Function to return unique values from the interval
    """
    lst = intervals.to_list()
    temp = [k.left for k in lst]
    temp.append(lst[-1].right)
    return temp



def plot_fd(fd_data, name=None, verbose=False)-> go.Figure:
    """
    Plotting function to plot all the trajectories from the simulated data.
    Function fetches data from the csv files for FCD data and traffic lights information.
    Returns a plotly figure.
    """

    ## Plot ###
    if verbose:
        print("[FD-PLOT] Plotting ...", end=" ")
    fig = make_subplots(rows=1, cols=3)
    ## Speed/Density plot
    fig.add_trace(go.Scatter(x=fd_data["density"],
                            y=fd_data["speed"],
                            mode="markers",
                            name="Speed vs Density" if name==None else name,
                            customdata=fd_data["SCNFolder"],
                            hovertemplate='Density: %{x}<br>Speed: %{y}<br>SIM: %{customdata}<extra></extra>'),
                            row=1, col=1)

    ## Flow/Density plot
    fig.add_trace(go.Scatter(x=fd_data["density"],
                            y=fd_data["flow"],
                            mode="markers",
                            name="Flow vs Density" if name==None else name,
                            customdata=fd_data["SCNFolder"],
                            hovertemplate='Density: %{x}<br>Flow: %{y}<br>SIM: %{customdata}<extra></extra>'),
                            row=1, col=2)

    ## Speed/Flow
    fig.add_trace(go.Scatter(x=fd_data["flow"],
                            y=fd_data["speed"],
                            mode="markers",
                            name="Speed vs Flow" if name==None else name,
                            customdata=fd_data["SCNFolder"],
                            hovertemplate='Flow: %{x}<br>Speed: %{y}<br>SIM: %{customdata}<extra></extra>'),
                            row=1, col=3)

    fig.update_xaxes(title_text='Density [veh/m]', row=1, col=1)
    fig.update_xaxes(title_text='Density [veh/m]', row=1, col=2)
    fig.update_xaxes(title_text='Flow [veh/s]', row=1, col=3)
    fig.update_yaxes(title_text='Speed [m/s]', row=1, col=1)
    fig.update_yaxes(title_text='Flow [veh/s]', row=1, col=2)
    fig.update_yaxes(title_text='Speed [m/s]', row=1, col=3)
    fig.update_layout(title="Fundamental Diagrams", title_x=0.5,
                     margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
    if verbose:
        print("Done")

    return fig



def plot_trajectories(carData, probeData, title,
                      showCamBand=False, verbose=False, ctmPlot=False,
                      grid=None,
                      cam_view=0,
                      probeMinViewDistance=0,
                      trafficLight = None
                        )->go.Figure:
    """
    Function to plot trajectories of vehicles.
    It plots car trajectories given by carData and probe vehicle trajectory given by probeData.
    """

    ### Plot ###
    if verbose:
        print("[PLOT] Plotting ... ", end=" ")
    # Plotting the trajectoies of cars.
    fig = px.line(carData, x="timestep_time", y="vehicle_x", color='vehicle_id', markers=True)
    fig.update_traces(line=dict(width=2), marker=dict(size=5))

    # Plotting the probe vehicle trajectory
    fg  = px.line(probeData, x="timestep_time", y="vehicle_x", color="vehicle_id", markers=True,
                    line_dash_sequence=['longdashdot' for _ in range(len(probeData))])
    fg.update_traces(line=dict(width=2), marker=dict(size=5))

    for trace in fg.data:
        trace['line']['color']="#ff0000"
        fig.add_trace(trace)

    # Plot Space-time Grid onto the trajectories
    # Check if grid is initialized
    if grid is not None:
        for ts in unique_interval_values(grid["cell_time"]):
            fig.add_vline(x=ts, line_width=0.5, line_color="grey", line_dash="dash", name="T-{}".format(ts))
        for sp in unique_interval_values(grid["cell_space"]):
            fig.add_hline(y=sp, line_width=0.5, line_color="grey", line_dash="dash", name="X-{}".format(sp))
    else:
        raise ValueError("SPACE-TIME Grid is not initialized !!!")


    # Add visulization on the cam-view band of the probe vehicle
    if showCamBand:
        for idx in probeData.index:
            fig.add_shape(type="rect",
                x0=probeData.loc[idx]["timestep_time"]-0.125,
                y0=probeData.loc[idx]["vehicle_x"]-cam_view,
                x1=probeData.loc[idx]["timestep_time"]+0.125,
                y1=probeData.loc[idx]["vehicle_x"]-probeMinViewDistance,
                line=dict(
                color="Red",
                    width=.1,
                ),
                fillcolor="Red",
                opacity=0.1,
            )

    if trafficLight is not None:
        fig.add_trace(go.Scatter(x=trafficLight["time"], y=[0 for itr in range(trafficLight.shape[0])],
                        mode="markers",
                        marker=dict(color=trafficLight["start"]),
                        showlegend=False))
        fig.add_trace(go.Scatter(x=trafficLight["time"], y=[101 for itr in range(trafficLight.shape[0])],
                        mode="markers",
                        marker=dict(color=trafficLight["end"]),
                        showlegend=False))



    fig.update_xaxes(title_text='Time [sec]', showgrid=False)
    fig.update_yaxes(title_text='Space [m]', showgrid=False, range=[-1, 101])
    fig.update_layout(title=title, title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    if ctmPlot:
        fig.add_hrect(y0=-20, y1=0, line_width=0, fillcolor="red", opacity=0.1)
        fig.add_hrect(y0=100, y1=120, line_width=0, fillcolor="red", opacity=0.1)
        fig.add_vrect(x0=probeData["timestep_time"].min(),
                        x1=probeData["timestep_time"].min()+grid["deltaT"],
                        line_width=0, fillcolor="green", opacity=0.1)
        fig.update_yaxes(title_text='Space [m]', showgrid=False, range=[-21, 121])



    if verbose:
        print("Done!")
        print("Total Number of vehicles observed: ", len(set(carData["vehicle_id"])))

    return fig


def plot_aggregated_density_matrix(matrix, title, verbose=False)->go.Figure:
    """
    Function to plot the heatmap for the density matrix from the space-time-density matrix.
    """
    ### Plot ###
    if verbose:
        print("[PLOT] Plotting ... ", end=" ")
    fig = go.Figure(data=go.Heatmap(
                x=[str(idx) for idx in matrix.columns],
                y=[str(idx) for idx in matrix.index],
                xgap=2,
                ygap=2,
                z = matrix.values,
                zmin=0,
                zmax=0.2,
                colorbar=dict(title='Density'),
                hovertemplate='Time: %{x}<br>Space: %{y}<br>Density: %{z}<extra></extra>'
            ))

    fig.update_xaxes(title_text='Time [sec]')
    fig.update_yaxes(title_text='Space [m]')
    fig.update_layout(title=title, title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    if verbose: print("Done!")
    return fig


def plot_fitness(repetation:int, num_generations:int, generation_fitness)->go.Figure:
    fig = go.Figure()
    for itr in range(repetation):
        fig.add_trace(go.Scatter(x=list(range(num_generations)), y=list(generation_fitness[:, itr]),
                                mode="lines",
                                name="Rep-{}".format(itr),
                                line = dict(width=1)))

    fig.add_trace(go.Scatter(x=list(range(num_generations)), y=generation_fitness.mean(axis=1),
                                mode="lines",
                                name="Mean-Fit",
                                line = dict(color='firebrick', width=4, dash='dot')))
    fig.update_xaxes(title_text='Generations')
    fig.update_yaxes(title_text='Fitness')
    fig.update_layout(title="Fitness vs Generation for {} Repetitions".format(repetation),
                      title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig

def plot_resiual_matrix(actual, predicted):
     ## Difference in the real and model values
    residuals = np.abs(actual.values - predicted.values).astype(float)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
                            x=[str(idx) for idx in predicted.columns],
                            y=[str(idx) for idx in predicted.index],
                            xgap=2,
                            ygap=2,
                            z = residuals,
                            text=np.round(residuals, 3),
                            texttemplate="%{text}",
                            textfont={"size":10},
                            zmin=residuals.min(),
                            zmax=residuals.max(),
                            colorscale="Blues",
                            colorbar=dict(title='Residuals'),
                            hovertemplate='Time: %{x}<br>Space: %{y}<br>Residual: %{z}<extra></extra>'))

    fig.update_xaxes(title_text='Time [sec]')
    fig.update_yaxes(title_text='Space [m]')
    fig.update_layout(title="Residuals", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def add_charateristic_line_to_fd(fig:go.Figure, vf:float, rho_j:float, rho_c:float, mode="lines")->go.Figure:
    # Define Triangular FD

    from cell_transmission_model import FD
    FD.parameters(vf=vf, rho_j=rho_j, rho_c=rho_c)

    densities = np.linspace(0, FD.rho_j, num=100)
    flows     = np.array([FD.flow(rho) for rho in densities])
    speed     = flows/densities

    # Plot
    # speed-density
    fig.add_trace({"mode": mode,
                    "line": {'width':5},
                    "x": densities,
                    "y": speed,
                    "type": "scatter",
                    "xaxis": "x",
                    "yaxis": "y",
                    "name": "Tri FD(Speed-vs-Density)" if mode=="markers" else "AVG.Speed vs AVG.Density"})
    # flow-density
    fig.add_trace({"mode": mode,
                    "line": {'width':5},
                    "x": densities,
                    "y": flows,
                    "type": "scatter",
                    "xaxis": "x2",
                    "yaxis": "y2",
                    "name": "Tri FD(Flow-vs-Density)" if mode=="markers" else "AVG.Flow vs AVG.Density"})
    # speed-flow
    fig.add_trace({"mode": mode,
                   "line": {'width':5},
                   "x": flows,
                   "y": speed,
                   "type": "scatter",
                   "xaxis": "x3",
                   "yaxis": "y3",
                   "name": "Tri FD(Speed-vs-Flow)" if mode=="markers" else "AVG.Speed vs AVG.Flow"})
    return fig


def add_charateristic_scenario_data_to_fdPlot(fig:go.Figure, data:pd.DataFrame)->go.Figure:
    # Define Triangular FD

    # Plot
    # speed-density
    fig.add_trace({"mode": "markers",
                    "marker": {"color": "darkslateblue"},
                    "line": {'width':2},
                    "x": data["Density"],
                    "y": data["Speed"],
                    "type": "scatter",
                    "xaxis": "x",
                    "yaxis": "y",

                    "name": "Scenario(Speed-vs-Density)"})
    # flow-density
    fig.add_trace({"mode": "markers",
                    "marker": {"color": "darkred"},
                    "line": {'width':2},
                    "x": data["Density"],
                    "y": data["Flow"],
                    "type": "scatter",
                    "xaxis": "x2",
                    "yaxis": "y2",
                    "name": "Scenario(Flow-vs-Density)"})
    # speed-flow
    fig.add_trace({"mode": "markers",
                   "line": {'width':5},
                   "marker": {"color": "seagreen"},
                   "x": data["Flow"],
                   "y": data["Speed"],
                   "type": "scatter",
                   "xaxis": "x3",
                   "yaxis": "y3",
                   "name": "Scenario(Speed-vs-Flow)"})

    fig.update_layout(showlegend=True)

    return fig





def animate_fd_plot_with_characteristics(SIMULATION_DATA_FOLDER):
    import pandas as pd
    import plotting as plt
    import plotly.graph_objects as go
    import os
    ## Checking the FD with calculated RHO_C values ##
    itr = 0
    # Read GA result file
    result = pd.read_csv("data/GAResult.csv", sep=";")
    result["RHO_C_update"] = result["BestRHO_C"].expanding().mean()

    # Data for the Traingular FD
    df = result[['SimulationFolder', 'Free Flow Speed [m/sec]', 'Jam Density', 'BestRHO_C', 'RHO_C_update']]
    for idx, data in df.iterrows():
        # Load the FD
        fd_plot = plt.load_data("plotly", os.path.join(SIMULATION_DATA_FOLDER, data['SimulationFolder'], "plot_scenario_fd.json"))
        # Add fd to plot

        itr+=1
        if itr>0:
            break


    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["data"] = fd_plot["data"]
    fig_dict["layout"] = fd_plot["layout"]
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "EXP:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # Add frames to animate
    for idx, data in df.iterrows():
        frame = {"data": [], "name": str(data["SimulationFolder"]), "layout": go.Layout(title_text="RHO_C {}".format(data['BestRHO_C']))}
        # Load the FD
        fd_plot = plt.load_data("plotly", os.path.join(SIMULATION_DATA_FOLDER, data['SimulationFolder'], "plot_scenario_fd.json"))
        # Add fd to plot
        fd_plot = plt.add_charateristic_line_to_fd(fd_plot,
                                           vf=data['Free Flow Speed [m/sec]'],
                                           rho_j=data['Jam Density'],
                                           rho_c=data['BestRHO_C'])

        fd_plot = plt.add_charateristic_line_to_fd(fd_plot,
                            vf=data['Free Flow Speed [m/sec]'],
                            rho_j=data['Jam Density'],
                            rho_c=data['RHO_C_update'],
                            mode="lines")

        frame["data"] = fd_plot["data"]
        fig_dict["frames"].append(frame)

        # Slider
        slider_step = {"args": [
                            [data["SimulationFolder"]],
                            {"frame": {"duration": 300, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 300}}
                            ],
                        "label": data["SimulationFolder"],
                        "method": "animate"}
        sliders_dict["steps"].append(slider_step)




    fig_dict["layout"]["sliders"] = [sliders_dict]
    fig = go.Figure(fig_dict)
    fig['layout']['paper_bgcolor'] = 'rgba(255,255,255,255)'
    fig.layout.yaxis2["range"] = (0, 0.7)
    return fig




def plot_fundamental_diagram(plotdata: pd.DataFrame,
                             X_data:str, X_title:str,
                             Y_data:str, Y_title:str,
                             plotTitle:str,
                             hightIDX=None,
                             )->go.Figure():
    data = plotdata.copy()
    data.SCN, uniques = pd.factorize(data.SCN)
    fig = go.Figure()
    for scn, frame in data.groupby('SCN'):
        fig.add_trace(go.Scatter(
                                x=frame[X_data], y=frame[Y_data],
                                mode="markers",
                                marker=dict(size=8,
                                            line=dict(color="Black", width=1),
                                            color=scn,
                                            colorscale="Rainbow"),
                                customdata = data['SimulationFolder'],
                                hovertemplate='<b>%{customdata}</b><br><br> X: %{x:.3f} <br> Y: %{y:.3f} <extra></extra>',
                                name=uniques[scn]
                            ))
    if hightIDX is not None:
        fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[data.loc[hightIDX, X_data]],
            y=[data.loc[hightIDX, Y_data]],
            marker_symbol="x",
            marker_line_color="midnightblue", marker_color="red",
            marker_line_width=2, marker_size=10,
            customdata = data.index.to_list(),
            hovertemplate='<b>%{customdata}</b><br><br> X: %{x:.3f} <br> Y: %{y:.3f} <extra></extra>',
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
    fig.update_layout(title=plotTitle, title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def init_rmse_plot():
    fig = go.Figure()
    # Sahding for scenario
    # fig.add_vrect(x0=0, x1=20, annotation_text="Scenario-1", annotation_position="top left",
    #               fillcolor="green", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=20, x1=40, annotation_text="Scenario-2", annotation_position="top left",
    #               fillcolor="blue", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=40, x1=56, annotation_text="Scenario-3", annotation_position="top left",
    #               fillcolor="cyan", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=56, x1=71, annotation_text="Scenario-4", annotation_position="top left",
    #               fillcolor="red", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=71, x1=91, annotation_text="Scenario-5", annotation_position="top left",
    #               fillcolor="yellow", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=91, x1=107, annotation_text="Scenario-6", annotation_position="top left",
    #               fillcolor="lime", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=107, x1=128, annotation_text="Scenario-7", annotation_position="top left",
    #               fillcolor="dimgrey", opacity=0.05, line_width=0)
    # fig.add_vrect(x0=128, x1=147, annotation_text="Scenario-8", annotation_position="top left",
    #               fillcolor="yellowgreen", opacity=0.05, line_width=0)
    fig.update_xaxes(title_text="Experiment", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text="RMSE", showline=True, linewidth=2, linecolor='black', mirror=True, range=[0, 0.04])
    fig.update_layout(title="RMSE vs Experiment", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_rmse(fig, x_data, y_data, name) -> go.Figure:
    fig.add_trace(go.Scatter(x=x_data,
                             y=y_data,
                             mode="lines+markers",
                             marker=dict(size=4),
                             fill='tozeroy',
                             hovertemplate='EXP: %{x}<br>RMSE: %{y}<extra></extra>',
                             name=name))
    return fig

def add_marker(fig, x, y):
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[x],
            y=[y],
            marker_symbol="x",
            marker_line_color="midnightblue", marker_color="white",
            marker_line_width=2, marker_size=10,
            showlegend=True,
            name="marker"
        ))
    return fig


def density_distribution(hist_data, group_labels)->go.Figure:
    # Create distplot with custom bin_size
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.01, show_rug=True, show_hist=False)

    fig.update_xaxes(title_text='Densities')
    fig.update_yaxes(title_text='Count')
    fig.update_layout(title="Distribution of Lane Densities",
                    title_x=0.5,
                    margin={"r":10,"t":40,"l":10,"b":10},
                    plot_bgcolor='rgba(0,0,0,0)')

    fig.data[0]['line']['width'] = 5
    fig.data[0]['line']['dash'] = 'dashdot'
    fig.data[0]['line']['color'] = 'black'

    return fig

def r2ErrorPlot(actual, predicted)->go.Figure:

    ## Fit a linear regression line between the actual and predicted
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression()
    regr.fit(predicted.reshape(-1, 1), actual)
    r2Score = regr.score(predicted.reshape(-1, 1), actual)

    ### PLOT the scatter points
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predicted,
                             y=actual,
                             mode="markers",
                             marker=dict(size=4, opacity=0.6),
                             hovertemplate='Actual: %{x}<br>Predicted: %{y}<extra></extra>'))

    ## Plot the regression line
    x_range = np.linspace(predicted.min(), predicted.max(), 100)
    y_range = regr.predict(x_range.reshape(-1, 1))
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name=f'R2 {r2Score:.3f}'))

    fig.update_xaxes(title_text="Predicted Densities [veh/m]", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text="Actual Densities [veh/m]", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(title=f"R2 Score {r2Score:.4f}", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig


def plot_rho_c_optimization(exp, optiaml_rhoc, cal_rhoc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exp,
                             y=optiaml_rhoc,
                             mode="lines+markers",
                             marker=dict(size=4),
                             hovertemplate='RHO_C*: %{y}<extra></extra>',
                             name="RHO_C*"))

    fig.add_trace(go.Scatter(x=exp,
                             y=cal_rhoc,
                             mode="lines+markers",
                             marker=dict(size=4),
                             hovertemplate='RHO_C: %{y}<extra></extra>',
                             name="RHO_C (GA)"))

    fig.add_vrect(x0=0, x1=20, annotation_text="Scenario-1", annotation_position="top left",
                  fillcolor="green", opacity=0.05, line_width=0)
    fig.add_vrect(x0=20, x1=40, annotation_text="Scenario-2", annotation_position="top left",
                  fillcolor="blue", opacity=0.05, line_width=0)
    fig.add_vrect(x0=40, x1=56, annotation_text="Scenario-3", annotation_position="top left",
                  fillcolor="cyan", opacity=0.05, line_width=0)
    fig.add_vrect(x0=56, x1=71, annotation_text="Scenario-4", annotation_position="top left",
                  fillcolor="red", opacity=0.05, line_width=0)
    fig.add_vrect(x0=71, x1=91, annotation_text="Scenario-5", annotation_position="top left",
                  fillcolor="yellow", opacity=0.05, line_width=0)
    fig.add_vrect(x0=91, x1=107, annotation_text="Scenario-6", annotation_position="top left",
                  fillcolor="lime", opacity=0.05, line_width=0)
    fig.add_vrect(x0=107, x1=128, annotation_text="Scenario-7", annotation_position="top left",
                  fillcolor="dimgrey", opacity=0.05, line_width=0)
    fig.add_vrect(x0=128, x1=147, annotation_text="Scenario-8", annotation_position="top left",
                  fillcolor="yellowgreen", opacity=0.05, line_width=0)

    fig.update_xaxes(title_text="EXP", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text="RHO_C* [veh/m]", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(title=f"Optimal value of RHO_C", title_x=0.5, hovermode="x", margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def plot_fcd_data(sc_object, verbose=False):
    """
    Plotting function to plot all the trajectories from the simulated data.
    Function fetches data from the csv files for FCD data and traffic lights information.
    Returns a plotly figure.
    """
    import plotly.graph_objects as go
    import plotly.express as px

    ### Plot Data ###
    # Probe vehicle only in the opposite lane to probe
    carObs = sc_object.get_observed_car_data()
    # All probe vehicels
    probeVeh = sc_object.get_probe_vehicle_data("all")
    # Read Traffic lights data
    traffic_start, traffic_end = sc_object.read_traffic_lights_data()

    ### Plot ###
    if verbose:
        print("[FCD-PLOT] Plotting ... ", end=" ")
    # Plotting observed cars
    fig = px.line(carObs, x="timestep_time", y="vehicle_x", color='vehicle_id', markers=True)
    fig.update_traces(line=dict(width=2), marker=dict(size=3, opacity=0.5), opacity=0.5)

    # Plotting probe vehicles
    fg  = px.line(probeVeh,
                    x="timestep_time",
                    y="vehicle_x",
                    color="vehicle_id",
                    line_dash_sequence=['longdashdot' for _ in range(len(probeVeh))])

    for trace in fg.data:
        trace['line']['color']="#ff0000"
        fig.add_trace(trace)

    # Plotting Traffic light signal
    fig.add_trace(go.Scatter(x=traffic_start["tlsState_time"], y=traffic_start["loc"],
                            mode="markers",
                            marker=dict(color=traffic_start["color"]),
                            showlegend=False))

    fig.add_trace(go.Scatter(x=traffic_end["tlsState_time"], y=traffic_end["loc"],
                            mode="markers",
                            marker=dict(color=traffic_end["color"]),
                            showlegend=False))

    fig.update_xaxes(title_text='Time [sec]')
    fig.update_yaxes(title_text='Space [m]')
    fig.update_layout(title="Vehicle Trajectories", title_x=0.5)
    if verbose:
        print("Done!")

    return fig



def plot_fd2(params, SIMULATION_DATA_FOLDER):
    import os
    # Collect traffic states from each simulation
    dirlist = [filename for filename in os.listdir(SIMULATION_DATA_FOLDER) if os.path.isdir(os.path.join(SIMULATION_DATA_FOLDER,filename))]
    dirlist.sort()
    fdTRAJ = pd.DataFrame()
    for simulatedDatafoler in dirlist:
        data_folder = os.path.join(SIMULATION_DATA_FOLDER, simulatedDatafoler)
        # read pickle df
        fdData = pd.read_pickle(os.path.join(data_folder, 'info_trafficStatesAtTimestep.pkl'))
        fdData["SCNFolder"] = [simulatedDatafoler for itr in range(fdData.shape[0])]
        fdTRAJ = pd.concat([fdTRAJ, fdData], ignore_index=True)
    # Plotting FD
    fdTRAJ = fdTRAJ.rename(columns={"Density": "density", "Flow": "flow", "Speed": "speed"})
    fig = plot_fd(fdTRAJ)
    fig = add_charateristic_line_to_fd(fig,
                                        vf=params['vf'],
                                        rho_c=params['rho_c'],
                                        rho_j=params['rho_j'])
    return fig


def residual_plot(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = np.arange(0, df.shape[0], step=1),
                             y = df['residual'],
                            mode='markers',
                            customdata = [d for d in zip(df['Actual'], df['Calc'])],
                            hovertemplate='Actual: %{customdata[0]:.4f}<br>Calc: %{customdata[1]:.4f}<br>Residual: %{y}<extra></extra>'),)
    fig.add_hline(y=0.0)
    fig.update_xaxes(title_text="observations [#]", showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(title_text="residuals ['Actual' - 'Calc']", range=[-0.05, 0.05], showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_layout(title=f"Residual Plot", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0})
    fig.update_layout(hovermode="x unified")
    return fig



## 3D plot for Fitness ##
# import pandas as pd
# from cell_transmission_model import CTM
# from sklearn.metrics import r2_score, mean_squared_error
# import numpy as np
# import plotting as plt
# import plotly.graph_objects as go

# densFul = pd.read_csv("data/densTuplecar.csv")
# densPar = pd.read_csv("data/densTuplepartial.csv")
# grid = dict(deltaX=20, deltaT=2)
# ctm = CTM(grid)


# # SPlit the VF and RHO_C values to give to CTM
# plotData = pd.DataFrame()
# vf_step = 0.2
# rho_step = 0.005
# for vf in np.arange(start=5, stop=10, step=vf_step):
#         for rho_c in np.arange(start=0.005, stop=0.075, step=rho_step):

#             params = {'vf'   : vf,
#                     'rho_c': rho_c,
#                     'rho_j': 0.15}

#             # Run CTM for each row
#             predFul    = densFul.apply(lambda data: ctm.simulate_single(rho_prv=data['x-1'], rho_cur=data['x'], rho_nxt=data['x+1'], fdParams=params), axis=1)
#             predPar = densPar.apply(lambda data: ctm.simulate_single(rho_prv=data['x-1'], rho_cur=data['x'], rho_nxt=data['x+1'], fdParams=params), axis=1)
#             # Goodness of fit values
#             fitFul = r2_score(densFul['t+1'], predFul)
#             fitPar = r2_score(densPar['t+1'], predPar)
#             plotData = pd.concat([plotData, pd.DataFrame({'vf': vf, 'rho_c':rho_c, 'r2_full':fitFul, 'r2_par':fitPar}, index=[0])], ignore_index=True)


# fig = go.Figure()
# fig.add_trace(go.Scatter3d(x=plotData['vf'],
#                             y=plotData['rho_c'],
#                             z=plotData['r2_full'],
#                             mode="markers",
#                             marker=dict(size=5),
#                             name="R2 Score - Full"))
# fig.add_trace(go.Scatter3d(x=plotData['vf'],
#                             y=plotData['rho_c'],
#                             z=plotData['r2_par'],
#                             mode="markers",
#                             marker=dict(size=5),
#                             name="R2 Score - Partial"))

# fig.update_layout(scene = dict(
#                     xaxis_title='VF',
#                     yaxis_title='RHO_C',
#                     zaxis_title='R2 Score'),
#                 title="Fitness plot for differnt FD parameters",
#                 title_x=0.5,
#                 margin=dict(r=10, b=10, l=10, t=40))

# fig.show()
# fig.write_html("Fit-compare-with-density-tuples.html")