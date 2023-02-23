import os
import pandas as pd
import numpy as np 
import plotting as plt
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


MAIN_FOLDER            = "data"
SUMO_FOLDER            = os.path.join(MAIN_FOLDER, "SUMO_Traffic_Scenario")
SIMULATION_DATA_FOLDER = os.path.join(MAIN_FOLDER, "SimulationData")
SIMULATION_METATDATA   = os.path.join(MAIN_FOLDER, "SimulationMetadata.csv")
DENSITY_TUPLE_CAR      = os.path.join(MAIN_FOLDER, "densTuplecar.csv")
DENSITY_TUPLE_PARTIAL  = os.path.join(MAIN_FOLDER, "densTuplepartial.csv")
GA_DATA_FOLDER  = os.path.join(MAIN_FOLDER, "GAExpResult")
GA_METATDATA    = os.path.join(MAIN_FOLDER, "GAResult.csv")

def markdown(title, header='h1', color='red', showLine=True) -> None:
    if showLine:
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    if header=='h1': st.markdown(f"<h1 style='text-align: center; color: {color};'>{title}</h1>", unsafe_allow_html=True)
    if header=='h2': st.markdown(f"<h2 style='text-align: center; color: {color};'>{title}</h2>", unsafe_allow_html=True)
    if header=='h3': st.markdown(f"<h3 style='text-align: center; color: {color};'>{title}</h3>", unsafe_allow_html=True)
    if header=='h4': st.markdown(f"<h4 style='text-align: center; color: {color};'>{title}</h4>", unsafe_allow_html=True)
    if header=='h5': st.markdown(f"<h5 style='text-align: center; color: {color};'>{title}</h5>", unsafe_allow_html=True)
    if showLine:
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

def seperater_line():
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

# Show interactive table
def show_table(table):    
    gb = GridOptionsBuilder.from_dataframe(table)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_default_column(min_column_width=30) #Column width
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('single', use_checkbox=True) #Enable multi-row selection
    gridOptions = gb.build()

    # Result Table
    selected_data = AgGrid( 
                        table,
                        gridOptions=gridOptions,
                        data_return_mode='AS_INPUT',
                        update_mode='MODEL_CHANGED',
                        fit_columns_on_grid_load=True,
                        theme='streamlit', #Add theme color to the table
                        enable_enterprise_modules=True,
                        height=300,
                        width='50%',
                        reload_data=True
                    )

    selected_data = selected_data["selected_rows"]
    return selected_data




################################################################################################
                                    #### SIMULATED DATA ####
################################################################################################
def sumo_simulation_data() -> None:
    markdown("SUMO SIMULATION DATA")

    ############# DENS TUPLES ##############
    dataType = st.radio("Type of Tuple Data", ('Partial Matrix', 'Full Matrix'))
    if dataType == "Partial Matrix":
        densTup = pd.read_csv(DENSITY_TUPLE_PARTIAL, sep=",", header=0)
    elif dataType == "Full Matrix":
        densTup = pd.read_csv(DENSITY_TUPLE_CAR, sep=",", header=0)
    else:
        pass


    # SIMULATION METADATA TABLE
    data = pd.read_csv(SIMULATION_METATDATA, sep=";", header=0)
    data['SCN'] = data.apply(lambda x: x['SimulationFolder'].split("_")[0], axis=1)
    data['ProbeID'] = data.apply(lambda x: x['SimulationFolder'].split("_")[1], axis=1)
    col = data.columns.to_list()
    col = col[-2:] + col[:-2]
    data = data[col]
    selected_data = show_table(data)

    ### PLOTS ####
    import plotly.figure_factory as ff
    col1, col2 = st.columns(2)
    with col1: 
        ## Car Speed and Probe Speed distribution across the experiments ##
        hist_data = [data["Car Speed"].to_numpy(), data["Probe Speed"].to_numpy()]
        group_labels = ['Opposite Lane Speed', 'Moving Camera Speed']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=1, show_rug=False, curve_type="kde", histnorm='probability')
        fig.update_xaxes(title_text="Speed [m/sec]", range=[0, 10])
        fig.update_layout(margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig)

        import plotly.express as px 
        fig = px.scatter(data, x="Probe Speed", y="Car Speed", marginal_x="histogram", color="SCN", marginal_y="histogram", hover_data=["SimulationFolder"])
        fig.update_xaxes(title_text="Moving Camera Speed [m/sec]",
                        range=[0, 10], showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=1)
        fig.update_yaxes(title_text="Avg. Traffic Speed in Opposite Lane [m/sec]",
                        range=[0, 10], showline=True, linewidth=2, linecolor='black', mirror=True, row=1, col=1)
        fig.update_traces(marker=dict(size=8,
                                    line=dict(width=1,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout(title="Moving Camera Speed vs Lane Traffic Speed", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig)

    with col2: 
        ## Car Density distribution across the experiments ##
        hist_data = [data["Car Density"].to_numpy()]
        group_labels = ['Opposite Lane Density']
        fig = ff.create_distplot(hist_data, group_labels, bin_size=0.01, show_rug=False, curve_type="kde", histnorm='probability')
        fig.update_xaxes(title_text="Avg. Traffic Density [veh/m]", range=[0, 0.15])
        fig.update_layout(margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))
        st.plotly_chart(fig)




    for rowData in selected_data:
        scnFolder = rowData['SimulationFolder']
        scn       = rowData['SCN']

        ######### Data in Simulation ######### 
        markdown(f"Data for {scnFolder}", header='h3', color='black')
        col1, col2 = st.columns(2)
        with col1: 
            st.plotly_chart(plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "plot_carData_CTM_trajectory.json")), 
                            use_container_width=True)
            seperater_line()
            st.plotly_chart(plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "plot_carData_trajectory.json")), 
                        use_container_width=True)
            seperater_line()
            st.plotly_chart(plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "plot_partialData_trajectory.json")), 
                        use_container_width=True)
        with col2:     
            st.plotly_chart(plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "plot_CTMMatrix.json")), 
                            use_container_width=True)
            seperater_line()
            st.plotly_chart(plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "plot_carMatrix.json")), 
                            use_container_width=True)
            seperater_line()
            st.plotly_chart(plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "plot_partialMatrix.json")), 
                               use_container_width=True)
        
        ## DNSITY TUPLES IMFORMATION ##
        df = densTup[densTup['SCNFolder'] == scnFolder]
        markdown(f"Density Tuples {scnFolder}", header='h3', color='black')
        st.write("Count: ", df.shape[0])
        st.dataframe(df, use_container_width=True)


        with st.expander("FD for the Scenario: "):
                    st.plotly_chart(plt.load_data('plotly', os.path.join(SUMO_FOLDER, scn, "output/fd_plot.json")), 
                        use_container_width=True)
        with st.expander("FCD Trajectories the Scenario: "):
                    st.plotly_chart(plt.load_data('plotly', os.path.join(SUMO_FOLDER, scn, "output/all_simulated_traj_plot.json")), 
                        use_container_width=True)

    with st.expander("Show All tuples: "):
        st.write("Count: ", densTup.shape[0])
        show_table(densTup)
    

def fd_estimation() -> None:
    markdown("FD ESTIMATION FROM DENSITY TUPLES")
    fdEstimationData = pd.read_csv(os.path.join(MAIN_FOLDER,'fdEstimationResult.csv'), sep=";", header=0)
    selected_data = show_table(fdEstimationData)
    
    for data in selected_data:
        denType     = data['Matrix Type']

        markdown(f"Fitness Plot", header='h3', color='black')
        col1, col2, col3 = st.columns(3)
        with col1: markdown(f"Best VF: {data['Best VF']}", header='h5', color='black', showLine=False)
        with col2: markdown(f"Best RHO_C: {data['Best RHO_C']}", header='h5', color='black', showLine=False)
        with col3: markdown(f"R2 Score: {data['Fitness value']}", header='h5', color='black', showLine=False)
        st.plotly_chart(plt.load_data('plotly', os.path.join(MAIN_FOLDER, f"FDEstimation_FitnessPlot_{denType}.json")), 
                use_container_width=True)

        with st.expander("Show FD plot: "):
            params = {'vf'   : data['Best VF'],
                    'rho_c': data['Best RHO_C'],
                    'rho_j': 0.15}
            st.plotly_chart(plt.plot_fd2(params, SIMULATION_DATA_FOLDER), use_container_width=True)

        with st.expander("Show All tuples: "):
            df = pd.read_csv(os.path.join(MAIN_FOLDER,f'densTuple{denType}.csv'), sep=",", header=0)
            st.write("Count: ", df.shape[0])
            st.dataframe(df, use_container_width=True)
################################################################################################


################################################################################################
                                    #### GA RESULTS ####
################################################################################################
def ga_results():
    param_search_data = pd.read_csv("data/GAPS_gen_pop.csv", sep=";", header=0)
    ## PLOTS ##
    col1, col2 = st.columns(2)
    with col1: 
        fig = plt.go.Figure()
        for gen, frame in param_search_data.groupby("Generation"):
            fig.add_trace(plt.go.Scatter(x=frame["Population"], y=frame["Avg. RMSE (140 EXP)"],
                                         name=f"NUM_GEN_{gen}"))
        fig.update_xaxes(title_text='Population')
        fig.update_yaxes(title_text='AVG RMSE (140 EXP)')
        fig.update_layout(title="AVG RMSE value for different GA Params", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    with col2: 
        fig = plt.ff.create_distplot([param_search_data["Avg. RMSE (140 EXP)"].to_list()],
                                ["AVG. RMSE"], show_hist=False, show_rug=False)
        fig.update_xaxes(title_text='Avg. RMSE (140 EXP)')
        st.plotly_chart(fig, use_container_width=True)


    markdown("GA Results with FD from Partial Matrix Density Tuples")
    markdown(f"nGen80_nPop600_nMat300_tour10_mutProb0-75", header='h3', color='black', showLine=False)

    # GA Results
    gaRes = pd.read_csv(GA_METATDATA, sep=";", header=0)
    gaRes.set_index("EXP", inplace=True, drop=False)
    gaRes.drop([48, 109], inplace=True)

    ## RMSE PLOT ##
    fig = plt.init_rmse_plot()
    # fig = plt.plot_rmse(fig,
    #                     x_data=gaRes["EXP"], y_data=gaRes["RMSE (NonMaked)"],
    #                     name="GA (Non-Masked)")
    fig = plt.plot_rmse(fig,
                    x_data=gaRes["EXP"], y_data=gaRes["RMSE (Masked)"],
                    name="RMSE-GA") 
    fig = plt.plot_rmse(fig,
                x_data=gaRes["EXP"], y_data=gaRes["CTM RMSE (Masked)"],
                name="RMSE-CTM")                
    st.plotly_chart(fig, use_container_width=True)


    with st.sidebar:
        show_input = st.checkbox("Show INPUT Data",  value=True)

    # GA METADATA TABLE
    data = gaRes.copy()
    data = data[["EXP", "SimulationFolder", "Scenario", "ProbeID", "RMSE (Masked)", "MAE (Masked)", "CTM RMSE (Masked)", "CTM MAE (Masked)",
                 "RMSE (NonMaked)", "MAE (NonMaked)"]]
    
    selected_data = show_table(data.round(5))
    for rowData in selected_data:
        expFolder       = rowData['EXP']
        scnFolder       = rowData['SimulationFolder']
        markdown(f'EXP: {rowData["EXP"]} / {rowData["Scenario"]}-{rowData["ProbeID"]}', header='h2', color='red', showLine=True)

        if show_input:
            markdown("INPUT DATA", header='h3', color='black', showLine=False)
            col1, col2 = st.columns(2)
            with col1:
                fig = plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, str(scnFolder), "plot_carData_trajectory.json"))
                st.plotly_chart(fig, use_container_width=True)
                fig = plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, str(scnFolder), "plot_partialData_trajectory.json"))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, str(scnFolder), "plot_carMatrix.json"))
                fig.data[0]["zmax"] = 0.16
                st.plotly_chart(fig, use_container_width=True)
                fig = plt.load_data('plotly', os.path.join(SIMULATION_DATA_FOLDER, str(scnFolder), "plot_partialMatrix.json"))
                fig.data[0]["zmax"] = 0.16
                st.plotly_chart(fig, use_container_width=True)
        
        markdown("OUTPUT DATA", header='h3', color='black', showLine=False)
        col1, col2, col3 = st.columns(3)
        with col1:
            markdown(f"GA RMSE: {rowData['RMSE (Masked)']}", header='h5', color='black', showLine=False)
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_actual_data_masked.json"))
            fig.data[0]["zmax"] = 0.16
            st.plotly_chart(fig, use_container_width=True)
            fitness = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_fitness.json"))
            fitness.layout['yaxis']['range'] = [-0.03, 0]
            st.plotly_chart(fitness, use_container_width=True)
            # markdown(f"GA RMSE: {rowData['RMSE (NonMaked)']}", header='h5', color='black', showLine=False)
            # st.plotly_chart(plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_actual_data.json")), 
            #                 use_container_width=True)

        # st.write(fitness.layout['yaxis']['range'] = [-0.03, 0])   
        
        with col2:
            markdown(f"GA MAE: {rowData['MAE (Masked)']}", header='h5', color='black', showLine=False)
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_best_output_masked.json"))
            fig.data[0]["zmax"] = 0.16
            st.plotly_chart(fig, use_container_width=True)
            res_plot1 = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_residuals_masked.json"))
            st.plotly_chart(res_plot1, use_container_width=True)
            # markdown(f"GA MAE: {rowData['MAE (NonMaked)']}", header='h5', color='black', showLine=False)
            # st.plotly_chart(plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_best_output.json")), 
            #                 use_container_width=True)
        
        with col3: 
            markdown(f"CTM RMSE: {rowData['CTM RMSE (Masked)']}", header='h5', color='black', showLine=False)
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_CTM_output_masked.json"))
            fig.data[0]["zmax"] = 0.16
            st.plotly_chart(fig, use_container_width=True)
            res_plot2 = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_CTM_residuals_masked.json"))
            st.plotly_chart(res_plot2, use_container_width=True)

    # Merge of simData and gaResults
    simData = pd.read_csv("data/SimulationMetadata.csv", sep=";")
    data = pd.merge(simData, gaRes)

    col1, col2 = st.columns(2)
    import plotly.express as px
    import plotly.graph_objects as go
    with col1: 
        ga_plot = px.scatter(data, x="Car Density", y="RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        ga_plot.update_traces(marker=dict(color='red'))
        ct_plot = px.scatter(data, x="Car Density", y="CTM RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")

        fig = go.Figure(data=ga_plot.data + ct_plot.data)
        fig.add_vline(x=0.05768,
                        line_width=1, line_dash="dash", line_color="black")
        fig.update_xaxes(title_text="Opposite Lane Density [veh/m]",
                        range=[0, 0.15], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="RMSE",
                        range=[0, 0.025], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_traces(marker=dict(size=8,
                                    line=dict(width=1,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout(title="GA-RMSE vs Opposite Lane Density", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
    
        # gaOLS = px.get_trendline_results(ga_plot)
        # st.write(gaOLS.px_fit_results[0].summary())

        ga_plot = px.scatter(data, x="Num_visible_cells", y="RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        ga_plot.update_traces(marker=dict(color='red'))

        fig = go.Figure(data=ga_plot.data)
        fig.update_xaxes(title_text="Number of Visible Cells",
                        range=[5, 40], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="RMSE",
                        range=[0, 0.025], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_traces(marker=dict(size=8,
                                    line=dict(width=1,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout(title="GA-RMSE vs Number of Visible Cells for GA", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)


    with col2: 
        ga_plot = px.scatter(data, x="Probe Speed", y="RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        ga_plot.update_traces(marker=dict(color='red'))
        ct_plot = px.scatter(data, x="Probe Speed", y="CTM RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        fig = go.Figure(data=ga_plot.data + ct_plot.data)
        fig.update_xaxes(title_text="Moving Camera Speed [m/sec]",
                        range=[2, 10], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="RMSE",
                        range=[0, 0.025], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_traces(marker=dict(size=8,
                                    line=dict(width=1,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout(title="RMSE vs Moving Camera Speed", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

################################################################################################
                                    #### STREAMLIT APP ####
################################################################################################
def main()-> None:
    page_names_to_funcs = {
        "SUMO Simulated Data": sumo_simulation_data,
        "FD Estimation": fd_estimation,
        "GA Results": ga_results
    }
    with st.sidebar:
        selected_page = st.selectbox("Select Type of Data", page_names_to_funcs.keys(), index=0)
        st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    page_names_to_funcs[selected_page]()


if __name__ =="__main__":
    main()