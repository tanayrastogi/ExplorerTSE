import os
import pandas as pd
import streamlit as st 
from st_aggrid import GridOptionsBuilder, AgGrid
from summary import plot_fundamental_diagram, load_data
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")


SIMULATION_DATA_FOLDER      = "SimulationData"
EXPERIMENT_DATA_FOLDER      = "ExperimentResults"
INFO_FOLDER                 = "Info"
EXPERIMENT_RESULTS          = os.path.join(INFO_FOLDER, "Results.csv")
SCENARIO_INFO_FILE          = os.path.join(INFO_FOLDER, "ScenarioData.csv")


                                               
################################################################################################
                                    #### RESULTS EXPLORE ####
################################################################################################
def explore() -> None:
    with st.sidebar:
        st.markdown("### Select plots to show")
        show_non_masked = st.checkbox("Show Non Masked Data",  value=False)
        show_masked     = st.checkbox("Show Masked Data",  value=True)
        show_fd         = st.checkbox("Show Fundamental Digrams", value=True)
        show_actual_data= st.checkbox("Show Actual Data", value=True)

    st.markdown("<h1 style='text-align: center; color: red;'>GA-MCTM Analysis using Simulated Data</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            ## GA Parameters
            All the scenarios have been running using the following GA parameters. Each scenario is computed 5 different time and the results are reported for the average of **5 runs**.

            - N generation: 150
            - N population: 800
            - N mating parents: 200
            - N tournaments: 5
            - Mutation: [0.1, 0.75]
            """
        )
    with col2:
        st.markdown("""
                ## CTM Parameters
        The fitness for each solution candidate is calculated using CTM with following parameters. 

        - Free Flow Speed: 20 m/s
        - Avg Car length: 4.5 m
        - Number of Lanes: 3
        - Jam Density: 3/4.5 = 0.6667 veh/m
        """)


    # Read CSV result
    result   = pd.read_csv(EXPERIMENT_RESULTS, sep=";", header=0)
    result["fullnames"] = result.apply(lambda x: x["Scenario"]+"_"+x["ProbeID"]+"_"+x["Timestamp"], axis=1)
    result.set_index("fullnames", drop=True, inplace=True)
    result["RMSE [veh]"] = result["RMSE Error"] * 20.0
    result["MAE [veh]"] = result["MAE Error"] * 20.0
    result["RMSE-M [veh]"] = result["RMSE Error - Masked"] * 20.0
    result["MAE-M [veh]"] = result["MAE Error - Masked"] * 20.0
    scenario = pd.read_csv(SCENARIO_INFO_FILE, sep=";", header=0, index_col=0)
    ## Merge Scenario and Results table ##
    merged = scenario.join(result)
    merged["EXPName"] = merged.index.values
    merged["Scenario"] = merged["EXPName"].apply(lambda x: x.split("_")[0])
    merged["ProbeID"] = merged["EXPName"].apply(lambda x: x.split("_")[1])
    merged = merged.dropna(axis=0)
    merged.set_index("EXP", drop=False, inplace=True)
    
    ########## TABLE ############
    # Show the result table
    st.markdown("<h2 style='text-align: center; color: black;'>RESULT</h2>", unsafe_allow_html=True) 
    table = merged.copy()
    table.sort_values(by=['RMSE-M [veh]'], inplace=True)
    table = table.round(3)
    col = ["EXP", "Scenario", "ProbeID", "EXPName", "Car Flow", "Car Density", "Car Speed", "Probe Time", "Probe Distance", "Probe Speed", "RMSE-M [veh]", "MAE-M [veh]"]
    table = table[col]

    # Show interactive table
    gb = GridOptionsBuilder.from_dataframe(table)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_default_column(min_column_width=2) #Column width
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

    if show_fd:    
        with st.expander("Show FD for Simulated Data", expanded=False):
            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", "Info/cDens_vs_cFlow.json"), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", "Info/cDens_vs_cSpeed.json"), use_container_width=True)
            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", "Info/cFlow_vs_cSpeed.json"), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", "Info/pSpeed_vs_cSpeed.json"), use_container_width=True)


    for data in selected_data:
        # Select the experiment number and show plots for the exp
        exp_number = int(data["EXP"])
        simulationData = data["EXPName"]
        exp_folder        = os.path.join(EXPERIMENT_DATA_FOLDER, str(exp_number))
        simulation_folder = os.path.join(SIMULATION_DATA_FOLDER, str(simulationData))
        st.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: red;'>EXPERIMENT {}</h2>".format(exp_number), unsafe_allow_html=True) 

        if show_fd:
            ########## FD ############
            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(plot_fundamental_diagram(merged,
                                                            X_data="Car Density", X_title="Traffic Density [veh/m]",
                                                            Y_data="Car Flow", Y_title="Traffic Flow [veh/s]",
                                                            error_type="RMSE [veh]", 
                                                            hightIDX = exp_number,
                                                            plotTitle="Flow-vs-Density RMSE analysis"))
            with col2: st.plotly_chart(plot_fundamental_diagram(merged,
                                                            X_data="Car Density", X_title="Traffic Density [veh/m]",
                                                            Y_data="Car Speed", Y_title="Traffic Speed [m/s]",
                                                            error_type="RMSE [veh]", 
                                                            hightIDX = exp_number, 
                                                            plotTitle="Speed-vs-Density RMSE analysis",
                                                            scaleColor="Blues"))
            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(plot_fundamental_diagram(merged,
                                                            X_data="Car Flow", X_title="Traffic Flow [veh/s]",
                                                            Y_data="Car Speed", Y_title="Traffic Speed [m/s]",
                                                            error_type="RMSE [veh]", 
                                                            hightIDX = exp_number,
                                                            plotTitle="Speed-vs-Flow RMSE analysis",
                                                            scaleColor="Greens"))
            with col2: st.plotly_chart(plot_fundamental_diagram(merged,
                                                        X_data="Probe Speed", X_title="Probe Speed [m/s]",
                                                        Y_data="Car Speed", Y_title="Traffic Speed [m/s]",
                                                        error_type="RMSE [veh]", 
                                                        hightIDX = exp_number,
                                                        plotTitle="Speed-vs-Speed RMSE analysis",
                                                        scaleColor="viridis"))


        ########## RESULTS ############
        if show_actual_data:
            ############## ACTUAL DATA ######################
            st.markdown("""<hr style="height:5px;border:none;color:#ff1100;background-color:#ff1100;" /> """, unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: Black;'>Actual Simulated Data </h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", os.path.join(simulation_folder, "carData_trajectory_plot.json")), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", os.path.join(simulation_folder, "partialData_trajectory_plot.json")), use_container_width=True)

            with col1: st.plotly_chart(load_data("plotly", os.path.join(simulation_folder, "carMatrix_plot.json")), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", os.path.join(simulation_folder, "partialMatrix_plot.json")), use_container_width=True)
     
        ################### Masked ###################
        if show_masked:
            st.markdown("""<hr style="height:5px;border:none;color:#ff1100;background-color:#ff1100;" /> """, unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: black;'>Results Matrix</h3>", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap="small")
            with col1: st.markdown("<h5 style='text-align: center; color: black;'>RMSE-M [veh]: {:.3f}</h5>".format(table.loc[exp_number, "RMSE-M [veh]"]), unsafe_allow_html=True)
            with col2: st.markdown("<h5 style='text-align: center; color: black;'>MAE-M [veh]: {:.3f}</h5>".format(table.loc[exp_number, "MAE-M [veh]"]), unsafe_allow_html=True)

            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "actual_data_masked.json")), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "best_output_masked.json")), use_container_width=True)

            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "residuals_masked.json")), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "fitness.json")), use_container_width=True)
        
        ################### NON Masked ###################
        if show_non_masked:    
            st.markdown("""<hr style="height:5px;border:none;color:#ff1100;background-color:#ff1100;" /> """, unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center; color: black;'>Results Matrix without Masked</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="small")
            with col1: st.markdown("<h5 style='text-align: center; color: black;'>RMSE [veh]: {:.3f}</h5>".format(merged.loc[exp_number, "RMSE [veh]"]), unsafe_allow_html=True)
            with col2: st.markdown("<h5 style='text-align: center; color: black;'>MAE [veh]: {:.3f}</h5>".format(merged.loc[exp_number, "MAE [veh]"]), unsafe_allow_html=True)
            
            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "actual_data.json")), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "best_output.json")), use_container_width=True)

            col1, col2 = st.columns(2, gap="small")
            with col1: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "residuals.json")), use_container_width=True)
            with col2: st.plotly_chart(load_data("plotly", os.path.join(exp_folder, "fitness.json")), use_container_width=True)
    

################################################################################################
                                    #### STREAMLIT APP ####
################################################################################################
def main()-> None:
    explore()

if __name__ =="__main__":
    main()

    # # Read CSV result
    # result   = pd.read_csv(EXPERIMENT_RESULTS, sep=";", header=0)
    # result["fullnames"] = result.apply(lambda x: x["Scenario"]+"_"+x["ProbeID"]+"_"+x["Timestamp"], axis=1)
    # result.set_index("fullnames", drop=True, inplace=True)
    # result["RMSE [veh]"] = result["RMSE Error"] * 20.0
    # result["MAE [veh]"] = result["MAE Error"] * 20.0
    # result["RMSE-M [veh]"] = result["RMSE Error - Masked"] * 20.0
    # result["MAE-M [veh]"] = result["MAE Error - Masked"] * 20.0
    # scenario = pd.read_csv(SCENARIO_INFO_FILE, sep=";", header=0, index_col=0)
    # ## Merge Scenario and Results table ##
    # merged = scenario.join(result)
    # merged["EXPName"] = merged.index.values
    # merged.set_index("EXP", drop=False, inplace=True)