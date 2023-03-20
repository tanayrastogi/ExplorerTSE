import os 
import streamlit as st
import pandas as pd 
from st_aggrid import GridOptionsBuilder, AgGrid
import plotting as plt
import numpy as np
st.set_page_config(layout="wide")

MAIN_FOLDER            = "data"
SIMULATION_DATA_FOLDER = os.path.join(MAIN_FOLDER, "SimulationData")
SIMULATION_METATDATA   = os.path.join(MAIN_FOLDER, "SimulationMetadata.csv")
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

## For Fitness calculation
def rmse_fitness(error):
    return  - np.sqrt(np.nanmean(np.square(error)))


                                    #### EXPERIMENTS ####
################################################################################################
def testing():
    ###### DROP Certain GA EXP because of BAD simulation data #####
    drop_exp = ["Scenario-3_test16_1197-1199", "Scenario-5_test16_995-1005", "Scenario-7_test0_5-14", "Scenario-8_test2_211-240"]

    ############ GA Param Search Plots ###############
    search_data = pd.read_csv(os.path.join("data/param_search_fit.csv"), sep=";", header=0)
    selected_row = search_data[search_data["MeanFit"] == search_data["MeanFit"].min()].index.to_list()
    
    ### RMSE Value for all EXP ###
    rmse_data = pd.read_csv(os.path.join("data/param_search_rmse.csv"), sep=";", header=0, index_col=0)
    temp = pd.DataFrame(rmse_data.mean(axis=0))
    temp.rename(columns={0: "AVG. RMSE"}, inplace=True)
    temp["NAME"] = temp.index.to_list()
    # st.dataframe(temp) 
    search_data = search_data.merge(temp, on="NAME")

    col1, col2 = st.columns(2, gap="small")
    ## PLOT FOR MUTATION AND TOUR ##
    with col1: 
        search_tour_mut = search_data[(search_data["Generation"] == 60) & (search_data["Population"] == 500)]
        # Plot for GA Param Search 
        fig = plt.go.Figure()
        for tour, frame in search_tour_mut.groupby("Tour"):
            fig.add_trace(plt.go.Scatter(x=frame["Mutation"], y=frame["MeanFit"],
                                        name=f"TOUR_{tour}",
                                        customdata= frame["EXP"],
                                        hovertemplate='EXP: %{customdata}<br>Mutation: %{x}<br>FIT: %{y}<extra></extra>'))
        fig.update_xaxes(title_text='Mutation')
        fig.update_yaxes(title_text='AVG FITNESS (140 EXP)')
        fig.update_layout(title="AVG FIT / Tournament - Mutation", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0})
        st.plotly_chart(fig, use_container_width=True)

    ## PLOT FOR GENERATION AND POPULATION ##  
    with col2:   
        search_gen_pop = search_data[search_data["Population"] != 500]
        fig = plt.go.Figure()
        for gen, frame in search_gen_pop.groupby("Generation"):
            fig.add_trace(plt.go.Scatter(x=frame["Population"], y=frame["MeanFit"],
                                         name=f"GEN_{gen}",
                                         customdata= frame["EXP"],
                                         hovertemplate='EXP: %{customdata}<br>Population: %{x}<br>FIT: %{y}<extra></extra>'))
        fig.update_xaxes(title_text='Population')
        fig.update_yaxes(title_text='AVG FITNESS (140 EXP)')
        fig.update_layout(title="AVG FIT / Generation - Population", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":0})
        st.plotly_chart(fig, use_container_width=True)



    ### Function to show each generation development of best solution for EXP Scenario-1_test7_441-456 ####
    # GA Results
    gaRes = pd.read_csv(GA_METATDATA, sep=";", header=0)
    ## Drop scenarios ##
    idx_to_drop = gaRes[gaRes["SimulationFolder"].isin(drop_exp)].index
    gaRes.drop(idx_to_drop, axis=0, inplace=True)
    gaRes.set_index("EXP", inplace=True, drop=False)
    
     ################ RMSE PLOT ################
    fig = plt.init_rmse_plot()
    fig = plt.plot_rmse(fig,
                    x_data=gaRes["EXP"], y_data=gaRes["RMSE (Masked)"],
                    name="RMSE-GA") 
    fig = plt.plot_rmse(fig,
                    x_data=gaRes["EXP"], y_data=gaRes["CTM RMSE (Masked)"],
                    name="RMSE-CTM")                
    fig.add_trace(plt.go.Scatter(x=gaRes["EXP"],
                    y=rmse_data.max(axis=1),
                    fill=None,
                    mode='lines+markers',
                    line_color='rgba(0,100,80,0.01)',
                    showlegend=False,
                ))
    fig.add_trace(plt.go.Scatter(
        x=gaRes["EXP"],
        y=rmse_data.min(axis=1),
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines+markers', line_color='rgba(0,100,80,0.01)',
        name="Min-Max"))
    st.plotly_chart(fig, use_container_width=True)

    ################ SUMMARY PLOT VS SPEED AND DENSITY ################
    simData = pd.read_csv(SIMULATION_METATDATA, sep=";", header=0)
    merge = pd.merge(simData, gaRes)
    col1, col2 = st.columns(2)
    with col1: 
        ga_plot = plt.px.scatter(merge, x="Car Density", y="RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        ga_plot.update_traces(marker=dict(color='red'))
        ct_plot = plt.px.scatter(merge, x="Car Density", y="CTM RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")

        fig = plt.go.Figure(data=ga_plot.data + ct_plot.data)
        fig.add_vline(x=0.05768,
                        line_width=1, line_dash="dash", line_color="black")
        fig.update_xaxes(title_text="Opposite Lane Density [veh/m]",
                        range=[0, 0.15], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="RMSE",
                        range=[0, 0.02], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_traces(marker=dict(size=8,
                                    line=dict(width=1,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout(title="GA-RMSE vs Opposite Lane Density", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)
    with col2: 
        ga_plot = plt.px.scatter(merge, x="Probe Speed", y="RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        ga_plot.update_traces(marker=dict(color='red'))
        ct_plot = plt.px.scatter(merge, x="Probe Speed", y="CTM RMSE (Masked)", hover_data=["SimulationFolder"], trendline="ols")
        fig = plt.go.Figure(data=ga_plot.data + ct_plot.data)
        fig.update_xaxes(title_text="Moving Camera Speed [m/sec]",
                        range=[2, 10], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="RMSE",
                        range=[0, 0.02], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_traces(marker=dict(size=8,
                                    line=dict(width=1,
                                            color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.update_layout(title="RMSE vs Moving Camera Speed", title_x=0.5, margin={"r":10,"t":40,"l":10,"b":10}, plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    # GA METADATA TABLE
    data = gaRes.copy()
    data = data[["EXP", "SimulationFolder", "Free Flow Speed [m/sec]", "Jam Density", "Opt Density", 
                "RMSE (Masked)", "CTM RMSE (Masked)"]]
    selected_data = show_table(data.round(5))
    for rowData in selected_data:
        expFolder       = rowData['EXP']
        scnFolder       = rowData['SimulationFolder']
        gaRMSE          = rowData['RMSE (Masked)']
        ctmRMSE         = rowData['CTM RMSE (Masked)']

        # Simulation Data
        actual  = plt.load_data('dataframe', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "carMatrix.pkl"))
        partial = plt.load_data('dataframe', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "partialMatrix.pkl"))
        overlap = plt.load_data('dataframe', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "overlapPercent.pkl")).fillna(value=np.nan)

        name = f'EXP: {rowData["EXP"]} - With all partial cells for fitness'

        markdown(name, header='h2', color='red', showLine=True)
        markdown("INPUT DATA", header='h3', color='black', showLine=True)
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
            st.plotly_chart(plt.plot_resiual_matrix(actual, partial), use_container_width=True)  



        markdown(f"FINAL OUTPUT/ GA RMSE: {gaRMSE}", header='h3', color='black', showLine=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_actual_data_masked.json"))
            fig.update_layout(title="ACTUAL DATA", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_fitness.json")), 
                            use_container_width=True)

        with col2:
            ## BEST MODEL GA
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_best_output_masked.json"))
            fig.data[0]["zmax"] = 0.15
            fig.update_layout(title="MODEL OUTPUT FROM GA", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
            ## FITNESS VALUE FOR GA
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_fitness_Residual_GA.json"))
            error = pd.DataFrame(fig.data[0]['z'], index=overlap.index, columns=overlap.columns).fillna(value=np.nan)
            if (rowData["EXP"] > 5) and  (rowData["EXP"] <= 10):
                error = error[overlap >= .99]
            fig.update_layout(title=f"RESIDUAL: PARTIAL vs GA /FIT: {round(rmse_fitness(error), 6)}", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
            ## RESIDUAL ACTUAL VS BEST MODEL
            res = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_residuals_masked.json"))
            res.update_layout(title=f"RESIDUAL: Actual vs GA /RMSE: {gaRMSE}", title_x=0.5)
            st.plotly_chart(res, use_container_width=True)


        with col3:
            ## CTM BEST MODEL
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_CTM_output_masked.json"))
            fig.update_layout(title="MODEL OUTPUT FROM CTM", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
            ## FITNESS VALUE FOR GA
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_fitness_Residual_CTM.json"))
            error = pd.DataFrame(fig.data[0]['z'], index=overlap.index, columns=overlap.columns).fillna(value=np.nan)
            if (rowData["EXP"] > 5) and  (rowData["EXP"] <= 10):
                error = error[overlap >= .99]
            fig.update_layout(title=f"RESIDUAL: PARTIAL vs CTM /FIT: {round(rmse_fitness(error), 6)}", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
            ## RESIDUAL ACTUAL VS CTM BEST MODEL
            fig = plt.load_data('plotly', os.path.join(GA_DATA_FOLDER, str(expFolder), "plot_CTM_residuals_masked.json"))
            fig.update_layout(title=f"RESIDUAL: Actual vs CTM /RMSE: {ctmRMSE}", title_x=0.5)
            st.plotly_chart(fig, use_container_width=True)
            
################################################################################################
                                    #### STREAMLIT APP ####
################################################################################################
def main()-> None:
    testing()
    pass


if __name__ =="__main__":
    main()