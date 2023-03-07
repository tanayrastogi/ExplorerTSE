import os 
import streamlit as st
import pandas as pd 
from st_aggrid import GridOptionsBuilder, AgGrid
import plotting as plt
st.set_page_config(layout="wide")

MAIN_FOLDER            = "data"
SIMULATION_DATA_FOLDER = os.path.join(MAIN_FOLDER, "SimulationData")
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



                                    #### EXPERIMENTS ####
################################################################################################
def testing():
    ### Function to show each generation development of best solution for EXP Scenario-1_test7_441-456 ####
    # GA Results
    gaRes = pd.read_csv(GA_METATDATA, sep=";", header=0)
    gaRes.set_index("EXP", inplace=True, drop=False)
    
    # GA METADATA TABLE
    data = gaRes.copy()
    data = data[["EXP", "SimulationFolder", "Free Flow Speed [m/sec]", "Jam Density", "Opt Density",
                "N Generations", "N Repeat", "N Population", "N MatingParents", "k Selection", 
                "Mutation Probabilities", "RMSE (Masked)", "CTM RMSE (Masked)"]]
    selected_data = show_table(data.round(5))
    for rowData in selected_data:
        expFolder       = rowData['EXP']
        scnFolder       = rowData['SimulationFolder']
        gaRMSE          = rowData['RMSE (Masked)']
        ctmRMSE         = rowData['CTM RMSE (Masked)']

        if rowData["EXP"] <=5:
            name = f'EXP: {rowData["EXP"]} - With all partial cells for fitness'
        elif (rowData["EXP"] > 5) and  (rowData["EXP"] <= 10):
            name = f'EXP: {rowData["EXP"]} - With partial cells with area coverage > 0.99 for fitness'

        markdown(name, header='h2', color='red', showLine=True)
        markdown("INPUT DATA", header='h3', color='black', showLine=True)
        actual    = plt.load_data('dataframe', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "carMatrix.pkl"))
        partial = plt.load_data('dataframe', os.path.join(SIMULATION_DATA_FOLDER, scnFolder, "partialMatrix.pkl"))
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
            fig.update_layout(title="RESIDUAL: PARTIAL vs GA", title_x=0.5)
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
            fig.update_layout(title="RESIDUAL: PARTIAL vs CTM", title_x=0.5)
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


if __name__ =="__main__":
    main()