import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

st.set_page_config(page_title="Cars Dashboard", page_icon=":oncoming_automobile:", layout="wide")


def main():
    # Sidebar
    choices = ["Data Exploration", "Data Visualization", "Predict Clusters"]

    st.sidebar.subheader("File Upload")

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

    choice = st.sidebar.radio("Select your Choice", choices)

    @st.cache(allow_output_mutation=True)
    def get_data():
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            return data

    df = get_data()

    # Main Page
    st.title(" :oncoming_automobile: Dashboard ")
    st.markdown("""___""")

    left_column, right_column = st.columns([1, 3])

    with right_column:
        if uploaded_file is not None:

            with st.expander("See the Data"):
                st.dataframe(df)
        else:
            st.write("Awaiting CSV file to be uploaded")

    if choice == 'Data Exploration' and uploaded_file is not None:
        left_column.subheader("Explore Your Data")
        right_column.subheader("Values Obtained")
        if left_column.checkbox("Show Shape"):
            right_column.write(df.shape)

        if left_column.checkbox("Show Columns"):
            all_columns = df.columns.to_list()
            right_column.write(all_columns)

        if left_column.checkbox("Summary"):
            right_column.write(df.describe())

        if left_column.checkbox("Show Selected Columns"):
            all_columns = df.columns.to_list()
            selected_columns = right_column.multiselect("Select Columns", all_columns)
            new_df = df[selected_columns]
            right_column.dataframe(new_df)

        if left_column.checkbox("Most Popular"):
            all_columns = df.columns.to_list()
            selected_column = right_column.selectbox("Select a Column", all_columns)
            new_df = df[selected_column]
            right_column.write(new_df.value_counts())

        if left_column.checkbox("Correlation Plot"):
            right_column.write('Analyzing relation between columns. Values close to one indicate a strong relationship.')
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), ax=ax)
            right_column.write(fig)

    elif choice == 'Data Visualization':
        left_column.subheader("Select kind of Graph")
        right_column.subheader("Plot Visualization")
        if uploaded_file is not None:
            plot_type = left_column.selectbox("Choose the type of plot",
                                              ["Area Plot", "Bar Chart", "Pie Chart", "Box Plot", "Histogram",
                                               "Line Chart",
                                               "Scatter Plot"])

            all_columns = df.columns.to_list()

            if plot_type == 'Area Plot':
                y_axis = left_column.selectbox("Select y-axis", all_columns)
                x_axis = left_column.selectbox("Select x-axis", all_columns)
                group_by = left_column.selectbox("Choose a feature to group the data", all_columns)
                fig = px.area(df, x=x_axis, y=y_axis, color=group_by)
                fig.update_layout(showlegend=True)
                if left_column.button("Generate Plot"):
                    right_column.write(fig)

            elif plot_type == 'Bar Chart':
                y_axis = left_column.selectbox("Select y-axis", all_columns)
                x_axis = left_column.selectbox("Select x-axis", all_columns)
                group_by = left_column.selectbox("Choose a feature to group the data", all_columns)
                fig = px.bar(df, x=x_axis, y=y_axis, color=group_by, barmode='group')
                fig.update_layout(showlegend=True)
                if left_column.button("Generate Plot"):
                    right_column.write(fig)

            elif plot_type == "Pie Chart":
                x_axis = left_column.selectbox("Select a numerical feature", all_columns)
                group_by = left_column.selectbox("Choose a feature to group the data", all_columns)
                fig = px.pie(df, values=x_axis, names=group_by)
                if left_column.button("Generate Plot"):
                    right_column.write(fig)

            elif plot_type == 'Box Plot':
                y_axis = left_column.selectbox("Select y-axis", all_columns)
                x_axis = left_column.selectbox("Select x-axis", all_columns)
                fig = px.box(df, y=y_axis, x=x_axis)
                if left_column.button("Generate Plot"):
                    right_column.write(fig)

            elif plot_type == "Histogram":
                x_axis = left_column.selectbox("Select x-axis", all_columns)
                fig = px.histogram(df, x=x_axis, nbins=10)
                fig.update_layout(showlegend=True)
                if left_column.button("Generate Plot"):
                    right_column.write(fig)

            elif plot_type == "Line Chart":
                y_axis = left_column.selectbox("Select y-axis", all_columns)
                x_axis = left_column.selectbox("Select x-axis", all_columns)
                group_by = left_column.selectbox("Choose a feature to group the data", all_columns)
                fig = px.line(df, x=x_axis, y=y_axis, color=group_by)
                fig.update_layout(showlegend=True)
                if left_column.button("Generate Plot"):
                    right_column.write(fig)

            elif plot_type == "Scatter Plot":
                choices = ["2", "3"]
                choice = left_column.radio("Relation between 2 or 3 variables ?", choices)
                if choice == '2':
                    y_axis = left_column.selectbox("Select y-axis", all_columns)
                    x_axis = left_column.selectbox("Select x-axis", all_columns)
                    group_by = left_column.selectbox("Choose a feature to group the data", all_columns)
                    fig = px.scatter(df, x=x_axis, y=y_axis, color=group_by, width=800, height=750)
                    fig.update_layout(showlegend=True)
                    if left_column.button("Generate Plot"):
                        right_column.write(fig)
                if choice == '3':
                    y_axis = left_column.selectbox("Select y-axis", all_columns)
                    x_axis = left_column.selectbox("Select x-axis", all_columns)
                    z_axis = left_column.selectbox("Select z-axis", all_columns)
                    group_by = left_column.selectbox("Choose a feature to group the data", all_columns)
                    fig = px.scatter_3d(df, x=x_axis, z=z_axis, y=y_axis, color=group_by, width=800, height=750)
                    fig.update_layout(showlegend=True)
                    if left_column.button("Generate Plot"):
                        right_column.write(fig)

    elif choice == 'Predict Clusters':
        left_column.subheader("Analyze different Clusters")
        right_column.subheader("Cluster Info")
        with right_column:
            with st.expander('What is Clustering ? '):
                st.write("In clustering, there are many variables taken in consideration which are complicated to be "
                         "predicted by normal tricks. Clusters generated by KMeans Clustering model can be used to "
                         "identify the strategic group that form a strong competition to the company products in "
                         "world market and it can also show the closest clusters to this group which also can be "
                         "put into use in some other cases.")

        if uploaded_file is not None:
            new_df = pd.read_csv(uploaded_file)

            # Building Model for K-Means

            all_columns = new_df.columns.to_list()
            cols = [i for i in new_df.columns if new_df[i].dtype != 'object']
            num_clusters = left_column.slider('Choose number of clusters :', 1, 16)

            km = KMeans(n_clusters=num_clusters, n_init=20, max_iter=400, random_state=0)
            clusters = km.fit_predict(new_df[cols])
            new_df['Cluster'] = clusters
            new_df.Cluster = (new_df.Cluster + 1).astype('object')

            with right_column:
                with st.expander('See data with respective clusters'):
                    st.dataframe(new_df)

            y_axis = left_column.selectbox("Select y-axis", all_columns)
            x_axis = left_column.selectbox("Select x-axis", all_columns)

            fig = px.scatter(df, x=x_axis, y=y_axis, color=new_df['Cluster'], width=800, height=750)
            fig.update_layout(showlegend=True)
            if left_column.button("Generate Plot"):
                right_column.write(fig)

            car_name = left_column.text_input('Enter name of a car from your data to see its variants and the '
                                              'clusters they belong to')
            if car_name:
                df_t = new_df[new_df.Car == car_name]
                right_column.write(new_df[new_df.Car == car_name])

                values = df_t.Cluster.unique()
                chosen_clusters = left_column.multiselect('Choose clusters from obtained clusters of the car', values)
                df_c = new_df[new_df.Cluster.isin(chosen_clusters)]

                right_column.write('Cars in selected clusters:')
                right_column.dataframe(df_c)

                features = left_column.selectbox("Select a feature", all_columns)
                fig = px.histogram(df_c, x=features, nbins=10)
                fig.update_layout(showlegend=True)
                if left_column.button("Count in above chosen clusters"):
                    right_column.write(fig)

    # ---- HIDE STREAMLIT STYLE ----
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
