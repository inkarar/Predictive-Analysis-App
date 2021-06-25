# Importing the libraries
import numpy as np
import pandas as pd
import plotly.express as px
import base64
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Predictive Analysis App",layout='wide')

LOGO_IMAGE = "Codebios_Logo.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:400 !important;
        font-size:20px !important;
        color: #f9a01b !important;
        padding-top: 50px !important;
    }
    .logo-img {
        float:left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


st.write('''
# Predictive Analysis App
''')

try:
        uploaded_file = st.file_uploader("Please Upload your .csv file", type="csv")

        if uploaded_file:
                dataset = pd.read_csv(uploaded_file, engine='python')

        view = st.selectbox('Do you want to see the uploaded csv file?',('yes','no'))

        if view == 'yes':
                st.table(dataset.head())


        xaxis = st.selectbox('Which of the following variables you want to display on the X-axis?',list(dataset.columns))
        yaxis = st.selectbox('Which of the following variables you want to display on the Y-axis?',list(dataset.columns))
        st.write(f'''
        ## Plotting '{yaxis}' vs '{xaxis}'
        ''')
        fig2 = px.line(x=dataset[xaxis],y=dataset[yaxis], labels={'x':f"{xaxis}", 'y':f"{yaxis}"})
        st.plotly_chart(fig2, use_container_width=True, sharing='streamlit')


        st.write('''
        ### Select all the variables you want to exclude from the analysis
        ''')
        var = st.multiselect('Which of the following variables you want to exclude?',list(dataset.columns))

        if var:
                dataset = dataset.drop(var,axis=1)
        else:
                pass

        st.write('''
        ### Select the target variable for which you want to predict values
        ''')
        target = st.selectbox('Which of the following variables is the target/dependent variable?',list(dataset.columns))

        yes = st.selectbox('Does your data have a Date column?',('no','yes'))

        if yes == 'yes':
                date = st.selectbox('Which of the following variables is the DateTime Variable?',list(dataset.columns))
                import datetime as dt
                dataset[date] = dataset[date].apply(lambda x: dt.datetime.strptime(x,"%Y-%m-%d"))
                dataset[date] = dataset[date].apply(lambda x: dt.datetime.timestamp(x)) #apply(lambda x: timestamp(x))

        X = dataset.drop(target,axis=1)
        y = dataset[target]

        # Encoding categorical data
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import mean_squared_error, r2_score

        st.write('''
        ### Select the categorical variables for data preprocessing
        ''')
        cat = st.multiselect('Which of the following variables are categorical variable? Leave blank if None',list(dataset.columns))
        category = [dataset.columns.get_loc(c) for c in cat if c in dataset]

        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), category)], remainder='passthrough')
        X = np.array(ct.fit_transform(X))

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

        # Training the Multiple Linear Regression model on the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        
        st.write(f'''
        ## Predicted and Actual values of Target Variable - {target}
        ''')
        #st.line_chart(pd.DataFrame(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1),columns=['predicted', 'actual']))
        chart = pd.DataFrame(np.concatenate((np.array(y_pred).reshape(len(y_pred),1), np.array(y_test).reshape(len(y_test),1)),1),columns=['predicted', 'actual'])
        fig1 = px.line(chart, x = chart.index, y = ['predicted','actual'], labels={'x':"Data Index", 'y':f"{target}"})
        st.plotly_chart(fig1, use_container_width=True, sharing='streamlit')

        st.write('''
        ## Mean Squared Error and R2 score from Regression Analysis:
        ''')
        st.write("MSE: ", mean_squared_error(np.array(y_test).reshape(len(y_test),1), np.array(y_pred).reshape(len(y_pred),1)))
        st.write("R2 score: ", r2_score(np.array(y_test).reshape(len(y_test),1), np.array(y_pred).reshape(len(y_pred),1)))

        st.write('-------------------------------------------------------')

except:
        st.write('''
        ## Please upload a csv file, select appropriate options from dropdown menu and check if your selected options makes sense
        ''')


st.markdown(
    f"""
    <div class="container">
        <p>&copy; Copyright All Rights Reserved. <a href="https://codebios.com/" target="_blank">Codebios Technology.</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
