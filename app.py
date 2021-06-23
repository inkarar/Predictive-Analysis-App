# Importing the libraries
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Predictive Analysis App",layout='wide')

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

        


        X = dataset.drop(target,axis=1).values
        y = dataset[target].values


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
        chart = pd.DataFrame(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1),columns=['predicted', 'actual'])
        fig1 = px.line(chart, x = chart.index, y = ['predicted','actual'], labels={'x':"Data Index", 'y':f"{target}"})
        st.plotly_chart(fig1, use_container_width=True, sharing='streamlit')

        st.write('''
        ## Mean Squared Error and R2 score from Regression Analysis:
        ''')
        st.write("MSE: ", mean_squared_error(y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)))
        st.write("R2 score: ", r2_score(y_test.reshape(len(y_test),1), y_pred.reshape(len(y_pred),1)))


except:
        st.write('''
        ## Please upload a csv file, select appropriate options from dropdown menu and check if your selected options makes sense
        ''')



