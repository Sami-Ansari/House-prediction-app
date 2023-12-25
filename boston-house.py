import streamlit as st
import pandas as pd

# import sharp
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor


st.write(
    """
# Boston House Price Prediction APP

This app predict the **Boston House Price**!
         """
)

st.write("---")

# load the Boston House Price Dataset
house = pd.read_csv("house.csv")
house.dropna(inplace=True)
X = house.drop(columns=["MEDV"])
Y = house["MEDV"]

st.sidebar.header("Specify Input Parameters")
st.write("""
         - CRIM per capita crime rate by town 
         - ZN proportion of residential land zoned for lots over 25,000 sq.ft. 
         - INDUS proportion of non-retail business acres per town 
         - CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
         - NOX nitric oxides concentration (parts per 10 million) 
         - RM average number of rooms per dwelling 
         - AGE proportion of owner-occupied units built prior to 1940 
         - DIS weighted distances to five Boston employment centres 
         - RAD index of accessibility to radial highways 
         - TAX full-value property-tax rate per $10,000 
         - PTRATIO pupil-teacher ratio by town 
         - B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
         - LSTAT lower status of the population 
         - MEDV Median value of owner-occupied homes in $1000's
         """)

def user_input_features():
    CRIM = st.sidebar.slider("CRIM", X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
    ZN = st.sidebar.slider("ZN", X.ZN.min(), X.ZN.max(), X.ZN.mean())
    INDUS = st.sidebar.slider("INDUS", X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
    CHAS = st.sidebar.slider("CHAS", X.CHAS.min(), X.CHAS.max(),int( X.CHAS.mean()))
    NOX = st.sidebar.slider("NOX", X.NOX.min(), X.NOX.max(), X.NOX.mean())
    RM = st.sidebar.slider("RM", X.RM.min(), X.RM.max(), X.RM.mean())
    AGE = st.sidebar.slider("AGE", X.AGE.min(), X.AGE.max(), X.AGE.mean())
    DIS = st.sidebar.slider("DIS", X.DIS.min(), X.DIS.max(), X.DIS.mean())
    RAD = st.sidebar.slider("RAD", X.RAD.min(), X.RAD.max(), int(X.RAD.mean()))
    TAX = st.sidebar.slider("TAX", X.TAX.min(), X.TAX.max(), int(X.TAX.mean()))
    PTRATIO = st.sidebar.slider(
        "PTRATIO", X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean()
    )
    B = st.sidebar.slider("B", X.B.min(), X.B.max(), X.B.mean())
    LSTAT = st.sidebar.slider("LSTAT", X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
    data = {
        "CRIM": CRIM,
        "ZN": ZN,
        "INDUS": INDUS,
        "CHAS": CHAS,
        "NOX": NOX,
        "RM": RM,
        "AGE": AGE,
        "DIS": DIS,
        "RAD": RAD,
        "TAX": TAX,
        "PTRATIO": PTRATIO,
        "B": B,
        "LSTAT": LSTAT,
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

    # Main Pannel

    # Print specified input parameters
st.header("Specified Input Parameters")
st.write(df)
st.write("---")

    # Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
 # Apply Model to make predicion
prediction = model.predict(df)

st.header("Predictionof MEDV")
st.write(prediction)
st.write("---")
