import os
#from carPricePredictor.pipeline.prediction import PredictionPipeline
import streamlit as st
import datetime
import pandas as pd
import joblib
import numpy as np


make_and_models = {
    "BMW": ['3 Series', '5 Series', 'X5'],
    "Chevrolet": ['Impala', 'Malibu', 'Silverado 1500'],
    "Chrysler": ['200', '300', 'Town and Country'],
    "Dodge": ['Avenger', 'Charger', 'Grand Caravan'],
    "Ford": ['Fusion', 'Escape', 'F-150'],
    "Honda": ['Accord', 'Civic', 'Odyssey'],
    "Hyundai": ['Elantra', 'Sonata', 'Accent'],
    "Kia": ['Sorento', 'Optima', 'Forte'],
    "Nissan": ['Altima', 'Maxima', 'Sentra'],
    "Toyota": ['Corolla', 'Camry', 'RAV4']
}

transmission_options = ['Manual', 'Automatic']

class PredictionPipeline:
    def __init__(self):
        # Load your model or initialize it here
        pass

    def predict_price(self, X):

        rf_model = joblib.load(os.path.join("model", "random_forest_model.pkl"))

        encoder_data = joblib.load(os.path.join("model", "saved_encoder_data.pkl"))

        le_make = encoder_data["make"]
        le_model = encoder_data["model"]
        le_transmission = encoder_data["transmission"]        
             

             
        X[:, 1] = le_make.fit_transform(X[:,1])
        X[:, 2] = le_model.fit_transform(X[:,2])
        X[:, 3] = le_transmission.fit_transform(X[:,3])

        X = X.astype(float)


        result = rf_model.predict(X)

        # Calculate lower and upper bounds of the range and round to the nearest integer
        lower_bound = round(result[0] * 0.9)
        upper_bound = round(result[0] * 1.1)

        # Convert the values to integers
        lower_bound = int(lower_bound)
        upper_bound = int(upper_bound)
        text1 = " and"
        

        prediction = f"{lower_bound} {text1} {upper_bound}"
        return prediction


def show_predict_page():
    st.title("Car Price Predictor")

    st.write("""
    This application was created for educational purposes and utilizes outdated data, 
    which may not accurately reflect current car prices. This work explores the comprehensive
    lifecycle of a machine learning (ML) project, focusing on the development of a prototype
    application for car price prediction. Emphasizing efficiency, reproducibility, and collaboration, 
    the project integrates essential tools like DVC and MLFlow to establish a robust framework. 
    The work presents both functional and non-functional requirements, providing insights into 
    data management, model training, deployment automation, monitoring, and maintenance. The software 
    architecture and workflow, is emphasizing scalability and modularity. Detailed implementations cover
    data ingestion, preprocessing, model training, evaluation, and prediction. Continuous integration and
    deployment (CI/CD) pipelines are explored, utilizing Amazon Cloud (ECR and EC2) and Streamlit.io for 
    deployment.
    """)

    st.write("""### I need some information to predict the price of your car""")

    # Define models for each make
    make = st.selectbox("Brand", list(make_and_models.keys()))
    model = make_and_models.get(make, [])
    model = st.selectbox("Model", model)
    transmission = st.selectbox("Transmission", transmission_options)
    odometer = st.slider("Mileage", 0, 999999, 0)
    condition = st.slider("Condition", 0, 50, 30)
    year = st.slider("Year", 1990, 2015, 2010)

    # Calculate age of the car
    current_year = datetime.datetime.now().year
    age = current_year - year

    # Create a DataFrame from the user inputs
    user_data = np.array([[year, make, model, transmission, condition, odometer, age]])
 

    # Initialize PredictionPipeline
    pipeline = PredictionPipeline()

    ok = st.button("Predict Price")

    if ok:
        # Call predict_price method of PredictionPipeline class
        result = pipeline.predict_price(user_data)
        st.write("Predicted Price: somewhere between $", result)

    st.write("\n\nCreated by Shoeb Hashem, shoeb.hashem@gmail.com")

if __name__ == "__main__":
    show_predict_page()

