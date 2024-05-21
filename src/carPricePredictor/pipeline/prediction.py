import os
import joblib



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








