import gradio as gr
import pandas as pd
import joblib


def clip_outliers(X):
    import pandas as pd
    X = pd.DataFrame(X)
    for col in X.columns:
        X[col] = X[col].clip(
            X[col].quantile(0.01),
            X[col].quantile(0.99)
        )
    return X.values


def add_features(X):
    import pandas as pd
    X = pd.DataFrame(X)

    solids = X.iloc[:, 2]
    conductivity = X.iloc[:, 5]
    organic = X.iloc[:, 6]

    X["solids_per_conductivity"] = solids / (conductivity + 1e-9)
    X["organic_ratio"] = organic / (solids + 1e-9)

    return X.values


#load model

model = joblib.load("water_potability_model.pkl")


#prediction logic

def predict_water(
    ph,
    Hardness,
    Solids,
    Chloramines,
    Sulfate,
    Conductivity,
    Organic_carbon,
    Trihalomethanes,
    Turbidity
):
    input_data = pd.DataFrame([{
        "ph": ph,
        "Hardness": Hardness,
        "Solids": Solids,
        "Chloramines": Chloramines,
        "Sulfate": Sulfate,
        "Conductivity": Conductivity,
        "Organic_carbon": Organic_carbon,
        "Trihalomethanes": Trihalomethanes,
        "Turbidity": Turbidity
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    result = "Drinkable" if pred == 1 else "Not Drinkable"
    return result, round(prob, 3)


#interface
app = gr.Interface(
    fn=predict_water,
    inputs=[
        gr.Number(label="pH"),
        gr.Number(label="Hardness"),
        gr.Number(label="Solids"),
        gr.Number(label="Chloramines"),
        gr.Number(label="Sulfate"),
        gr.Number(label="Conductivity"),
        gr.Number(label="Organic Carbon"),
        gr.Number(label="Trihalomethanes"),
        gr.Number(label="Turbidity"),
    ],
    outputs=[
        gr.Text(label="Prediction"),
        gr.Number(label="Drinkable Probability"),
    ],
    title="Water Potability Prediction",
    description="Predict whether water is safe for drinking"
)

app.launch()
