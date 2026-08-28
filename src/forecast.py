from xgboost import XGBRegressor
import yaml


with open("config.yaml") as f:
    config = yaml.safe_load(f)


model_path = config["model"]["model_path"]
model = XGBRegressor()
model.load_model(model_path)

def forecast_sales(input_data):
    try:
        forecast = model.predict(input_data)
        forecast = forecast[0]

        return forecast

    except Exception as e:
        print(e)
    