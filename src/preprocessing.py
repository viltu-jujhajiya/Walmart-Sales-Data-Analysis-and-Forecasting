import pandas as pd
from fastapi import HTTPException

def process_features(data):
    try:
        '''Creating date-wise features'''
        data["Date"] = pd.to_datetime(data["Date"])
        data["DayOfWeek"] = (data["Date"].dt.dayofweek + 1) % 7 + 1   
        data["Month"] = data["Date"].dt.month
        data["Year"] = data["Date"].dt.year
        data["WeekOfYear"] = data["Date"].dt.isocalendar().week.astype(int)
        data["WeekOfMonth"] = ((data["Date"].dt.day - 1) // 7) + 1
        
        return data

    except Exception as e:
        print(f"Exception in process_features: {e}")


def encode_features(data_config, data):
    try:
        '''Type encoding'''
        type_encoding = data_config["Type_encoding"]
        data["Type"] = data["Type"].map(type_encoding)

        '''IsHoliday encoding'''
        data["IsHoliday"] = data["IsHoliday"].astype(int)

        return data
    
    except Exception as e:
        print(f"Exception in encode_features: {e}")         


def process_input(data_config, input):
    try:
        input = pd.DataFrame([input.model_dump()])

        processed_data = process_features(input)
        if processed_data["DayOfWeek"].iloc[0] != 6:
            raise HTTPException(status_code=422, detail=f"Day of the week should be saturday, Choose a date that occurs on Friday")
        
        testing_data = encode_features(data_config, processed_data)

        testing_data = testing_data[data_config["test_features"]]

        return testing_data

    except Exception as e:
        print(f"Exception in process_input: {e}")