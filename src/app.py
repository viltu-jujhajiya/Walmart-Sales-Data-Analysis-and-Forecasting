from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from preprocessing import process_input
from forecast import forecast_sales, config

    
class ForecastInput(BaseModel):
    Store: int
    Date: str
    IsHoliday: bool
    Temperature: float
    Fuel_Price: float
    CPI: float
    Unemployment: float
    Type: Literal['A', 'B', 'C']
    Size: int
    lag1: float
    lag4: float
    RollingMean_4Weeks: float


app = FastAPI()


@app.post("/walmart/weeklysales/forecast")
async def weekly_forecast(request_data: ForecastInput):
    try:
        processed_input = process_input(config["data"], request_data)

        forecasted_sales = forecast_sales(processed_input)

        return {"message": f"Request validated successfully. Forecasted value: {forecasted_sales}"}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input data: {e}")

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5667)