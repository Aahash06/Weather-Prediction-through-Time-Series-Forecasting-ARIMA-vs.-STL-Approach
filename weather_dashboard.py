import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error

# Load data
weather_df = pd.read_csv(r'C:\Users\ASUS\Documents\nagpur.csv\nagpur.csv', parse_dates=['date_time'], index_col='date_time')

# Define features based on available columns in the dataset
features = ['maxtempC', 'mintempC']  # Add more features if needed

# Define app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Weather Forecasting Dashboard"),
    html.Label("Select Feature:"),
    dcc.Dropdown(
        id='feature-dropdown',
        options=[{'label': feature, 'value': feature} for feature in features],
        value=features[0]
    ),
    dcc.Graph(id='time-series-plot')
])

# Define function to train ARIMA model
def train_arima_model(data, feature):
    model = ARIMA(data[feature], order=(5,1,0))
    arima_model = model.fit()
    return arima_model

# Define function to perform STL decomposition and forecast
def forecast_stl(data, feature, steps):
    stl = STL(data[feature], seasonal=13)
    res = stl.fit()
    trend = res.trend
    seasonal = res.seasonal
    residual = res.resid
    
    # Forecast by adding the trend and seasonal components
    forecast = trend + seasonal
    
    return forecast[-steps:]

# Define callback to update time series plot
@app.callback(
    Output('time-series-plot', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_time_series_plot(selected_feature):
    try:
        # ARIMA
        arima_model = train_arima_model(weather_df, selected_feature)
        arima_forecast = arima_model.forecast(steps=len(weather_df))

        # STL
        stl_forecast = forecast_stl(weather_df, selected_feature, len(weather_df))

        # Create traces for actual values and forecasts
        trace_actual = go.Scatter(x=weather_df.index, y=weather_df[selected_feature], mode='lines', name='Actual')
        trace_arima = go.Scatter(x=weather_df.index, y=arima_forecast, mode='lines', name='ARIMA Forecast')
        trace_stl = go.Scatter(x=weather_df.index, y=stl_forecast, mode='lines', name='STL Forecast')

        return {
            'data': [trace_actual, trace_arima, trace_stl],
            'layout': go.Layout(
                title=f"{selected_feature} Forecast",
                xaxis={'title': 'Date'},
                yaxis={'title': 'Temperature (C)'}
            )
        }
    except Exception as e:
        print(f"Error updating time-series plot: {e}")
        return {'data': [], 'layout': {}}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
