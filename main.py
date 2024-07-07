import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import datetime

START = '2015-01-01'
END = date.today().strftime("%Y-%m-%d")

st.markdown(
    "<div style='text-align: center;'>"
    "<h1>Stock Analysis</h1>"
    "<br>"
    "</div>",
    unsafe_allow_html=True
)

user_input = st.text_input("Enter Ticker", "")

    
    
# Display the input
st.button("Analyze")

def load_data(ticker, start = START, end = END):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data = load_data(user_input)
index = len(data) - 1

st.markdown(
    "<div style='text-align: center;'>"
    "<h2>Current Data</h2>"
    "<br>"
    "</div>",
    unsafe_allow_html=True
)

chartData = data.tail(365)[::-1]

formatted_dates = [datetime.strptime(str(date)[:-9], "%Y-%m-%d").strftime("%B %d, %Y") for date in chartData["Date"]]
chartData["Date"] = formatted_dates

st.write("1 Year Raw Data")

st.write(chartData)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()
    

df_train = data[['Date', "Close"]]
df_train = df_train.rename(columns={
        "Date": "ds",
        "Close": "y"
                                    })

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write('forecast data')
forecastData = plot_plotly(m, forecast)
st.plotly_chart(forecastData)

