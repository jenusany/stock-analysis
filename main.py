import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = '2015-01-01'
TODAY = date.today().strftime("%Y-%m-%d")

st.markdown(
    "<div style='text-align: center;'>"
    "<h1>Stock Prediciton App</h1>"
    "<br>"
    "</div>",
    unsafe_allow_html=True
)

tickers = yf.Tickers('')

stocks = ("AAPL", "GOOG")
selected_stocks = st.selectbox("Select Stock Ticker", stocks)

n_years = st.slider("Years of prediciton:", 1,4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_stocks)
data_load_state.text("Done")

st.subheader("Raw data")
st.write(data.tail(5))

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
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write('forecast data')
forecastData = plot_plotly(m, forecast)
st.plotly_chart(forecastData)

st.write('forecast components')
forecastComp = m.plot_components(forecast)
st.write(forecastComp)