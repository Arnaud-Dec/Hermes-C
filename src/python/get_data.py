import yfinance as yf
import os

os.makedirs("data", exist_ok=True)

actif = yf.Ticker("BTC-USD")

data = actif.history(period="2y", interval="1d")

data = data[["Close"]]
data = data.dropna()

print("Save in data/data.csv...")
data.to_csv("data/data.csv")
print("Finish")