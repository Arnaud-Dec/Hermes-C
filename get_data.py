import yfinance as yf

actif = yf.Ticker("BTC-USD")

data = actif.history(period="2y" , interval="1d")

data = data[["Close"]] # on garde que les close
data = data.dropna() # on dégage les valeur vide
data.to_csv("data.csv")