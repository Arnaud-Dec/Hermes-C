import yfinance as yf
import os

# --- Configuration ---
TICKER = "BTC-USD"
PERIOD = "2y"
INTERVAL = "1d"
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "data.csv")

def main():
    print(f"[INFO] Starting data download for {TICKER}...")

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download data
    try:
        actif = yf.Ticker(TICKER)
        df = actif.history(period=PERIOD, interval=INTERVAL)
        
        if df.empty:
            print("[ERROR] No data downloaded. Check your internet or the ticker name.")
            return

        # Keep only Close price and remove empty rows
        df = df[["Close"]]
        df = df.dropna()

        # Save to CSV
        df.to_csv(CSV_PATH)
        print(f"[SUCCESS] Data saved to {CSV_PATH} ({len(df)} rows)")

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")

if __name__ == "__main__":
    main()