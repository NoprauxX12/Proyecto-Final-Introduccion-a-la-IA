import yfinance as yf
import pandas as pd

# Descarga USD/COP en 1H (últimos 2 años máximo gratis)
df = yf.download("COP=X", period="730d", interval="1h")
df = df.reset_index()
df = df.rename(columns={"Datetime": "Date", "Open": "Open", "High": "High", "Low": "Low", "Close": "Close"})
df["Date"] = df["Date"].dt.strftime("%Y.%m.%d %H:%M")
df["Volume"] = 1
df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
df.to_csv("datasets/USDCOP_1h_data.csv", sep=";", index=False)
print(f"Guardado: {len(df)} filas")
print(df.head(3).to_string())