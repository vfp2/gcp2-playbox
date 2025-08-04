#!/usr/bin/python3

import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load credentials from .env
load_dotenv()
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'  # Change to live URL if needed

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, base_url=BASE_URL)

# ETFs tracking VIX and S&P 500
symbols = ['SPY', 'IVV', 'VOO', 'VXX', 'UVXY']

# Fetch and print latest trade prices
for symbol in symbols:
    try:
        trade = api.get_latest_trade(symbol)
        print(f"{symbol} - Price: ${trade.price}")
    except Exception as e:
        print(f"{symbol} - Error: {e}")

