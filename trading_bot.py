import os
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import talib
import ollama
import logging
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
import requests
import yfinance as yf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

# ===== Configuration =====
# Alpaca API Configuration
API_KEY = 'PK1RTYE41AZHNX2DCN0L'
API_SECRET = 'Gz1HzRUsPbcfgMxMFeT7LuzxfGbkQbJdTJKOLQg9'
BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL

# Trading Configuration
RISK_PER_TRADE = 0.02  # 2% risk per trade
STOP_LOSS_ATR_MULTIPLIER = 1.5  # ATR multiplier for stop loss
TAKE_PROFIT_ATR_MULTIPLIER = 2.0  # ATR multiplier for take profit
MAX_POSITIONS = 3  # Maximum number of concurrent positions
POSITION_SIZE = 0.1  # 10% of account per position
POSITION_SIZE_PERCENTAGE = 0.1  # 10% of account per position

# Market Configuration
MARKET_SYMBOLS = [
    # US Stocks
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC", "PYPL",
    # Indices (using correct Yahoo Finance symbols)
    "^GSPC",  # S&P 500
    "^DJI",   # Dow Jones
    "^IXIC",  # NASDAQ
    "^FTSE",  # FTSE 100
    "^GDAXI", # DAX
    "^FCHI",  # CAC 40
    "^N225",  # Nikkei 225
    "^HSI",   # Hang Seng
    "^STOXX50E" # Euro Stoxx 50
]

# Symbol mapping for yfinance
YFINANCE_SYMBOL_MAP = {
    "^GSPC": "SPY",    # Using SPY ETF as proxy for S&P 500
    "^DJI": "DIA",     # Using DIA ETF as proxy for Dow Jones
    "^IXIC": "QQQ",    # Using QQQ ETF as proxy for NASDAQ
    "^FTSE": "^FTSE",  # FTSE 100
    "^GDAXI": "^GDAXI", # DAX
    "^FCHI": "^FCHI",  # CAC 40
    "^N225": "^N225",  # Nikkei 225
    "^HSI": "^HSI",    # Hang Seng
    "^STOXX50E": "^STOXX50E" # Euro Stoxx 50
}

# Strategy Configuration
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
EMA_FAST = 9
EMA_SLOW = 21
EMA_SLOWER = 50
BB_PERIOD = 20
BB_STD = 2.0

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral:7b"  # Using Mistral 7B model

# News API Configuration
NEWS_API_KEY = '986367fc4934459fa3fdefa6a8a727d0'  # News API key
NEWS_API_URL = 'https://newsapi.org/v2/everything'

# Trading parameters for each symbol
TRADING_PARAMS = {
    # US Stocks
    "AAPL": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "MSFT": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "AMZN": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "GOOGL": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "META": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "TSLA": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "NVDA": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "AMD": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "INTC": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    "PYPL": {
        "position_size": 0.1,
        "profit_target": 100,
        "stop_loss": 50,
        "timeframe": "15Min"
    },
    # Indices
    "^GSPC": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^DJI": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^IXIC": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^FTSE": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^GDAXI": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^FCHI": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^N225": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^HSI": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    },
    "^STOXX50E": {
        "position_size": 0.2,
        "profit_target": 200,
        "stop_loss": 100,
        "timeframe": "15Min"
    }
}

class TradingBot:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url)
        self.positions = {}
        self.last_trade_time = {}
        self.pdt_blocklist = {}
        self.ollama_client = ollama.Client(host=OLLAMA_BASE_URL)

    def round_price(self, price):
        return round(price, 2)

    def get_historical_prices(self, symbol, timeframe='1Min', limit=100):
        try:
            logging.info(f"Fetching historical data for {symbol} with timeframe {timeframe}")
            # Map timeframe to yfinance interval
            interval_map = {
                '1Min': '1m',
                '5Min': '5m',
                '15Min': '15m',
                '1H': '1h',
                '1D': '1d'
            }
            
            # Get yfinance symbol
            yf_symbol = YFINANCE_SYMBOL_MAP.get(symbol, symbol.replace('^', ''))
            logging.info(f"Using yfinance symbol: {yf_symbol}")
            
            # Get data from yfinance
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period='1d', interval=interval_map.get(timeframe, '1m'))
            
            if df.empty:
                logging.error(f"No data available for {symbol}")
                return None
            
            logging.info(f"Retrieved {len(df)} data points for {symbol}")
            
            # Create DataFrame with required columns
            df = pd.DataFrame({
                'timestamp': df.index,
                'open': df['Open'],
                'high': df['High'],
                'low': df['Low'],
                'close': df['Close'],
                'volume': df['Volume']
            })
            
            # Limit to requested number of bars
            if len(df) > limit:
                df = df.tail(limit)
                logging.info(f"Limited data to {limit} bars")
            
            return df
        
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {str(e)}", exc_info=True)
            return None

    def get_news_analysis(self, symbol):
        try:
            if not NEWS_API_KEY:
                return "News analysis disabled"
            
            logging.info(f"Fetching news for {symbol}")
            # Fetch news articles
            params = {
                'q': symbol,
                'apiKey': NEWS_API_KEY,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 5
            }
            response = requests.get(NEWS_API_URL, params=params)
            news_data = response.json()
            
            if news_data['status'] != 'ok' or not news_data['articles']:
                return "No recent news available"
            
            logging.info(f"Retrieved {len(news_data['articles'])} news articles for {symbol}")
            
            # Analyze news sentiment
            news_summary = "Recent News:\n"
            for article in news_data['articles']:
                news_summary += f"- {article['title']}\n"
                news_summary += f"  {article['description']}\n"
            
            return news_summary
        except Exception as e:
            logging.error(f"Error fetching news for {symbol}: {str(e)}", exc_info=True)
            return "Error fetching news"

    def get_ollama_analysis(self, symbol, df):
        try:
            logging.info(f"Getting Ollama analysis for {symbol}")
            current_price = df['close'].iloc[-1]
            price_change = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100

            rsi = talib.RSI(df['close'], timeperiod=14)
            macd, macd_signal, _ = talib.MACD(df['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
            ema_fast = talib.EMA(df['close'], timeperiod=EMA_FAST)
            ema_slow = talib.EMA(df['close'], timeperiod=EMA_SLOW)
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            momentum = talib.MOM(df['close'], timeperiod=10)
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            trend_strength = ((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]) * 100
            news_analysis = self.get_news_analysis(symbol)

            prompt = f"""
    You are a confident, risk-calibrated trading assistant operating in real-time.

    Your task is to decide whether to BUY, SELL, or HOLD {symbol} based on these metrics:

    Price: ${current_price:.2f} ({price_change:.2f}% change)
    RSI: {rsi.iloc[-1]:.2f}
    MACD: {macd.iloc[-1]:.2f}
    MACD Signal: {macd_signal.iloc[-1]:.2f}
    Trend Strength: {trend_strength:.2f}%
    ATR: {atr.iloc[-1]:.2f}
    Momentum: {momentum.iloc[-1]:.2f}
    Volume Ratio: {volume_ratio:.2f}
    News Headlines Sentiment: {news_analysis}

    IMPORTANT:
    - HOLD should be avoided unless signals are perfectly neutral or conflicting.
    - If RSI is under 35 and momentum is negative, strongly consider SELL.
    - If RSI is over 65 and momentum is positive, strongly consider BUY.
    - If MACD is crossing above Signal, and trend is up — BUY.
    - If MACD is below Signal, and trend is down — SELL.

    You must make a decision. Be firm.

    Return only one word: BUY, SELL, or HOLD.
    """

            response = self.ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt)
            decision = response['response'].strip().upper()
            logging.info(f"Ollama decision for {symbol}: {decision}")

            if decision not in ['BUY', 'SELL', 'HOLD']:
                logging.warning(f"Invalid Ollama decision: {decision}")
                return 'HOLD'

            # Override logic only for extreme cases
            if decision == 'BUY' and rsi.iloc[-1] > 75 and trend_strength < -1:
                logging.info("Overriding BUY to HOLD due to overbought and downtrend")
                return 'HOLD'
            elif decision == 'SELL' and rsi.iloc[-1] < 25 and trend_strength > 1:
                logging.info("Overriding SELL to HOLD due to oversold and strong uptrend")
                return 'HOLD'

            return decision
        except Exception as e:
            logging.error(f"Error in Ollama analysis for {symbol}: {str(e)}", exc_info=True)
            return 'HOLD'

    def process_trade_logic(self, symbol, df, decision):
        try:
            logging.info(f"Processing trade logic for {symbol} with decision: {decision}")
            current_price = df['close'].iloc[-1]
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            stop_loss_distance = atr * STOP_LOSS_ATR_MULTIPLIER
            take_profit_distance = atr * TAKE_PROFIT_ATR_MULTIPLIER

            logging.info(f"Current price: ${current_price:.2f}")
            logging.info(f"ATR: {atr:.2f}")
            logging.info(f"Stop loss distance: {stop_loss_distance:.2f}")
            logging.info(f"Take profit distance: {take_profit_distance:.2f}")

            # Skip if symbol already failed due to PDT today
            today = datetime.now().date()
            if hasattr(self, 'pdt_blocklist') and symbol in self.pdt_blocklist.get(today, set()):
                logging.warning(f"Skipping {symbol} due to PDT blocklist for today.")
                return

            position = self.positions.get(symbol)

            # Check if PDT is triggered
            account = self.api.get_account()
            if account.pattern_day_trader == 'true':
                logging.warning(f"PDT restriction active. Skipping {symbol}.")
                self.pdt_blocklist.setdefault(today, set()).add(symbol)
                return
            
            if position:
                logging.info(f"Existing position found for {symbol}")
                # Calculate profit/loss
                entry_price = position['entry_price']
                current_pnl = (current_price - entry_price) / entry_price * 100
                
                logging.info(f"Position details for {symbol}:")
                logging.info(f"Entry Price: ${entry_price:.2f}")
                logging.info(f"Current Price: ${current_price:.2f}")
                logging.info(f"P/L: {current_pnl:.2f}%")
                
                # Check if we should close the position
                if current_pnl >= TRADING_PARAMS[symbol]['profit_target']:
                    logging.info(f"Taking profit on {symbol} at {current_pnl:.2f}%")
                    self.api.close_position(symbol)
                    del self.positions[symbol]
                    return
                elif current_pnl <= -TRADING_PARAMS[symbol]['stop_loss']:
                    logging.info(f"Cutting loss on {symbol} at {current_pnl:.2f}%")
                    self.api.close_position(symbol)
                    del self.positions[symbol]
                    return

            # Calculate position size
            account = self.api.get_account()
            equity = float(account.equity)
            position_size = equity * TRADING_PARAMS[symbol]["position_size"]
            qty = int(position_size / current_price)

            stop_price = round(current_price - stop_loss_distance, 2)
            take_profit_price = round(current_price + take_profit_distance, 2)
            if decision == "BUY":
                logging.info(f"Processing BUY signal for {symbol}")
                try:
                    self.api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="buy",
                        type="market",
                        time_in_force="gtc",
                        order_class="bracket",
                        stop_loss={"stop_price": stop_price},
                        take_profit={"limit_price": take_profit_price}
                    )
                    logging.info(f"Bracket BUY order placed for {symbol}")
                    self.positions[symbol] = {"side": "long", "quantity": qty}
                except Exception as e:
                    logging.error(f"Error placing bracket order for {symbol}: {e}")

            elif decision == "SELL":
                position = self.positions.get(symbol)
                if position and position["side"] == "long":
                    # Exit the long position
                    logging.info(f"Closing long position on {symbol}")
                    try:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=position["quantity"],
                            side="sell",
                            type="market",
                            time_in_force="gtc"
                        )
                        logging.info(f"Exited long position on {symbol}")
                        self.positions.pop(symbol)
                    except Exception as e:
                        logging.error(f"Error exiting position for {symbol}: {e}")
                else:
                    # SHORT entry if allowed
                    try:
                        account = self.api.get_account()
                        if account.pattern_day_trader == 'true':
                            logging.warning(f"PDT restriction active. Skipping short for {symbol}.")
                            return
                        logging.info(f"Entering SHORT position on {symbol}")
                        self.api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side="sell",
                            type="market",
                            time_in_force="gtc",
                            order_class="bracket",
                            stop_loss={"stop_price": stop_price},
                            take_profit={"limit_price": take_profit_price}
                        )
                        logging.info(f"Bracket SHORT order placed for {symbol}")
                        self.positions[symbol] = {"side": "short", "quantity": qty}
                    except Exception as e:
                        logging.error(f"Error placing short bracket order for {symbol}: {e}")


        except Exception as e:
            logging.error(f"Error processing trade logic for {symbol}: {str(e)}", exc_info=True)

    def run(self):
        logging.info("Starting trading bot")
        while True:
            try:
                for symbol in MARKET_SYMBOLS:
                    logging.info(f"\nAnalyzing {symbol}...")
                    
                    # Get historical data
                    df = self.get_historical_prices(
                        symbol,
                        timeframe=TRADING_PARAMS[symbol]['timeframe']
                    )
                    
                    if df is None or len(df) < 50:
                        logging.warning(f"Not enough data for {symbol}")
                        continue
                    
                    # Get Ollama's analysis
                    decision = self.get_ollama_analysis(symbol, df)
                    logging.info(f"Decision for {symbol}: {decision}")
                    
                    # Process trade logic
                    self.process_trade_logic(symbol, df, decision)
                    
                    # Sleep to avoid rate limits
                    time.sleep(1)
                
                # Sleep before next iteration
                logging.info("Completed analysis cycle, waiting for next iteration...")
                time.sleep(60)
                
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}", exc_info=True)
                time.sleep(60)

if __name__ == "__main__":
    logging.info("Initializing trading bot")
    bot = TradingBot(API_KEY, API_SECRET, BASE_URL)
    bot.run()
