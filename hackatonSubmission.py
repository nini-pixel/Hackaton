# A copy-paste

import pandas as pd
import numpy as np
from lets_plot import *
from statsmodels.api import OLS, add_constant
from statsmodels.stats.sandwich_covariance import cov_hac, cov_white_simple
from statsmodels.tsa.stattools import adfuller
import requests
from io import StringIO
import yfinance as yf
import scipy.optimize as opt
import json
import time
from typing import Dict, Optional
import math
from tenacity import retry, stop_after_attempt, wait_random_exponential

LetsPlot.setup_html()

# Load CPI data
cpiData = pd.read_csv('cpi_index_all_00-25.csv', skiprows=12, 
                     names=['Year', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 'Half1', 'Half2'])
cpiData.set_index('Year', inplace=True)

# Ticker lists (pre-validated)
tickers00 = ["SCCO", "VALE", "GGB", "MO", "BOOM", "BPT", "DECK", "BBD", "AMED",
            "CLH", "BRFS", "MNST", "MED", "BWEN", "HUSA", "SID", "AIMD", "CIG"]
tickers07 = ["UAVS"]
tickers08 = ["AIMD", "NIXX", "TOMZ"]
tickers10 = ["ARMN"]
tickers13 = ["NUTX"]
tickers14 = ["PETV"]
tickers15 = ["BLNE", "AIMD", "GRNQ", "FNGR", "PNBK", "AVXL", "ENVB", "DLPN",
            "WKHS", "AUID", "AMBO", "THTX"]
tickers17 = ["AQB", "FRHC", "HIVE"]
tickers18 = ["ALAR", "ALBT"]
tickers19 = ["AXSM", "FTLF", "SOBR", "TNK", "NCRA", "OXBRW", "KOD", "QIPT", 
            "SBEV", "RLMD", "EVER", "RCEL", "DRRX", "IVDA", "CTM", "NCPL",
            "NISN", "BATL", "MBOT", "SAVA", "OESX", "CDLX", "NGNE", "ENPH", "PNTG"]
tickers20 = ["FFIE"]
tickers21 = ["GFRX"]
tickers22 = ["APCXM"]
tickers23 = ["AZTR", "MDXH"]
tickers24 = ["WLDSW", "MTEKW", "NXLIW", "RGTIW", "TSSI", "SNYR", "RZLVW", "GRRRW",
            "AISPW", "SOUNW", "WGS", "DRUG", "ILLRW", "KULR", "QUBT", "LENZ", "PDYN",
            "FLDDW", "WGSWW", "ZIVO", "RVSNW", "RGTI", "RCAT", "PSIX", "MNPR"]
tickersIndexes = ["DGNX", "AIMAW", "MGAM", "AREBW", "DATSW", "RGC", "GATEW", "HONDW",
                "YOSH", "WLDSW", "LXEH", "MNDR", "FTFT", "CRVO", "TOI", "TWNPV", "MLGO",
                "GITS", "DOMH", "TWNP", "GATE", "NMAX", "ABTS", "SKBL", "GSPC", "MSCI", "AAPL", "DJI"]

# API Configuration
URL = "mts-prism.com"
PORT = 8082
TEAM_API_CODE = "d747eea0e03cea824389395740436f6d"

# Helper Functions
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=10))
def safe_yfinance_request(ticker):
    """Safe wrapper for yfinance requests with retries"""
    try:
        t = yf.Ticker(ticker)
        time.sleep(0.2)  # Rate limiting protection
        return t.info
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return None

def stockPrice(stockName, T0, T1):
    """Get stock prices with error handling"""
    try:
        data = yf.download(stockName, start=T0, end=T1, progress=True)
        if data.empty:
            return None, None
        return round(data.iloc[0, 0], 4), round(data.iloc[-1, 0], 4)
    except Exception:
        return None, None

def stockSector(stockName):
    """Get sector with error handling"""
    info = safe_yfinance_request(stockName)
    return info.get("sector") if info else None

def stockIndustry(stockName):
    """Get industry with error handling"""
    info = safe_yfinance_request(stockName)
    return info.get("industry") if info else None

def systematicRisk(stockName):
    """Get beta with error handling"""
    info = safe_yfinance_request(stockName)
    return info.get("beta") if info else None

def inflationRisk(stockName, T0, T1):
    """Calculate inflation-adjusted returns"""
    try:
        startPrice, endPrice = stockPrice(stockName, T0, T1)
        if None in (startPrice, endPrice):
            return 1
        
        nominalReturn = (endPrice - startPrice) / startPrice
        
        yearStart, monthStart = map(int, T0.split("-"))
        yearEnd, monthEnd = map(int, T1.split("-"))
        
        endCPI = cpiData.loc[yearStart, monthStart]
        beginningCPI = cpiData.loc[yearEnd, monthEnd]
        inflationRate = (endCPI - beginningCPI) / beginningCPI
        
        if (nominalReturn - inflationRate) > 0:
            return 1 
        else:
            return 0
    except Exception:
        return 0

def liquidityRisk(stockName):
    """Check liquidity with error handling"""
    try:
        info = safe_yfinance_request(stockName)
        if not info:
            return 1
            
        avg_volume = info.get("averageVolume", 0)
        market_cap = info.get("marketCap", 0)
        
        if avg_volume > 100000 or market_cap > 500000 :
            return 1 
        else:
            return 0
    except Exception:
        return 0

# Core Portfolio Functions
def getTickerGroup(T0, T1):
    """Get appropriate tickers for time period"""
    year = int(T1.split("-")[0])
    tickerList = []
    
    # Base tickers
    if year <= 2010:
        tickerList.extend(tickers00 + tickersIndexes)
    elif 2010 < year < 2020:
        tickerList.extend(tickers10 + tickersIndexes)
    else:
        tickerList.extend(tickersIndexes)
    
    # Year-specific additions
    year_mapping = {
        2007: tickers07, 2008: tickers08, 2010: tickers10,
        2013: tickers13, 2014: tickers14, 2015: tickers15,
        2017: tickers17, 2018: tickers18, 2019: tickers19,
        2020: tickers20, 2021: tickers21, 2022: tickers22,
        2023: tickers23, 2024: tickers24
    }
    
    if year in year_mapping:
        tickerList.extend(year_mapping[year])
    
    return list(set(tickerList))

def choosingStocks(tickerList, T0, T1, avoid_sectors=[]):
    """Filter stocks based on criteria"""
    stockList = []
    print(f"\n{'='*50}\nStarting stock selection for {T0} to {T1}")
    print(f"Avoiding sectors: {avoid_sectors}")
    
    for ticker in tickerList:
        try:
            flag = 0

            # Price data check
            startPrice, endPrice = stockPrice(ticker, T0, T1)
            if None in (startPrice, endPrice):
                print(f"{ticker}: Data currently unavailable...")
                continue

            # Get sector and basic info
            info = safe_yfinance_request(ticker)
            if not info:
                print(f"{ticker}: Data currently unavailable...")
                continue
                
            sector = info.get("sector")
            if not sector:
                print(f"{ticker}: No sector data.")
                continue
                
            # Sector filter
            if sector in avoid_sectors:
                print(f"{ticker}: Let's exclude this stock as it is part of the ({sector}) sector.")
                answer = input("Do you wish to change your mind and INCLUDE this stock? (yes/no): ")
                if answer == "yes" or "Yes" or "YES":
                    print(f"Shall we {ticker} continue?")
                    continue
                else:
                    print(f"Excluding {ticker}")
                    continue 
            
            # Systematic risk
            if systematicRisk(ticker) > 1.2:
                flag += 1
                print(f"{ticker}: This stock is quite volatile, but you can diversify it away")
                continue

            # Liquidity filter
            if liquidityRisk(ticker) == 0:
                flag += 1
                if flag == 2:
                    print(f"{ticker}: Do you really want this stock? It is quite risky...")
                print(f"{ticker}: Be careful, you might not be able to retrieve your cash!")
                continue
                
            # Inflation risk check
            if inflationRisk(ticker, T0, T1) == 0:
                flag += 1
                print(f"{ticker}: Do not invest! You will lose the value of your investment.")
                if flag > 1: 
                    "This was a terrible investment anyways..."
                continue
                
            # All checks passed
            stockList.append(ticker)
            print(f"{ticker}: Accepted")
            
        except Exception as e:
            print(f"{ticker}: Error during processing - {str(e)}")
            continue
    
    print(f"\nSelected {len(stockList)} stocks: {stockList}")
    return stockList

def calculate_risk_tolerance_score(age, salary, budget):
    """Calculate risk tolerance"""
    age_factor = 1 - (age / 100)
    salary_factor = min(salary / 200000, 1)
    budget_factor = min(budget / 50000, 1)
    return (0.4 * age_factor) + (0.3 * salary_factor) + (0.3 * budget_factor)

def calc_weight(riskAversion, investmentReturn, stockName):
    """Calculate stock weight in portfolio"""
    beta = systematicRisk(stockName)
    if beta and beta > 0 and riskAversion > 0:
        return abs(investmentReturn) / (riskAversion * beta)
    return 0

def calc_amount_of_stock_to_buy(stockList, riskAversion, T0, T1, budget):
    """Calculate optimal stock quantities"""
    if not stockList:
        return []
    
    portfolio_data = []
    for stock in stockList:
        startPrice, endPrice = stockPrice(stock, T0, T1)
        return_pct = (endPrice - startPrice) / startPrice
        weight = calc_weight(riskAversion, return_pct, stock)
        current_price = endPrice
        
        if weight > 0 and current_price > 0:
            portfolio_data.append({
                'ticker': stock,
                'weight': weight,
                'price': current_price
            })
    
    if not portfolio_data:
        return []
    
    total_weight = sum(item['weight'] for item in portfolio_data)
    portfolio = []
    for item in portfolio_data:
        shares = int((item['weight'] / total_weight * budget) // item['price'])
        if shares > 0:
            portfolio.append((item['ticker'], shares))
    
    return portfolio

# API Communication Functions
def send_get_request(path):
    """Send GET request to server"""
    headers = {"X-API-Code": TEAM_API_CODE}
    response = requests.get(f"http://{URL}:{PORT}/{path}", headers=headers)
    return (True, response.text) if response.status_code == 200 else (False, response.text)

def send_post_request(path, data=None):
    """Send POST request to server"""
    headers = {"X-API-Code": TEAM_API_CODE, "Content-Type": "application/json"}
    response = requests.post(f"http://{URL}:{PORT}{path}", 
                           data=json.dumps(data), 
                           headers=headers)
    return (True, response.text) if response.status_code == 200 else (False, response.text)

def get_context():
    """Get client context"""
    return send_get_request("/request")

def send_portfolio(weighted_stocks):
    """Submit portfolio to server"""
    data = [{"ticker": stock[0], "quantity": stock[1]} for stock in weighted_stocks]
    return send_post_request("/submit", data=data)

def parse_nested_json(json_str):
    """Parse nested JSON response"""
    try:
        outer_dict = json.loads(json_str)
        if 'message' in outer_dict:
            return json.loads(outer_dict['message'])
        return outer_dict
    except Exception:
        return {}

# Main Execution
if __name__ == "__main__":
    # Get client context
    success, context = get_context()
    if not success:
        print(f"Error getting context: {context}")
        exit()
    result = parse_nested_json(context)
    '''
    result = {
      "timestamp": "2025-04-06T05:29:26.155091195Z",
      "start": "2014-06-06",
      "end": "2014-08-09",
      "age": 27,
      "employed": True,
      "salary": 28036,
      "budget": 8483,
      "dislikes": [
        "Crypto Assets",
        "Finance or Crypto Assets"
      ]
    }
    '''
    print("\nClient Requirements:")
    print(f"Period: {result['start']} to {result['end']}")
    print(f"Disliked Sectors: {result['dislikes']}")
    print(f"Age: {result['age']}, Salary: ${result['salary']:,}, Budget: ${result['budget']:,}")
    
    # Get and filter stocks
    potential_list = getTickerGroup(result["start"], result["end"])
    
    my_stock = choosingStocks(potential_list, result["start"], result["end"], 
                            avoid_sectors=result["dislikes"])
    # Fallback if no stocks pass filters
    if not my_stock:
        print("\n No stocks passed all filters. Using fallback...")
        my_stock = [t for t in potential_list 
                   if stockPrice(t, result["start"], result["end"])[0] is not None][:5]
        print(f"Fallback Stocks: {my_stock}")
    
    # Calculate portfolio
    risk_score = calculate_risk_tolerance_score(result["age"], result["salary"], result["budget"])
    print(f"\nRisk Tolerance Score: {risk_score:.2f}")
    
    portfolio = calc_amount_of_stock_to_buy(my_stock, risk_score, 
                                          result["start"], result["end"], 
                                          result["budget"])
    
    if not portfolio:
        print("\n Error: Could not construct portfolio")
    else:
        print("\n Final Portfolio:")
        for stock, amount in portfolio:
            print(f"{stock}: {amount} shares")
        
        # Submit portfolio
        success, response = send_portfolio(portfolio)
        if success:
            print(f"\n Portfolio submitted successfully!")
            used_budget = sum(price * qty for ticker, qty in portfolio if (price := yf.Ticker(ticker).info.get("currentPrice", 0)))
            print(f"\n🧾 Used budget: ${used_budget:,.2f} / ${result['budget']:,.2f}")

        else:
            print(f"\n Submission failed.")
    
