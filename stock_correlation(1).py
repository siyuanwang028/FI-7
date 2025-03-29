import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_stock_data(tickers, period):
    """
    tickers: stock code list
    period: time period(e.g. '1y', '2y', '5y')
    return: DataFrame with closing prices of each stock
    """
    data = yf.download(tickers, period=period)['Close']
    
    # if only one stock, convert it to a DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=[tickers])
    
    return data

def calculate_returns(prices):
    """calculate daily returns"""
    return prices.pct_change().dropna()

def calculate_rolling_correlation(returns, window=30):
    if len(returns.columns) == 2:
        return returns.iloc[:, 0].rolling(window=window).corr(returns.iloc[:, 1])
    else:
        return None

def calculate_rolling_volatility(returns, window=30):
    return returns.rolling(window=window).std() * np.sqrt(252)  # annualized

def calculate_portfolio_volatility(vol1, vol2, corr, weight1, weight2):
    return np.sqrt(
        weight1**2 * vol1**2 + 
        weight2**2 * vol2**2 + 
        2 * weight1 * weight2 * vol1 * vol2 * corr
    )

def task1():
    """execute task1: analyze the correlation and portfolio risk between two stocks"""
    print("===== task1: analyze the correlation between two stocks =====")
    
    # 1. user input
    stock1 = input("input the first stock code (e.g. AAPL): ").strip().upper()
    stock2 = input("input the second stock code (e.g. MSFT): ").strip().upper()
    
    period_input = input("input the time period (e.g. 1y for one year): ").strip().lower()
    
    weight1_input = float(input(f"input the weight of {stock1} (percentage, e.g. 60): "))
    weight2_input = float(input(f"input the weight of {stock2} (percentage, e.g. 40): "))
    
    # verify the sum of weights is 100%
    if abs(weight1_input + weight2_input - 100) > 0.01:
        print("error: the sum of weights must be 100%")
        return
    
    # convert weights to decimals
    weight1 = weight1_input / 100
    weight2 = weight2_input / 100
    
    # 2. get data
    tickers = [stock1, stock2]
    try:
        prices = get_stock_data(tickers, period_input)
    except Exception as e:
        print(f"error: {e}")
        return
    
    # 3. calculate
    returns = calculate_returns(prices)
    
    # correlation
    rolling_corr = calculate_rolling_correlation(returns)
    
    # volatility
    vol1 = calculate_rolling_volatility(returns[stock1])
    vol2 = calculate_rolling_volatility(returns[stock2])
    
    # portfolio volatility
    portfolio_vol = pd.Series(index=rolling_corr.index)
    for date in rolling_corr.index:
        if pd.notna(rolling_corr[date]) and pd.notna(vol1[date]) and pd.notna(vol2[date]):
            portfolio_vol[date] = calculate_portfolio_volatility(
                vol1[date], vol2[date], rolling_corr[date], weight1, weight2
            )
    
    # 4. plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # correlation
    ax1.set_xlabel('date')
    ax1.set_ylabel('30 days rolling correlation', color='blue')
    ax1.plot(rolling_corr.index, rolling_corr, 'b-', label='correlation')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim([-1.1, 1.1])
    
    # portfolio volatility
    ax2 = ax1.twinx()
    ax2.set_ylabel('30 days rolling portfolio volatility (%)', color='red')
    ax2.plot(portfolio_vol.index, portfolio_vol * 100, 'r-', label='portfolio volatility')
    ax2.tick_params(axis='y', labelcolor='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title(f'{stock1} and {stock2} correlation and portfolio volatility')
    plt.tight_layout()
    
    # output
    latest_corr = rolling_corr.iloc[-1] if not rolling_corr.empty else None
    latest_vol = portfolio_vol.iloc[-1] * 100 if not portfolio_vol.empty else None
    
    print("\nlatest value:")
    print(f"30 days rolling correlation: {latest_corr:.4f}")
    print(f"portfolio volatility: {latest_vol:.4f}%")
    
    plt.show()

def task2():
    """execute task2: analyze the correlation between multiple stocks"""
    print("\n===== task2: analyze the correlation between multiple stocks =====")
    
    # 1. user input
    tickers_input = input("input the stock codes (max 10, separated by commas, e.g. AAPL,MSFT,GOOGL): ").strip().upper()
    tickers = [ticker.strip() for ticker in tickers_input.split(',')]
    
    # error handling
    if len(tickers) > 10:
        print("error: max 10 stocks")
        return
    if len(tickers) < 2:
        print("error: at least 2 stocks")
        return
    
    period_input = input("input the time period (e.g. 1y for one year): ").strip().lower()
    
    # 2. get data
    try:
        prices = get_stock_data(tickers, period_input)
    except Exception as e:
        print(f"error: {e}")
        return
    
    # 3. calculate
    returns = calculate_returns(prices)
    
    # correlation matrix
    recent_returns = returns.tail(30)
    correlation_matrix = recent_returns.corr()
    
    # 4. plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1,
                linewidths=0.5, fmt='.2f')
    plt.title('correlation heatmap of the latest 30 days')
    plt.tight_layout()
    
    # 5. output
    plt.show()

def main():
    print("stock correlation and portfolio risk analysis")
    print("------------------------------------------")
    
    while True:
        print("\nselect task:")
        print("1. analyze the correlation and portfolio risk between two stocks")
        print("2. analyze the correlation between multiple stocks")
        print("0. exit")
        
        choice = input("\ninput the option (0-2): ")
        
        if choice == '1':
            task1()
        elif choice == '2':
            task2()
        elif choice == '0':
            print("program exited.")
            break
        else:
            print("invalid option, please input again.")

if __name__ == "__main__":
    main() 