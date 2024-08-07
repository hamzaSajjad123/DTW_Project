import numpy as np
import yfinance as yf
yf.pdr_override() 
from pandas_datareader import data as pdr
import pandas as pd
from datetime import time
import seaborn as sns
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from dtw import *
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


class corr_cal:
    def __init__(self, start_date, end_date) -> None:
        self.start_date = start_date
        self.end_date = end_date

    def load(self,stuff):
        pass

    def allSP500(self):
        sp500_components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        sp500_symbols = sp500_components['Symbol'].tolist()

        closing_prices_list = []
        closing_prices_columns = []

        for i in sp500_symbols:
            stock_data = yf.download(i, start=self.start_date, end=self.end_date)
            if not stock_data.empty:
                closing_prices_list.append(stock_data['Adj Close'])
                closing_prices_columns.append(i)

        closing_prices = pd.concat(closing_prices_list, axis=1)
        closing_prices.columns = closing_prices_columns  # Set column names to symbol?

        corr_matrix = closing_prices.corr()

        #CORRELATION + FILTER
        corr_pairs = corr_matrix.unstack()
        sorted_corr_pairs = corr_pairs.sort_values(ascending=False)
        filtered_corr_pairs = sorted_corr_pairs[sorted_corr_pairs < 1.0]

        #N correlated pairs
        N = 100 
        top_corr_pairs = filtered_corr_pairs.head(N)
        seen_pairs = set()
        ret_arr = []
        for pair, correlation in top_corr_pairs.items():
            symbol1, symbol2 = pair[0], pair[1]
            if symbol1 != symbol2 and (symbol1, symbol2) not in seen_pairs and (symbol2, symbol1) not in seen_pairs:
                seen_pairs.add((symbol1, symbol2))
                seen_pairs.add((symbol2, symbol1))

                ret_arr.append(((symbol1, symbol2), correlation))
                #print(f"{symbol1} and {symbol2}: {correlation:.4f}")
        
        return ret_arr
      
    def facebook_info(self):
        GetFacebookInformation = yf.Ticker("AAPL")
        print("Company Sector : ", GetFacebookInformation.info['sector'])
        print("Price Earnings Ratio : ", GetFacebookInformation.info['trailingPE'])
        print(" Company Beta : ", GetFacebookInformation.info['beta'])

    def plot_correlation_data(self, symbol_pair, correlation):
        symbol1, symbol2 = symbol_pair
    
        data1 = yf.download(symbol1, start=self.start_date, end=self.end_date)
        data2 = yf.download(symbol2, start=self.start_date, end=self.end_date)

        adj_close1 = data1['Adj Close']
        adj_close2 = data2['Adj Close']
        
        plt.figure(figsize=(10, 6))
        plt.plot(adj_close1, label=symbol1)
        plt.plot(adj_close2, label=symbol2)
        plt.title(f'Correlation Plot: {symbol1} vs. {symbol2} (Correlation: {correlation:.4f})')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()

    def calculate_DTW(self, symbol_set):
        symbol1,symbol2 = symbol_set
        #start_date = '2023-01-01'
        #end_date = '2023-11-01'

        data1 = yf.download(symbol1, start=self.start_date, end=self.end_date)
        data2 = yf.download(symbol2, start=self.start_date, end=self.end_date)
        adj_close1 = data1['Adj Close']
        adj_close2 = data2['Adj Close']

        # BADDDDDDDDDDDDDD
        distance, _ = fastdtw(adj_close1.values, adj_close2.values)

        return distance
    
    def plot_dtw(self, symbol_set):
        symbol1, symbol2 = symbol_set

        data1 = yf.download(symbol1, start=self.start_date, end=self.end_date)
        data2 = yf.download(symbol2, start=self.start_date, end=self.end_date)
        adj_close1 = data1['Adj Close']
        adj_close2 = data2['Adj Close']
        time = data1.index

        # DTW of lines
        alignment = dtw(adj_close1.values, adj_close2.values, keep_internals=True, step_pattern=rabinerJuangStepPattern(6, "c")) # CHANGE RABINER PATTERN = 6

        
        plt.figure(figsize=(10, 6))
        plt.plot(time, adj_close1, label=symbol1)
        plt.plot(time, adj_close2, label=symbol2)

        # Storeing coordinates 
        dtw_coordinates = []
        for i, j in zip(alignment.index1, alignment.index2):
            coordinates = {"x1": time[i], "y1": adj_close1[i], "x2": time[j], "y2": adj_close2[j]}
            dtw_coordinates.append(coordinates)
            plt.plot([time[i], time[j]], [adj_close1[i], adj_close2[j]], color='red', linestyle='--', linewidth=0.5)

        print("DTW Coordinates:", dtw_coordinates)

        # The allignment LINES
        custom_lines = [Line2D([0], [0], color='red', linestyle='--', linewidth=0.5)]
        plt.legend(custom_lines, ['Alignment Path'])
        plt.title(f'DTW Alignment: {symbol1} and {symbol2}')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close Price')
        plt.legend()
        plt.show()
        return dtw_coordinates

    def predict_price(self, symbol_set,new_symbol1_price):
        dtw_coordinates = self.plot_dtw(symbol_set)
        features = np.array([[coord["y1"]] for coord in dtw_coordinates]) 
        target = np.array([coord["y2"] for coord in dtw_coordinates])

        features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # TRAIN REGRESSION MODEL
        model = LinearRegression() 
        model.fit(features_train, target_train)
        predictions = model.predict(features_test)
        mse = mean_squared_error(target_test, predictions) # LOW MSE BETTER
        print("MSE:", mse)
        predicted_symbol2_price = model.predict([[new_symbol1_price]])
        print("Predicted Price for Symbol 2:", predicted_symbol2_price[0])





