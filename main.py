import corr_asset

print("hello")

#Find correlation
#Apply DTW 
#Make a prediction
start_date= '2022-01-01'
end_date='2022-12-31'

p1 = corr_asset.corr_cal(start_date, end_date)
""" sp500_allCorr = p1.allSP500()
print(sp500_allCorr)
print(sp500_allCorr[7][0])
#dtw = p1.calculate_DTW(sp500_allCorr[7][0])
p1.plot_correlation_data(sp500_allCorr[7][0],sp500_allCorr[7][1]) """



#all_coords = p1.plot_dtw(('KEY', 'FITB'))
#print(all_coords)
p1.predict_price(('KEY', 'FITB'), new_symbol1_price=15)
#p1.predict_price(('UDR', 'CPT'), new_symbol1_price= 60)