import matplotlib
import numpy as np
import openpyxl
import pandas as pd
import sklearn 

from matplotlib import pyplot as plt
from numpy import nan
from pandas import read_excel
from sklearn.linear_model import LinearRegression


data_df_raw = pd.read_excel('/Users/sarahaisagbon/Downloads/Question_DataSet.xlsx')

#There are a few missing values. I've observed that the dates are the last day of the month. Assuming that the dates missing are in the right place on the table. 
data_df_raw.loc[9, 'Date'] = '2008-01-31'
data_df_raw.loc[130,'Date'] = '2018-02-28'
data_df_raw.loc[131,'Date'] = '2018-03-31'
#some of the entries in the date column are incorrect, but based on the previous assumption I can fix it.
data_df_raw.loc[172,'Date'] = '2021-08-31'
data_df_raw.loc[181,'Date'] = '2022-05-31'

#convert date strings to datetime dtype
data_df_raw['Date']= pd.to_datetime(data_df_raw['Date'])
#set as index
data_df_raw.set_index('Date', inplace=True)

#I was initially going to delete the rows with missing values but instead I will get the median value for 
#replace nan with mean

data_df_raw.iloc[9,0] = data_df_raw.iloc[8:11,0].median()


data_df_raw.iloc[130,0] = data_df_raw.iloc[129:132,0].median()
data_df_raw.iloc[131,0] = data_df_raw.iloc[130:133,0].median()


#check for 0 in values
df = data_df_raw[data_df_raw.iloc[:,0] == 0]
print(df)

#replace 0 with median
data_df_raw.iloc[41,0] = data_df_raw.iloc[40:43,0].median()

#check if the there are any outliers 
df1 = data_df_raw[data_df_raw.iloc[:,0] >= 42]
print(df1)

#I see that 30/04/2019 is 10x bigger than the other values in 2019 so I'll assume the decimal point has been put in the wrong place.
data_df_raw.iloc[144,0] = '19.0475'

#plot the data
plt.figure(figsize=(15,5))
plt.plot(data_df_raw['US3387 '])
plt.title('Index Price', fontsize=15)
plt.ylabel('Price in dollars.')
plt.xlabel('Date')
plt.show()


test_data = data_df_raw.iloc[-94:]
train_data = data_df_raw.iloc[:-94]
y_train = train_data['US3387 ']
y_test = test_data['US3387 ']
np.reshape(y_train, (94,-1))
np.reshape(x_test, (94,-1))
np.reshape(y_test, (94,-1))

lr = LinearRegression()
lr.fit(y_train)

y_pred = lr.predict(x_test)
test_data['y_pred'] = y_pred
print(test_data)

#Question 2 (Derivative Pricing)
# S = stock underlying # K = strike price # Price = premium paid for option
def long_call(S, K, Price):
    # Long Call Payoff = max(Stock Price - Strike Price, 0)     # If we are long a call, we would only elect to call if the current stock price is greater than      # the strike price on our option     
    P = list(map(lambda x: max(x - K, 0) - Price, S))
    return P

def long_put(S, K, Price):
    # Long Put Payoff = max(Strike Price - Stock Price, 0)     # If we are long a call, we would only elect to call if the current stock price is less than      # the strike price on our option     
    P = list(map(lambda x: max(K - x,0) - Price, S))
    return P
   
def short_call(S, K, Price):
    # Payoff a shortcall is just the inverse of the payoff of a long call     
    P = long_call(S, K, Price)
    return [-1.0*p for p in P]

def bear_spread(S, E1, E2, Price1, Price2):
    
    P_1 = long_call(S, E1, Price1)
    P_2 = short_call(S, E2, Price2)
    return [x+y for x,y in zip(P_1, P_2)] 


S = [t for t in range(0,70)] # Define some series of stock-prices 
P1 = bear_spread(S, 35, 25, 10, 20)
long_c = long_call(S, 35, 10)
short_c = short_call(S, 25, 20)
    
plt.plot(S, P1, 'orange')
plt.plot(S, long_c, 'r--')
plt.plot(S, short_c, 'b--')

plt.legend(["Payoff", "Long Call", "Short Call"])
plt.title('US3387 Payoff Functions', fontsize=20, fontweight='bold')
plt.xlabel('Settle Price', ha='center', fontsize=14, fontweight='bold')
plt.ylabel('Payoff', va='center', rotation='vertical', fontsize=14, fontweight='bold')

plt.show()