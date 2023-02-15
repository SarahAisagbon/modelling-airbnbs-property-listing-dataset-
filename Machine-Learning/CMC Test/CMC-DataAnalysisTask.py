import pandas as pd
import csv
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

plot1 = df.plot(x="Time", y=["BANK1.BID", "BANK1.ASK"])
plot2 = df.plot(x="Time", y=["BANK2.BID", "BANK2.ASK"])
plot3 = df.plot(x="Time", y=["BANK3.BID", "BANK3.ASK"])
plot4 = df.plot(x="Time", y=["BANK4.BID", "BANK4.ASK"])
plot5 = df.plot(x="Time", y=["BANK5.BID", "BANK5.ASK"])
plt.show()

BANK1 = df[["BANK1.BID", "BANK1.ASK"]]
BANK2 = df[["BANK2.BID", "BANK2.ASK"]]
BANK3 = df[["BANK3.BID", "BANK3.ASK"]]
BANK4 = df[["BANK4.BID", "BANK4.ASK"]]
BANK5 = df[["BANK5.BID", "BANK5.ASK"]]

BANK1_spread = BANK1.iloc[:, 1] - BANK1.iloc[:, 0]
BANK2_spread = BANK2.iloc[:, 1] - BANK2.iloc[:, 0]
BANK3_spread = BANK3.iloc[:, 1] - BANK3.iloc[:, 0]
BANK4_spread = BANK4.iloc[:, 1] - BANK4.iloc[:, 0]
BANK5_spread = BANK5.iloc[:, 1] - BANK5.iloc[:, 0]

mean_values = [BANK1_spread.mean(), BANK2_spread.mean(), BANK3_spread.mean(), BANK4_spread.mean(), BANK5_spread.mean()]
var_values = [BANK1_spread.var(), BANK2_spread.var(), BANK3_spread.var(), BANK4_spread.var(), BANK5_spread.var()]
mid_values = [BANK1.median(), BANK2.median(), BANK3.median(), BANK4.median(), BANK5.median()]
banks = ["BANK1", "BANK2", "BANK3", "BANK4", "BANK5"]

#tightest overall spread
tightest_spread = min(mean_values)
tightest_spread_bank = banks[mean_values.index(tightest_spread)]
print(tightest_spread_bank)

#most stable spread 
stable_spread = min(var_values)
stable_spread_bank = banks[var_values.index(stable_spread)]
print(stable_spread_bank)

#most reliable mid-price
#I thought about what would make a reliable mid-point and I think a reliable mid-point would be a mid-price that is within the bid-ask range for most of the period.
mid_prices = [midprice.mean() for midprice in mid_values]

yes_1 = yes_2 = yes_3 = yes_4 = yes_5 = 0
for i in range(len(df)):
    if BANK1.iloc[i, 0] < mid_prices[0] and BANK1.iloc[i, 1] > mid_prices[0]:
        yes_1 += 1
    if BANK2.iloc[i, 0] < mid_prices[1] and BANK2.iloc[i, 1] > mid_prices[1]:
        yes_2 += 1
    if BANK3.iloc[i, 0] < mid_prices[2] and BANK3.iloc[i, 1] > mid_prices[2]:
        yes_3 += 1
    if BANK4.iloc[i, 0] < mid_prices[3] and BANK4.iloc[i, 1] > mid_prices[3]:
        yes_4 += 1
    if BANK5.iloc[i, 0] < mid_prices[4] and BANK5.iloc[i, 1] > mid_prices[4]:
        yes_5 += 1

mid_point = [yes_1, yes_2, yes_3, yes_4, yes_5]
most_reliable = banks[mid_point.index(max(mid_point))]
print(most_reliable)
