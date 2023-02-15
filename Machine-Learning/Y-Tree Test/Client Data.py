import pandas as pd
import math

df = pd.read_excel('Data Sheet for Test (2).xlsx')

#Question 1
ISA_TV = round(df.groupby('Account type')['Total Value (Client base currency)'].sum()['ISA/ Stocks and Shares'], 2)

#Question 2
SIPP_TV = round(df.groupby('Account type')['Total Value (Client base currency)'].sum()['Pension/ SIPP'], 2)

#Question 3
Public = df['Asset Class'].value_counts()['Public Equities']
PubEq = round(Public/89 * 100, 1)

#Question 4
max_x = df.loc[df['JPY'].idxmax()]
exposure = max_x['Asset Name']

#Question 5
ill_df = df['Asset Name'][df['Liquidity'] == 'Illiquid']

#Question 7
df1 = df[df['Asset Base Currency'] == 'USD']
rat = df1['Total Value (Client base currency)']/df1['Total Value (asset base currency)']
USDGBP = round(rat.mean(), 2)

#Question 8
rl = df['Y Risk Level'].mean()
print(rl)
