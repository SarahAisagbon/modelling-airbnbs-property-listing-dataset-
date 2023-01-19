import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import datetime as dt


car_df_raw = pd.read_csv('/Users/sarahaisagbon/Car_Insurance.csv')

#the name of the columns and the corresponding dtype
#car_df_raw.info()

#Check how many missing values each column has
missing_value = car_df_raw.isna().sum()

#Drop rows with missing values
df=car_df_raw.dropna(axis=0)
# Reset index after drop
df=car_df_raw.dropna().reset_index(drop=True)

#Get the mean, the median, and the quartiles of columns with numerical values
df2 = df.describe()


#list of the different job categories
unique_jobs = df["Job"].unique()
#how many unique jobs?
no_unique_jobs = df["Job"].nunique()

#most frequent job category
most_freq = df['Job'].value_counts().idxmax()

#new column with the duration of each call in seconds
df['CallEnd'] = pd.to_datetime(df['CallEnd'])
df['CallStart'] = pd.to_datetime(df['CallStart'])
df['Call Duration'] = df['CallEnd'] - df['CallStart']
df['Call Duration'] = df['Call Duration'].dt.total_seconds()

#What is the average duration of each call?
average_dur = df['Call Duration'].mean()

df.info()

#Filter the dateframe to people contacted in the first half of the year.
def mon_convert(mon):
    return dt.datetime.strptime(mon, '%b').month
#create a copy 
df_copy = df.copy()
df_copy['LastContactMonth'] = df_copy['LastContactMonth'].apply(mon_convert)
#filter down to only the first half of the year
JanJun_df = df_copy[df_copy['LastContactMonth'] < 7]
#reset the index
JanJun_df = JanJun_df.reset_index(drop=True)

#Find the most common form of communication amongst those contacted in the first half of the year
most_common_com = JanJun_df['Communication'].value_counts().idxmax()

