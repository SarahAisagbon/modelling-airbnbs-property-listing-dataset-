import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt

athletes_df_raw = pd.read_excel('/Users/sarahaisagbon/Downloads/Olympics/Athletes.xlsx')
coaches_df_raw = pd.read_excel('/Users/sarahaisagbon/Downloads/Olympics/Coaches.xlsx')
gender_df_raw = pd.read_excel('/Users/sarahaisagbon/Downloads/Olympics/EntriesGender.xlsx')
medals_df_raw = pd.read_excel('/Users/sarahaisagbon/Downloads/Olympics/Medals.xlsx')
teams_df_raw = pd.read_excel('/Users/sarahaisagbon/Downloads/Olympics/Teams.xlsx')

athletes_df_raw.info()
coaches_df_raw.info()
gender_df_raw.info()
medals_df_raw.info()
teams_df_raw.info()

#Plot a bar graph showing the number of athletes in each discipline.
gender_df_raw.plot.bar(x='Discipline', y = 'Total')
plt.title("The Number of Athletics in Each Discipline")
plt.xlabel("Discipline")
plt.ylabel("Number of Athletes")

#Create a new bar graph splitting the total number by gender.
gender_df_raw.plot.bar(x='Discipline', stacked = True)
plt.title("The Total Number of Athletics per discipline by gender")
plt.xlabel("Discipline")
plt.ylabel("Number of Athletes")

#Plot a graph showing the number of athletes per country.
athletes_df = athletes_df_raw['NOC'].value_counts()
athletes_df.plot.bar()
plt.xlabel("Country")
plt.ylabel("Number of Athletes")
plt.show()