import pandas as pd
import numpy as np

def remove_rows_with_missing_ratings(df_raw):
    #Drop rows with missing values
    #cols = list(df_raw.filter(like='rating').columns)
    rating_cols = [col for col in df_raw.columns if "rating" in col]
    df=df_raw.dropna(subset=rating_cols)
    # Reset index after drop
    df=df_raw.dropna(subset=rating_cols).reset_index(drop=True)
    return df 

def convert_description(lst_of_str):
    #Remove [About this space and ] 
    lst_of_str = str(lst_of_str)[22: -1]
    #Turn the string into a list
    lst_of_str = lst_of_str.split(", '")
    #Create a list of strings we will remove from the list of strings
    remove_str = ["The space'", "", "Guest access'", "Other things to note'", "'"]

    #remove ' from each string in the list
    for ele in lst_of_str:
        ele = ele[0:-1]

    #remove elements in the remove_str from the list of strings
    for string in remove_str:
        if string in lst_of_str:
            lst_of_str.remove(string)
        else:
            pass
    
    #Turn the list of strings into a string
    lst_of_str = ','.join(lst_of_str)
    
    #Clean up the string and remove excess commas, full stops, spaces and quotation marks
    lst_of_str = lst_of_str.replace("',", " ")
    lst_of_str = lst_of_str.replace('",', " ")
    lst_of_str = lst_of_str.replace(".'", ".")
    lst_of_str = lst_of_str.replace('.   ', ". ")
    return lst_of_str


def combine_description_strings(df_raw):
    #make a copy of the dataframe
    df=df_raw.copy()
    #apply the function convert_description to the Description column
    df['Description'] = df['Description'].apply(convert_description)
    return df

def set_default_feature_values(df_raw):
    #set the na values in the feature columns as 1
    df=df_raw.copy()
    columns = ["guests", "beds", "bathrooms", "bedrooms"]
    df[columns]=df[columns].fillna(1)
    return df
    
def clean_tabular_data(df_raw):
    #calls each function sequentially
    df_raw2 = remove_rows_with_missing_ratings(df_raw)
    df_raw3 = combine_description_strings(df_raw2)
    df = set_default_feature_values(df_raw3)
    return df

def load_airbnb(df_raw, label):
    if label == "Category":
        features = df_raw.select_dtypes(include=np.number)
        label_vector = df_raw["Category"]
    else:
        df_numerical = df_raw.select_dtypes(include=np.number)
        label_vector = df_numerical[label]
        features = df_numerical.drop(label, axis=1)
    return (features, label_vector)

if __name__ == "__main__":
    df_raw = pd.read_csv(r"/Users/sarahaisagbon/Documents/GitHub/Data-Science/airbnb-property-listings/tabular_data/listing.csv", index_col=0)
    clean_df = clean_tabular_data(df_raw)
    #removed unnamed column
    clean_df.to_csv("/Users/sarahaisagbon/Documents/GitHub/Data-Science/airbnb-property-listings/tabular_data/clean_tabular_data.csv")
    label = "Price_Night"
    full_data = load_airbnb(clean_df, label)
    






