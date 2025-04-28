# src/data_preprocessing.py

import pandas as pd # type: ignore

def load_data(petrol_price_path: str, sales_data_path: str):
    """Load petrol prices and product sales data."""
    petrol_df = pd.read_csv("data/petrol_price.csv")
    sales_df = pd.read_csv("data/product_sales.csv")
    return petrol_df, sales_df


def preprocess_data(petrol_df: pd.DataFrame, sales_df: pd.DataFrame) -> pd.DataFrame:
    petrol_df = petrol_df.copy()
    petrol_df.dropna(inplace=True)
    petrol_df.reset_index(drop=True, inplace=True)

    sales_df = sales_df.copy()
    sales_df.dropna(inplace=True)
    sales_df.reset_index(drop=True, inplace=True)

    final_df = pd.concat([petrol_df, sales_df], axis=1)

    return final_df

    # Feature Engineering example: we can later add moving averages, trends etc
    sales_df['Profit_Margin_Percent'] = (sales_df['Profit_Per_Unit'] / sales_df['Selling_Price']) * 100
    
    # Drop unnecessary columns if any
    features = sales_df[['Stock_Available', 'Selling_Price', 'Purchase_Price', 
                         'Profit_Per_Unit', 'Profit_Margin_Percent', 'Total_Sales_Amount', 'Total_Profit']]
    
    target = sales_df['Units_Sold']
    
    return features, target
