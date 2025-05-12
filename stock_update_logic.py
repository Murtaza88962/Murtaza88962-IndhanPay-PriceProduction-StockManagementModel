# src/stock_update_logic.py

import pandas as pd # type: ignore

def predict_future_sales(model, features: pd.DataFrame):
    """Predict future units sold using trained model."""
    predictions = model.predict(features)
    return predictions

def update_stock(sales_df: pd.DataFrame, predicted_units_sold):
    """Update stock availability based on predictions."""
    sales_df['Predicted_Units_Sold'] = predicted_units_sold
    sales_df['Updated_Stock'] = sales_df['Stock_Available'] - sales_df['Predicted_Units_Sold']
    
    # Ensure stock doesn't go negative
    sales_df['Updated_Stock'] = sales_df['Updated_Stock'].apply(lambda x: max(x, 0))
    
    return sales_df
