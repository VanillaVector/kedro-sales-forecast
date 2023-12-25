import pandas as pd


def preprocess_stores(stores: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for stores.

    Args:
        stores: Raw data.
    Returns:
        Preprocessed data, with `Type` firt transformed to two binary columns
        `Type_A` and `Type_B`, and droped afterwards.
    """
    stores['Type_A'] = stores['Type'].apply(lambda x: 0 if x=='A' else 1)
    stores['Type_B'] = stores['Type_A'].apply(lambda x: 0 if x==1 else 1)
    stores = stores.drop(['Type'], axis=1)
    return stores

def preprocess_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for shuttles.

    Args:
        sales: Raw data.
    Returns:
        Preprocessed data, renaming `Store` to `unique_id`, `Date` to `ds`,
        `Weekly_Sales` to `y` as required by mlforecast.MLForecast.
    """
    sales = sales.rename(columns={'Store':'unique_id', 'Date':'ds', 'Weekly_Sales':'y'})
    return sales

def create_store_sales_weekly(
    sales: pd.DataFrame, stores: pd.DataFrame, calendar: pd.DataFrame
) -> pd.DataFrame:
    """Combines all data to create a model input table.

    Args:
        sales: Preprocessed data for Sales.
        stores: Preprocessed data for Stores.
        calendar: Raw data for calendar.
    Returns:
        Model input table.

    """
    store_sales_weekly = sales.merge(stores, left_on="unique_id", right_on="Store")
    store_sales_weekly = store_sales_weekly.merge(calendar, left_on="ds", right_on="Date")
    
    store_sales_weekly["ds"] = pd.to_datetime(store_sales_weekly["ds"])
    store_sales_weekly = store_sales_weekly.drop(["Store", "Date"], axis=1)
    
    return store_sales_weekly
