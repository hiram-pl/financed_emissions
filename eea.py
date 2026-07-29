import requests, pandas as pd
from urllib.parse import quote

def disco(sql, p=1, n=1000):
    """Takes a SQL string, page, and rows per page. Calls an API and returns a DataFrame with the results of the SQL query."""
    url = f"https://discodata.eea.europa.eu/sql?query={quote(sql)}&p={p}&nrOfHits={n}"  #Builds the request url. Everything after ? is the query string. SQL contains spaces, quote prevents it from breaking.
    r = requests.get(url, timeout=120) #fires the get, gives up after two mins
    r.raise_for_status() # Raises an exception on an HTTP-level failure
    j = r.json() #parses response body into python dicts and lists
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return pd.DataFrame(j["results"])


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Function for comparing different approaches
def score_rf_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


from xgboost import XGBoostRegressor
from sklearn.metrics import mean_absolute_error
# Function for comparing different approaches
def score_xgb_dataset(X_train, X_valid, y_train, y_valid):
    model = XGBoostRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)