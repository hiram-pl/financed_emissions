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