import requests, pandas as pd
#quote is the url encoder
from urllib.parse import quote

#TBL so I don't have to rewrite every time
TBL = "[CO2Emission].[latest].[co2cars_2025Pv31]"

def disco(sql, p=1, n=1000):
    """Takes a SQL string, page, and rows per page. Calls an API and returns a DataFrame with the results of the SQL query."""
    url = f"https://discodata.eea.europa.eu/sql?query={quote(sql)}&p={p}&nrOfHits={n}"  #Builds the request url. Everything after ? is the query string. SQL contains spaces, quote prevents it from breaking.
    r = requests.get(url, timeout=120) #fires the get, gives up after two mins
    r.raise_for_status() # Raises an exception on an HTTP-level failure
    j = r.json() #parses response body into python dicts and lists
    if "errors" in j:
        raise RuntimeError(j["errors"])
    return pd.DataFrame(j["results"])

disco(f"SELECT TOP 100 * "
f"FROM {TBL}")

#how big is this actually?
#disco(f"SELECT COUNT(*) AS n" 
#f"FROM {TBL}")

#target distribution (name the aggregate, or it errors)
#disco(f"SELECT Ft AS fuel, COUNT(*) AS n, AVG(CAST(Ewltp AS float)) AS mean_co2" 
#f"FROM {TBL}" 
#f"GROUP BY Ft" 
#f"ORDER BY n DESC")

#cardinality of the scary columns
#disco(f"SELECT COUNT(DISTINCT Mk) AS n_makes, COUNT(DISTINCT Cn) AS n_names" 
#f"FROM {TBL}")

#missingness, one column at a time
#disco(f"SELECT COUNT(*) AS n_null" 
#f"FROM {TBL}" 
#f"WHERE Ewltp IS NULL")

#used DuckDB bc embedded OLAP (column store, analytical)
