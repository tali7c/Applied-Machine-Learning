"""Dataset loaders for the Applied Machine Learning lab assignments.

Each loader makes the canonical call named in Anchor-Datasets.pdf first. If that
download is blocked — on the campus network it fails with
CERTIFICATE_VERIFY_FAILED — it falls back to a CSV in ./data/.

    from aml_data import load_california, load_telco, load_stock
"""
import os

import pandas as pd

DATA = os.environ.get("AML_DATA", "data")


def load_california():
    """A1 — California Housing, 20,640 x 9, target MedHouseVal in $100k units."""
    try:
        from sklearn.datasets import fetch_california_housing
        return fetch_california_housing(as_frame=True).frame
    except Exception as exc:                      # campus filter, or no network
        path = os.path.join(DATA, "housing.csv")
        if not os.path.exists(path):
            raise RuntimeError(
                "fetch_california_housing() failed (%s) and %s is missing. Run the "
                "pre-fetch at home, or download the StatLib CSV into data/." % (exc, path))
        raw = pd.read_csv(path)
        raw["total_bedrooms"] = raw.total_bedrooms.fillna(raw.total_bedrooms.median())
        return pd.DataFrame({
            "MedInc": raw.median_income,
            "HouseAge": raw.housing_median_age,
            "AveRooms": raw.total_rooms / raw.households,
            "AveBedrms": raw.total_bedrooms / raw.households,
            "Population": raw.population,
            "AveOccup": raw.population / raw.households,
            "Latitude": raw.latitude,
            "Longitude": raw.longitude,
            "MedHouseVal": raw.median_house_value / 100000.0,
        })


def load_telco():
    """A2 — Telco Customer Churn, 7,043 x 21. Kaggle: blastchar/telco-customer-churn."""
    for name in ("WA_Fn-UseC_-Telco-Customer-Churn.csv", "telco.csv", "Telco-Customer-Churn.csv"):
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            return pd.read_csv(path)
    raise RuntimeError("Put the Telco CSV in %s/ — see Anchor-Datasets.pdf." % DATA)


def load_stock(ticker="IBM", start="2013-01-01"):
    """C1 — one ticker of daily history, Date + Close, sorted ascending."""
    single = os.path.join(DATA, "%s.csv" % ticker.lower())
    if os.path.exists(single):
        s = pd.read_csv(single, parse_dates=["Date"])
        s = s[["Date", "Close"]]
    else:
        path = os.path.join(DATA, "stockdata.csv")
        if not os.path.exists(path):
            raise RuntimeError(
                "No %s and no %s. Download a daily history with yfinance or from "
                "Kaggle into data/." % (single, path))
        wide = pd.read_csv(path, parse_dates=["Date"])
        s = wide[["Date", ticker]].rename(columns={ticker: "Close"})
    s = s[s.Date >= start].sort_values("Date").reset_index(drop=True)
    return s


def load_mnist():
    """B2 — MNIST, 70,000 x 784 and the labels as integers.

    Tries fetch_openml first, as Anchor-Datasets.pdf specifies. Falls back to
    IDX files in ./data/ when the download is blocked.
    """
    try:
        from sklearn.datasets import fetch_openml
        m = fetch_openml("mnist_784", version=1, as_frame=False)
        import numpy as np
        return m.data.astype("uint8"), m.target.astype(int)
    except Exception as exc:
        import gzip
        import numpy as np
        cache = os.path.join(DATA, "mnist.npz")
        if os.path.exists(cache):
            with np.load(cache) as z:
                return z["X"], z["y"]

        def _imgs(p):
            with gzip.open(p, "rb") as f:
                magic, n, r, c = np.frombuffer(f.read(16), ">u4")
                if magic != 2051:
                    raise ValueError("%s is not an IDX image file" % p)
                return np.frombuffer(f.read(), "u1").reshape(n, r * c)

        def _labs(p):
            with gzip.open(p, "rb") as f:
                magic, n = np.frombuffer(f.read(8), ">u4")
                if magic != 2049:
                    raise ValueError("%s is not an IDX label file" % p)
                return np.frombuffer(f.read(), "u1")

        need = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        paths = [os.path.join(DATA, n) for n in need]
        if not all(os.path.exists(p) for p in paths):
            raise RuntimeError(
                "fetch_openml('mnist_784') failed (%s) and no IDX files in %s/. "
                "Run the pre-fetch at home, or place the four IDX .gz files in data/."
                % (exc, DATA))
        X = np.vstack([_imgs(paths[0]), _imgs(paths[2])])
        y = np.concatenate([_labs(paths[1]), _labs(paths[3])]).astype(int)
        np.savez_compressed(cache, X=X, y=y)
        return X, y
