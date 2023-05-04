# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import sys
import time
import json
from collections import OrderedDict
import modin.pandas as pd


def read(filename):
    column_types = {
        "trip_id": "int64",
        "vendor_id": "string",
        "pickup_datetime": "timestamp",
        "dropoff_datetime": "timestamp",
        "store_and_fwd_flag": "string",
        "rate_code_id": "int64",
        "pickup_longitude": "float64",
        "pickup_latitude": "float64",
        "dropoff_longitude": "float64",
        "dropoff_latitude": "float64",
        "passenger_count": "int64",
        "trip_distance": "float64",
        "fare_amount": "float64",
        "extra": "float64",
        "mta_tax": "float64",
        "tip_amount": "float64",
        "tolls_amount": "float64",
        "ehail_fee": "float64",
        "improvement_surcharge": "float64",
        "total_amount": "float64",
        "payment_type": "string",
        "trip_type": "float64",
        "pickup": "string",
        "dropoff": "string",
        "cab_type": "string",
        "precipitation": "float64",
        "snow_depth": "int64",
        "snowfall": "float64",
        "max_temperature": "int64",
        "min_temperature": "int64",
        "average_wind_speed": "float64",
        "pickup_nyct2010_gid": "float64",
        "pickup_ctlabel": "float64",
        "pickup_borocode": "float64",
        "pickup_boroname": "string",
        "pickup_ct2010": "float64",
        "pickup_boroct2010": "float64",
        "pickup_cdeligibil": "string",
        "pickup_ntacode": "string",
        "pickup_ntaname": "string",
        "pickup_puma": "float64",
        "dropoff_nyct2010_gid": "float64",
        "dropoff_ctlabel": "float64",
        "dropoff_borocode": "float64",
        "dropoff_boroname": "string",
        "dropoff_ct2010": "float64",
        "dropoff_boroct2010": "float64",
        "dropoff_cdeligibil": "string",
        "dropoff_ntacode": "string",
        "dropoff_ntaname": "string",
        "dropoff_puma": "float64",
    }

    all_but_dates = {
        col: valtype
        for (col, valtype) in column_types.items()
        if valtype not in ["timestamp"]
    }
    dates_only = [
        col for (col, valtype) in column_types.items() if valtype in ["timestamp"]
    ]

    df = pd.read_csv(
        filename,
        header=0,
        dtype=all_but_dates,
        parse_dates=dates_only,
    )

    df.shape  # to trigger real execution on omnisci
    return df


def q1_omnisci(df):
    q1_pandas_output = df.groupby("cab_type").size()
    q1_pandas_output.shape  # to trigger real execution on omnisci
    return q1_pandas_output


def q2_omnisci(df):
    q2_pandas_output = df.groupby("passenger_count").agg({"total_amount": "mean"})
    q2_pandas_output.shape  # to trigger real execution on omnisci
    return q2_pandas_output


def q3_omnisci(df):
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    q3_pandas_output = df.groupby(["passenger_count", "pickup_datetime"]).size()
    q3_pandas_output.shape  # to trigger real execution on omnisci
    return q3_pandas_output


def q4_omnisci(df):
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    df["trip_distance"] = df["trip_distance"].astype("int64")
    q4_pandas_output = (
        df.groupby(["passenger_count", "pickup_datetime", "trip_distance"], sort=False)
        .size()
        .reset_index()
        .sort_values(
            by=["pickup_datetime", 0], ignore_index=True, ascending=[True, False]
        )
    )
    q4_pandas_output.shape  # to trigger real execution on omnisci
    return q4_pandas_output


def hdk_warmap_query():
    # Trigger HDK initialization by executing a quick trivial
    # query. It is necessary for correct time measurement of ETL part.
    df = pd.DataFrame({"a": [1, 2, 3]})
    df = df + 1
    df.shape


def measure(func, *args, **kw):
    t0 = time.time()
    res = func(*args, **kw)
    t1 = time.time()
    return res, t1 - t0


def run(input_file):
    hdk_warmap_query()

    res = OrderedDict()
    df, res["Reading"] = measure(read, input_file)
    _, res["Q1"] = measure(q1_omnisci, df)
    _, res["Q2"] = measure(q2_omnisci, df)
    _, res["Q3"] = measure(q3_omnisci, df.copy())
    _, res["Q4"] = measure(q4_omnisci, df.copy())
    return res


def main():
    if len(sys.argv) != 2:
        print(
            f"USAGE: python taxi.py <data file name>"
        )
        return
    result = run(sys.argv[1])
    json.dump(result, sys.stdout, indent=4)


if __name__ == "__main__":
    main()
