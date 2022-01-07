This is a data science processing workload for a proposed SPEC Workstation
benchmark. It required modin, so the easiest way to install it is to create
modin conda environment, like this:

```
conda create -n modin -c conda-forge modin-all scikit-learn-intelex xgboost
conda activate modin
```

To run it use launcher script. Currently, three benchmarks are included into
workload: taxi, census and plasticc. Use `python launcher.py -h` for command
line switches help.

Examples:
---------
```
python launcher.py -m taxi
```
runs NY Taxi benchmark with default number of records.
```
python launcher.py -m taxi -tr 1000000
```
runs NT Taxi benchmark with 1M of records
```
python launcher.py -m census
```
runs Census benchmark with default number of records.
```
python launcher.py -m census -cr 10000
```
runs Census benchmark with 10K of records
