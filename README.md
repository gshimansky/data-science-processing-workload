This is a data science processing workload for a proposed SPEC Workstation
benchmark. It required modin, so the easiest way to install it is to create
modin conda environment, like this:

```
conda create -n modin -c conda-forge modin-all
conda activate modin
```

To run it use launcher script. Currently, three benchmarks are included into
workload: taxi, census and plasticc. Use `python launcher.py -h` for command
line switches help.
