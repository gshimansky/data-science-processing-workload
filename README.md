Installation on Linux
---------------------
Run script `install-conda.sh`. It will install miniconda and create an environment
named `data-science-processing-workload`. This environment contains all required packages.

Installation on Windows
-----------------------
First make it possible to run scripts under the current user with the following
PowerShell command:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Unrestricted
```
After that it should be possible to execute PowerShell script `install-conda.ps1`. When running
in command prompt you can execute it like this:
```
powershell .\install-conda.ps1
```
It will install miniconda and create an environment named `data-science-processing-workload`.
This environment contains all required packages. This script also registers Python
that miniconda install as the default Python interpreter in the system. If this is not
desired, please remove switch `/RegisterPython=0` and `/AddToPath=1` from miniconda
command line in `install-conda.ps1`.

If you want to run benchmarks on Linux installed in Windows Subsystem for Linux (WSL)
on Windows, make sure that you use WSL generation 2 (WSL2). In this case Linux runs
as a separate VM that has its own IP address and Ray running in WSL2 doesn't fail to
connect to its workers.

Description
-----------

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

Examples
--------
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
