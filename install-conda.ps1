$CONDA_DIR="$Env:LOCALAPPDATA\miniconda"
$CONDA_ENV_NAME="data-science-processing-workload"

Invoke-WebRequest -Uri https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -OutFile $env:TEMP\Miniconda3-latest-Windows-x86_64.exe

$process = Start-Process -FilePath "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe" `
  -ArgumentList ("/S", "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=1", "/D=$CONDA_DIR") `
  -NoNewWindow -PassThru -Wait

& "$CONDA_DIR\Scripts\conda.exe" create -n $CONDA_ENV_NAME --yes -c conda-forge modin-all ray-dashboard scikit-learn-intelex xgboost

"Execute the following command in your terminal to be able to run launcher script:"
""
"conda activate ${CONDA_ENV_NAME}"
