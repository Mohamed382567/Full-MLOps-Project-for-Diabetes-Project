param(
    [switch]$Force
)

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot

$envName = "mlops311"
$envFile = Join-Path $projectRoot "environment.yml"
$reqFile = Join-Path $projectRoot "requirements.txt"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "Conda not found. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) or Anaconda and re-run this script." -ForegroundColor Yellow
    exit 1
}

# Remove existing env if --Force set
$envs = & conda env list 2>$null
if ($envs -match $envName) {
    if ($Force) {
        Write-Host "Removing existing environment $envName..."
        conda env remove -n $envName -y
    } else {
        Write-Host "Environment '$envName' already exists. Re-run with -Force to recreate." -ForegroundColor Yellow
        exit 0
    }
}

Write-Host "Creating conda environment '$envName' from environment.yml..."
conda env create -f $envFile -n $envName

Write-Host "Upgrading pip inside the new environment..."
conda run -n $envName python -m pip install --upgrade pip

if (Test-Path $reqFile) {
    Write-Host "Installing pip requirements from requirements.txt inside the environment..."
    conda run -n $envName python -m pip install --no-cache-dir -r $reqFile
} else {
    Write-Host "No requirements.txt found at $reqFile — skipping pip install step." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. To use the environment interactively run:" -ForegroundColor Green
Write-Host "  conda activate $envName"
Write-Host "Or in VS Code: Ctrl+Shift+P -> Python: Select Interpreter -> choose 'Python 3.11 (mlops311)'."

Pop-Location
