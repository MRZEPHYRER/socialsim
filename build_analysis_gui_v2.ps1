param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$SpecFile = Join-Path $ProjectRoot "SOCIALSIM_Analysis_V2.spec"
$DistPath = Join-Path $ProjectRoot "dist"
$WorkPath = Join-Path $ProjectRoot "build/analysis_gui_v2"

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller --version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is not installed in the selected Python environment."
    }

    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath $DistPath `
        --workpath $WorkPath `
        $SpecFile

    if ($LASTEXITCODE -ne 0) {
        throw "Analysis GUI V2 executable build failed."
    }

    $Executable = Join-Path $DistPath "SOCIALSIM_Analysis_V2.exe"
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Build finished without producing $Executable"
    }

    Write-Host "Built single-file Analysis GUI V2 executable:"
    Write-Host $Executable
}
finally {
    Pop-Location
}
