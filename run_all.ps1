# ==============================
# QMR (QL) vs GEO vs RND batch runner
# ==============================

$ErrorActionPreference = "Stop"

# --- settings ---
$ndronesList = @(10, 20, 30, 50)
$algs = @("QL", "GEO", "RND")
$seedStart = 0
$seedEnd = 20
$expMetric = "ndrones_"   # json_and_plot expects this label

Write-Host "=== Creating output directories ==="
New-Item -ItemType Directory -Force -Path ".\data\evaluation_tests" | Out-Null
New-Item -ItemType Directory -Force -Path ".\data\plots" | Out-Null

Write-Host "=== Batch experiments start ==="
foreach ($nd in $ndronesList) {
    foreach ($alg in $algs) {
        Write-Host ""
        Write-Host ">>> Running: nd=$nd alg=$alg seeds=$seedStart..$seedEnd"
        python -m src.experiments.experiment_ndrones -nd $nd -i_s $seedStart -e_s $seedEnd -alg $alg

        Write-Host ">>> Plotting (single alg): nd=$nd alg=$alg"
        python -m src.experiments.json_and_plot -nd $nd -i_s $seedStart -e_s $seedEnd -exp_suffix $alg -exp_metric $expMetric
    }

    Write-Host ""
    Write-Host ">>> Plotting (compare QL/GEO/RND): nd=$nd"
    python -m src.experiments.json_and_plot -nd $nd -i_s $seedStart -e_s $seedEnd -exp_suffix QL GEO RND -exp_metric $expMetric
}

Write-Host ""
Write-Host "=== DONE ==="
Write-Host "Check:"
Write-Host "  - JSON: .\data\evaluation_tests\"
Write-Host "  - PNG:  .\data\plots\"
