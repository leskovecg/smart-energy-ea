# quick_online_test.ps1
# Hiter test za online Active Learning (uncertainty, init=100, batch=50, iters=20)

$BaseDir   = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir    = "$BaseDir\src"
$Data      = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables    = "$BaseDir\tables"

# ustvari tables direktorij če še ne obstaja
if (!(Test-Path -Path $Tables)) {
    New-Item -ItemType Directory -Path $Tables | Out-Null
}

Write-Host "=== Online AL quick test (uncertainty) ==="
py -3.13 "$SrcDir\experiments\run_online_active_learning_with_simulator.py" `
    --data $Data `
    --strategy uncertainty `
    --init 100 `
    --batch 50 `
    --iters 20 `
    --test-size 0.1 `
    --n_estimators 100 `
    --max_depth 20 `
    --min_samples_split 4 `
    --min_samples_leaf 2 `
    --class_weight balanced_subsample `
    --n_jobs -1 `
    --seed 42 `
    --tables-dir $Tables

Write-Host "=== Končano! Rezultati so v $Tables ==="
