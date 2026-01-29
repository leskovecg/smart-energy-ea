# hitri_test.ps1
# PowerShell skripta za hitro testiranje online (AL) in offline baseline

# poti
$BaseDir   = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir    = "$BaseDir\src"
$Data      = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables    = "$BaseDir\tables"

# ustvari tables direktorij če ne obstaja
if (!(Test-Path -Path $Tables)) {
    New-Item -ItemType Directory -Path $Tables | Out-Null
}

Write-Host "=== Začenjam ONLINE Active Learning test ==="
py -3.13 "$SrcDir\experiments\run_online_active_learning_with_simulator.py" `
    --data $Data `
    --strategy entropy `
    --init 100 `
    --batch 50 `
    --iters 5 `
    --test-size 0.1 `
    --n_estimators 200 `
    --max_depth 20 `
    --min_samples_split 4 `
    --min_samples_leaf 2 `
    --class_weight balanced_subsample `
    --n_jobs -1 `
    --seed 42 `
    --tables-dir $Tables

Write-Host "=== Začenjam OFFLINE baseline test ==="
py -3.13 "$SrcDir\experiments\run_offline_random_forest_baseline.py" `
    --data $Data `
    --test-size 0.1 `
    --n_estimators 200 `
    --max_depth 20 `
    --min_samples_split 4 `
    --min_samples_leaf 2 `
    --class_weight balanced_subsample `
    --n_jobs -1 `
    --seed 42 `
    --tables-dir $Tables

Write-Host "=== Testiranje končano. Rezultati so v $Tables ==="
