# grid_all_strategies.ps1
# Poženi 10_run_al.py za več kombinacij in vse 4 strategije

$BaseDir   = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir    = "$BaseDir\src"
$Data      = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables    = "$BaseDir\tables"

# ustvari tables direktorij če še ne obstaja
if (!(Test-Path -Path $Tables)) {
    New-Item -ItemType Directory -Path $Tables | Out-Null
}

# kombinacije parametrov
$paramCombos = @(
    @{ Init=100; Batch=50;  Iters=20; N_Estimators=200; MaxDepth=20; MinSplit=4; MinLeaf=2 },
    @{ Init=200; Batch=50;  Iters=30; N_Estimators=600; MaxDepth=30; MinSplit=4; MinLeaf=2 },
    @{ Init=150; Batch=100; Iters=20; N_Estimators=800; MaxDepth=40; MinSplit=2; MinLeaf=1 }
)

# vse 4 strategije
$strategies = @("uncertainty","entropy","margin","random")

$seed = 42

foreach ($c in $paramCombos) {
    foreach ($s in $strategies) {
        Write-Host "=== Začenjam test: strategy=$s, init=$($c.Init), batch=$($c.Batch), iters=$($c.Iters) ==="

        py -3.13 "$SrcDir\experiments\run_online_active_learning_with_simulator.py" `
            --data $Data `
            --strategy $s `
            --init $($c.Init) `
            --batch $($c.Batch) `
            --iters $($c.Iters) `
            --test-size 0.1 `
            --n_estimators $($c.N_Estimators) `
            --max_depth $($c.MaxDepth) `
            --min_samples_split $($c.MinSplit) `
            --min_samples_leaf $($c.MinLeaf) `
            --class_weight balanced_subsample `
            --n_jobs -1 `
            --seed $seed `
            --tables-dir $Tables
    }
}

Write-Host "=== Vse kombinacije in strategije zaključene. Rezultati so v $Tables ==="
