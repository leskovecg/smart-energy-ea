$BaseDir = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir  = "$BaseDir\src"
$Data    = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables  = "$BaseDir\tables"

# <<< POMEMBNO: pot do python.exe >>>
# Če imaš venv:
# $Python = "$BaseDir\.venv\Scripts\python.exe"
# Če nimaš venv ali želiš sistemskega:
$Python = (& py -3.13 -c "import sys; print(sys.executable)").Trim()
Write-Host "Using Python: $Python"

$Seed = 42
$MaxParallel = 4
$Strategies = @("uncertainty","entropy","margin","random")

$ParamCombos = @(
    @{ Tag="A"; Init=300; Batch=100; Iters=20; N_Estimators=1000; MaxDepth=$null; MinSplit=2; MinLeaf=1 },
    @{ Tag="B"; Init=200; Batch=75;  Iters=24; N_Estimators=800;  MaxDepth=25;   MinSplit=2; MinLeaf=1 },
    @{ Tag="C"; Init=150; Batch=150; Iters=12; N_Estimators=1200; MaxDepth=30;   MinSplit=2; MinLeaf=1 },
    @{ Tag="D"; Init=250; Batch=50;  Iters=40; N_Estimators=600;  MaxDepth=20;   MinSplit=2; MinLeaf=2 }
)

if (!(Test-Path $Tables)) { New-Item -ItemType Directory -Path $Tables | Out-Null }

function Build-Args {
    param([string]$Strategy, [hashtable]$C)
    $args = @(
        "$SrcDir\experiments\run_online_active_learning_with_simulator.py",     # PRVI argument: pot do skripte
        "--data", "$Data",
        "--strategy", "$Strategy",
        "--init", "$($C.Init)",
        "--batch", "$($C.Batch)",
        "--iters", "$($C.Iters)",
        "--test-size", "0.1",
        "--n_estimators", "$($C.N_Estimators)",
        "--min_samples_split", "$($C.MinSplit)",
        "--min_samples_leaf", "$($C.MinLeaf)",
        "--class_weight", "balanced_subsample",
        "--n_jobs", "-1",
        "--seed", "$Seed",
        "--tables-dir", "$Tables"
    )
    if ($null -ne $C.MaxDepth) { $args += @("--max_depth", "$($C.MaxDepth)") }
    return $args
}

$jobs = @()
foreach ($C in $ParamCombos) {
    foreach ($S in $Strategies) {
        while ((Get-Job -State Running).Count -ge $MaxParallel) { Start-Sleep -Seconds 2 }

        $name = ("AL_{0}_Set{1}_init{2}_batch{3}_iters{4}_nest{5}_depth{6}_leaf{7}" -f
            $S, $C.Tag, $C.Init, $C.Batch, $C.Iters, $C.N_Estimators,
            ($C.MaxDepth -as [string]), $C.MinLeaf)

        Write-Host ">> Start $name"
        $args = Build-Args -Strategy $S -C $C

        $job = Start-Job -Name $name -ScriptBlock {
            param($PythonExe, $ArgsArray, $WorkDir)
            Set-Location $WorkDir
            & $PythonExe @ArgsArray
        } -ArgumentList @($Python, $args, $BaseDir)

        $jobs += $job
    }
}

Write-Host ">> Zagnanih jobov: $($jobs.Count). Čakam na zaključek..."
Wait-Job -Job $jobs | Out-Null

# Logi
$log = Join-Path $Tables ("logs_new_combos_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
foreach ($j in $jobs) {
    "### " + $j.Name | Out-File -FilePath $log -Append -Encoding UTF8
    (Receive-Job -Job $j -Keep) | Out-File -FilePath $log -Append -Encoding UTF8
    "`n" | Out-File -FilePath $log -Append -Encoding UTF8
}
Write-Host "=== Končano. CSV-ji v $Tables, log: $log ==="
