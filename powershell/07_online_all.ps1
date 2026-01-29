# 13_grid_online_alt_combos.ps1
# ONLINE Active Learning (simulator): 4 nove kombinacije (E–H) × 4 strategije, paralelno

$BaseDir = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir  = "$BaseDir\src"
$Data    = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables  = "$BaseDir\tables"

# Če imaš venv, nastavi pot do njega:
# $Python = "$BaseDir\.venv\Scripts\python.exe"
$Python = (& py -3.13 -c "import sys; print(sys.executable)").Trim()
Write-Host "Using Python: $Python"

$Seed = 42
$MaxParallel = 4
$Strategies = @("uncertainty","entropy","margin","random")

# NOVE kombinacije (različne od A–D):
$ParamCombos = @(
    @{ Tag="E"; Init=400; Batch=100; Iters=16; N_Estimators=1200; MaxDepth=$null; MinSplit=2; MinLeaf=1 },
    @{ Tag="F"; Init=200; Batch=125; Iters=16; N_Estimators=1000; MaxDepth=35;   MinSplit=2; MinLeaf=1 },
    @{ Tag="G"; Init=300; Batch=75;  Iters=24; N_Estimators=800;  MaxDepth=25;   MinSplit=2; MinLeaf=2 },
    @{ Tag="H"; Init=500; Batch=50;  Iters=20; N_Estimators=1000; MaxDepth=20;   MinSplit=2; MinLeaf=2 }
)

if (!(Test-Path $Tables)) { New-Item -ItemType Directory -Path $Tables | Out-Null }

function Build-Args {
    param([string]$Strategy, [hashtable]$C)
    $args = @(
        "$SrcDir\experiments\run_online_active_learning_with_simulator.py",
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
$total = $ParamCombos.Count * $Strategies.Count
$completed = 0

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

        # ko se job konča, povečaj števec
        Register-ObjectEvent -InputObject $job -EventName StateChanged -Action {
            if ($event.Sender.State -eq 'Completed' -or $event.Sender.State -eq 'Failed') {
                [System.Threading.Interlocked]::Increment([ref]$using:completed) | Out-Null
                Write-Host ("   Progress: {0}/{1} done" -f $using:completed, $using:total)
            }
        } | Out-Null

        $jobs += $job
    }
}

Write-Host (">> Zagnanih jobov: {0}. Čakam na zaključek..." -f $jobs.Count)
Wait-Job -Job $jobs | Out-Null

# Shrani loge
$log = Join-Path $Tables ("logs_alt_combos_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
foreach ($j in $jobs) {
    "### " + $j.Name | Out-File -FilePath $log -Append -Encoding UTF8
    (Receive-Job -Job $j -Keep) | Out-File -FilePath $log -Append -Encoding UTF8
    "`n" | Out-File -FilePath $log -Append -Encoding UTF8
}

Write-Host "=== Končano. CSV-ji v $Tables, log: $log ==="
