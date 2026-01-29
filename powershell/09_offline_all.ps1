# 09_offline_all.ps1 (FIXED)
$BaseDir = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir  = "$BaseDir\src"
$Data    = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables  = "$BaseDir\tables"
$Python = (& py -3.13 -c "import sys; print(sys.executable)").Trim()
Write-Host "Using Python: $Python"

$Seed        = 42
$TestSize    = 0.10
$MaxParallel = 4

$ParamCombos = @(
    @{ Tag="O5"; N_Estimators=1400; MaxDepth=$null; MinSplit=2; MinLeaf=1; ClassW="balanced" },
    @{ Tag="O6"; N_Estimators=1000; MaxDepth=35;    MinSplit=2; MinLeaf=2; ClassW="balanced" },
    @{ Tag="O7"; N_Estimators=900;  MaxDepth=25;    MinSplit=4; MinLeaf=1; ClassW=$null      },
    @{ Tag="O8"; N_Estimators=1200; MaxDepth=30;    MinSplit=2; MinLeaf=3; ClassW="balanced_subsample" }
)

if (!(Test-Path $Tables)) { New-Item -ItemType Directory -Path $Tables | Out-Null }

function Build-Args {
    param([hashtable]$C)
    $args = @(
        "$SrcDir\experiments\run_offline_random_forest_baseline.py",
        "--data", "$Data",
        "--test-size", "$TestSize",
        "--seed", "$Seed",
        "--n_estimators", "$($C.N_Estimators)",
        "--min_samples_split", "$($C.MinSplit)",
        "--min_samples_leaf", "$($C.MinLeaf)",
        "--n_jobs", "-1",
        "--tables-dir", "$Tables"
    )
    if ($null -ne $C.MaxDepth) { $args += @("--max_depth", "$($C.MaxDepth)") }
    if ($null -ne $C.ClassW)   { $args += @("--class_weight", "$($C.ClassW)") }
    return $args
}

$jobs = @()
foreach ($C in $ParamCombos) {
    while ((Get-Job -State Running).Count -ge $MaxParallel) { Start-Sleep -Seconds 2 }

    $cwLabel = if ($null -eq $C.ClassW) { "none" } else { $C.ClassW }
    $name = ("OFFLINE_{0}_nest{1}_depth{2}_split{3}_leaf{4}_cw{5}" -f
        $C.Tag, $C.N_Estimators, ($C.MaxDepth -as [string]), $C.MinSplit, $C.MinLeaf, $cwLabel)

    Write-Host ">> Start $name"
    $args = Build-Args -C $C

    $job = Start-Job -Name $name -ScriptBlock {
        param($PythonExe, $ArgsArray, $WorkDir)
        Set-Location $WorkDir
        & $PythonExe @ArgsArray
    } -ArgumentList @($Python, $args, $BaseDir)

    $jobs += $job
}

Write-Host (">> Zagnanih jobov: {0}. Čakam na zaključek..." -f $jobs.Count)
Wait-Job -Job $jobs | Out-Null

$log = Join-Path $Tables ("logs_offline_all_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
foreach ($j in $jobs) {
    "### " + $j.Name | Out-File -FilePath $log -Append -Encoding UTF8
    (Receive-Job -Job $j -Keep) | Out-File -FilePath $log -Append -Encoding UTF8
    "`n" | Out-File -FilePath $log -Append -Encoding UTF8
}
Write-Host "=== Končano. CSV-ji v $Tables, log: $log ==="
