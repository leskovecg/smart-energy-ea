# 14_grid_online_exploration.ps1
# ONLINE Active Learning (simulator): nove (še neuporabljene) kombinacije I–L × vse 4 strategije
# - variiramo class_weight (None/balanced/balanced_subsample)
# - večje/deeper RF, drugačen min_samples_*
# - dodan drugi seed za robustnost modela

$BaseDir = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SrcDir  = "$BaseDir\src"
$Data    = "$BaseDir\data\simulation_security_labels_n-1.csv"
$Tables  = "$BaseDir\tables"

# python – po želji venv:
# $Python = "$BaseDir\.venv\Scripts\python.exe"
$Python = (& py -3.13 -c "import sys; print(sys.executable)").Trim()
Write-Host "Using Python: $Python"

$MaxParallel = 4
$Strategies  = @("uncertainty","entropy","margin","random")
$Seeds       = @(42, 1337)   # 2 različna seeda

# ---- Kombinacije I–L (razlikujejo se od prejšnjih A–H) ----
$ParamCombos = @(
    # I: večji init, večji batch, srednja globina, leaf=1, class_weight=None
    @{ Tag="I"; Init=350; Batch=120; Iters=18; N_Estimators=900;  MaxDepth=24;  MinSplit=2; MinLeaf=1; ClassW=$null },
    # J: manjši init, velik batch, kratke iteracije, zelo veliko dreves, balanced
    @{ Tag="J"; Init=120; Batch=200; Iters=10; N_Estimators=1500; MaxDepth=28;  MinSplit=2; MinLeaf=1; ClassW="balanced" },
    # K: srednji init, manjši batch, daljše iteracije, plitvejša drevesa, leaf=3
    @{ Tag="K"; Init=220; Batch=60;  Iters=36; N_Estimators=700;  MaxDepth=18;  MinSplit=2; MinLeaf=3; ClassW="balanced_subsample" },
    # L: večji init, srednji batch, srednje iteracije, zelo globoka ali neskončna drevesa, leaf=2
    @{ Tag="L"; Init=400; Batch=80;  Iters=22; N_Estimators=1100; MaxDepth=$null; MinSplit=2; MinLeaf=2; ClassW="balanced" }
)

# privzeti test split (lahko povišaš na 0.2 za strožji test)
$TestSize = 0.1

if (!(Test-Path $Tables)) { New-Item -ItemType Directory -Path $Tables | Out-Null }

function Build-Args {
    param([string]$Strategy, [hashtable]$C, [int]$Seed)
    $args = @(
        "$SrcDir\experiments\run_online_active_learning_with_simulator.py",
        "--data", "$Data",
        "--strategy", "$Strategy",
        "--init", "$($C.Init)",
        "--batch", "$($C.Batch)",
        "--iters", "$($C.Iters)",
        "--test-size", "$TestSize",
        "--n_estimators", "$($C.N_Estimators)",
        "--min_samples_split", "$($C.MinSplit)",
        "--min_samples_leaf", "$($C.MinLeaf)",
        "--n_jobs", "-1",
        "--seed", "$Seed",
        "--tables-dir", "$Tables"
    )
    if ($null -ne $C.MaxDepth) { $args += @("--max_depth", "$($C.MaxDepth)") }
    if ($null -ne $C.ClassW)   { $args += @("--class_weight", "$($C.ClassW)") }
    return $args
}

$jobs = @()
$total = $ParamCombos.Count * $Strategies.Count * $Seeds.Count
$completed = 0

foreach ($C in $ParamCombos) {
    foreach ($S in $Strategies) {
        foreach ($Seed in $Seeds) {
            while ((Get-Job -State Running).Count -ge $MaxParallel) { Start-Sleep -Seconds 2 }

            $name = ("AL_{0}_Set{1}_seed{2}_init{3}_batch{4}_iters{5}_nest{6}_depth{7}_leaf{8}_cw{9}" -f
                $S, $C.Tag, $Seed, $C.Init, $C.Batch, $C.Iters, $C.N_Estimators,
                ($C.MaxDepth -as [string]), $C.MinLeaf, ($C.ClassW -as [string]))

            Write-Host ">> Start $name"
            $args = Build-Args -Strategy $S -C $C -Seed $Seed

            $job = Start-Job -Name $name -ScriptBlock {
                param($PythonExe, $ArgsArray, $WorkDir)
                Set-Location $WorkDir
                & $PythonExe @ArgsArray
            } -ArgumentList @($Python, $args, $BaseDir)

            Register-ObjectEvent -InputObject $job -EventName StateChanged -Action {
                if ($event.Sender.State -eq 'Completed' -or $event.Sender.State -eq 'Failed') {
                    [System.Threading.Interlocked]::Increment([ref]$using:completed) | Out-Null
                    Write-Host ("   Progress: {0}/{1} done" -f $using:completed, $using:total)
                }
            } | Out-Null

            $jobs += $job
        }
    }
}

Write-Host (">> Zagnanih jobov: {0}. Čakam na zaključek..." -f $jobs.Count)
Wait-Job -Job $jobs | Out-Null

# shrani loge
$log = Join-Path $Tables ("logs_exploration_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
foreach ($j in $jobs) {
    "### " + $j.Name | Out-File -FilePath $log -Append -Encoding UTF8
    (Receive-Job -Job $j -Keep) | Out-File -FilePath $log -Append -Encoding UTF8
    "`n" | Out-File -FilePath $log -Append -Encoding UTF8
}

Write-Host "=== Končano. CSV-ji v $Tables, log: $log ==="
