# run_overnight.ps1
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Paths
$ROOT      = "C:\Users\gl8304\Documents\Projekti\IJS\smart-energy-ea"
$SCRIPTDIR = Join-Path $ROOT "src\"
$DATA      = Join-Path $ROOT "data\simulation_security_labels_n-1.csv"
$TABLES    = Join-Path $SCRIPTDIR "tables"
$LOGS      = Join-Path $TABLES "logs"

# Create dirs
New-Item -ItemType Directory -Force -Path $TABLES | Out-Null
New-Item -ItemType Directory -Force -Path $LOGS   | Out-Null

# Common params
$INIT   = 100
$BATCH  = 50
$ITERS  = 20
$TEST   = 0.1
$AVGSEC = 2.3

# What to run overnight
$strategies = @("margin","uncertainty")  # po želji dodaj: "entropy","random"
$seeds      = 1..3

Write-Host ">>> Starting overnight AL runs..." -ForegroundColor Cyan
Set-Location $SCRIPTDIR

foreach ($s in $strategies) {
  foreach ($seed in $seeds) {
    $ts  = Get-Date -Format "yyyyMMdd_HHmmss"
    $log = Join-Path $LOGS "run_${s}_seed${seed}_$ts.log"
    Write-Host ">>> [$s | seed=$seed] starting... (log: $log)" -ForegroundColor Yellow

    & python "$SCRIPTDIR\experiments\run_online_active_learning_with.py" `
      --data       "$DATA" `
      --strategy   "$s" `
      --init       $INIT `
      --batch      $BATCH `
      --iters      $ITERS `
      --test-size  $TEST `
      --seed       $seed `
      --avg-sim-sec $AVGSEC `
      --tables-dir "$TABLES" 2>&1 | Tee-Object -FilePath $log

    if ($LASTEXITCODE -ne 0) {
      Write-Host "!!! [$s | seed=$seed] FAILED (exit $LASTEXITCODE). See: $log" -ForegroundColor Red
    } else {
      Write-Host ">>> [$s | seed=$seed] done." -ForegroundColor Green
    }
  }
}

Write-Host ">>> All runs finished. Logs at: $LOGS" -ForegroundColor Cyan

# (Optional) Avtomatska agregacija figur po koncu:
# & python "$SCRIPTDIR\analysis.py" `
#   --tables-dir  "$TABLES" `
#   --figures-dir (Join-Path $SCRIPTDIR "figures") `
#   --acc-targets 0.90 0.92 `
#   --auc-targets 0.85 0.90
