# ECSIE -- A/B flag-matrix benchmark runner (Windows PowerShell 5.1 compatible)
#
# Runs a manifest of env-flag configurations through ecsie_bench_measure_tps,
# N reps each, with GPU-contention sentinel checks between configs, and emits
# a per-config median/IQR summary judged by lr.tps (NEVER warm_step_rate --
# see docs/benchmarks.md and the warm-step-rate pitfall).
#
# Usage:
#   powershell -File benchmarks\ab_matrix.ps1 -Manifest benchmarks\ab_manifest.json [-OutDir benchmarks\results\ab_<stamp>]
#
# Manifest format (JSON):
# {
#   "model":    "A:\\AI\\Models\\Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
#   "workload": "benchmarks\\workloads\\long_chatml.json",
#   "reps": 5,
#   "configs": [
#     { "name": "base",    "env": { "ECSIE_SPEC": "off" } },
#     { "name": "spec_on", "env": { "ECSIE_SPEC": "on"  } }
#   ]
# }
#
# Conventions enforced here:
#  - every config starts from a CLEAN env (all ECSIE_* vars removed first)
#  - contention sentinel: GPU util/mem sampled before each config; batch aborts
#    if another compute process appears or utilisation exceeds threshold
#  - one bench at a time; reps are sequential (parallel reps would contend)

param(
    [Parameter(Mandatory = $true)] [string]$Manifest,
    [string]$OutDir = "",
    [int]$GpuUtilThresholdPct = 15,
    [int]$GpuMemThresholdMiB = 2500
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # benchmarks\ -> repo root

$mf = Get-Content $Manifest -Raw | ConvertFrom-Json
if ($OutDir -eq "") {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutDir = Join-Path $repo ("benchmarks\results\ab_" + $stamp)
}
New-Item -ItemType Directory -Force $OutDir | Out-Null

function Clear-EcsieEnv {
    Get-ChildItem Env: | Where-Object { $_.Name -like "ECSIE_*" } | ForEach-Object {
        Remove-Item ("Env:" + $_.Name)
    }
}

function Test-GpuQuiet {
    # Returns $true if the GPU looks idle enough for a clean measurement.
    $q = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    if (-not $q) { Write-Warning "nvidia-smi unavailable; skipping sentinel"; return $true }
    $parts = ($q -split ",") | ForEach-Object { $_.Trim() }
    $util = [int]$parts[0]; $mem = [int]$parts[1]
    if ($util -gt $GpuUtilThresholdPct) {
        Write-Warning ("GPU util {0}% > {1}% -- contention" -f $util, $GpuUtilThresholdPct); return $false
    }
    if ($mem -gt $GpuMemThresholdMiB) {
        Write-Warning ("GPU mem {0} MiB > {1} MiB -- another workload resident" -f $mem, $GpuMemThresholdMiB); return $false
    }
    return $true
}

function Get-LrTps {
    param([string]$CsvPath, [string]$StderrPath)
    # Primary: the --out-tps CSV written by measure_tps. Judged metric is the
    # run-level tps (lr.tps). Parser finalised against the actual CSV schema:
    # falls back to the LAST "tps=<x>" on a [latency] line in stderr, which is
    # the cumulative tokens/total-wall figure (NOT warm_step_rate).
    if (Test-Path $CsvPath) {
        try {
            $rows = Import-Csv $CsvPath
            if ($rows.Count -gt 0) {
                $col = $rows[0].PSObject.Properties.Name | Where-Object { $_ -match "^(lr\.)?tps$" } | Select-Object -First 1
                if ($col) { return [double]($rows[-1].$col) }
            }
        } catch { }
    }
    if (Test-Path $StderrPath) {
        $m = Select-String -Path $StderrPath -Pattern "tps=([0-9.]+)" -AllMatches |
             Select-Object -ExpandProperty Matches -ErrorAction SilentlyContinue
        if ($m) { return [double]($m[-1].Groups[1].Value) }
    }
    return $null
}

function Get-Median { param([double[]]$v)
    $s = $v | Sort-Object; $n = $s.Count
    if ($n -eq 0) { return $null }
    if ($n % 2 -eq 1) { return $s[[int][math]::Floor($n/2)] }
    return ($s[$n/2 - 1] + $s[$n/2]) / 2.0
}

$bench = Join-Path $repo "bin\ecsie_bench_measure_tps.exe"
$summary = @()

function Get-WarmRate {
    param([string]$StderrPath)
    # Diagnostic only -- NEVER the judged metric (warm_step_rate pitfall).
    if (Test-Path $StderrPath) {
        $m = Select-String -Path $StderrPath -Pattern "warm_step_rate=([0-9.]+)" -AllMatches |
             Select-Object -ExpandProperty Matches -ErrorAction SilentlyContinue
        if ($m) { return [double]($m[-1].Groups[1].Value) }
    }
    return $null
}

foreach ($cfg in $mf.configs) {
    if (-not (Test-GpuQuiet)) { throw "Sentinel abort before config '$($cfg.name)' -- GPU not quiet." }
    Clear-EcsieEnv
    $envDesc = @()
    if ($cfg.env) {
        foreach ($p in $cfg.env.PSObject.Properties) {
            Set-Item ("Env:" + $p.Name) $p.Value
            $envDesc += ("{0}={1}" -f $p.Name, $p.Value)
        }
    }
    $tpsList = @(); $warmList = @()
    for ($r = 1; $r -le $mf.reps; $r++) {
        $csv = Join-Path $OutDir ("{0}_r{1}.csv" -f $cfg.name, $r)
        $serr = Join-Path $OutDir ("{0}_r{1}.stderr" -f $cfg.name, $r)
        Write-Host ("[ab] {0} rep {1}/{2}  ({3})" -f $cfg.name, $r, $mf.reps, ($envDesc -join " "))
        & cmd /c "`"$bench`" --model `"$($mf.model)`" --workload `"$($mf.workload)`" --out-tps `"$csv`" 2> `"$serr`""
        $tps = Get-LrTps -CsvPath $csv -StderrPath $serr
        $warm = Get-WarmRate -StderrPath $serr
        if ($null -eq $tps) { Write-Warning ("rep {0} produced no parseable tps" -f $r) }
        else {
            $tpsList += $tps
            if ($null -ne $warm) { $warmList += $warm }
            Write-Host ("[ab]   -> lr.tps = {0}  (warm={1})" -f $tps, $warm)
        }
    }
    Clear-EcsieEnv
    $med = Get-Median $tpsList
    $sorted = $tpsList | Sort-Object
    $iqrTxt = ""
    if ($tpsList.Count -ge 4) {
        $q1 = $sorted[[int][math]::Floor(($sorted.Count - 1) * 0.25)]
        $q3 = $sorted[[int][math]::Floor(($sorted.Count - 1) * 0.75)]
        $iqrTxt = ("{0:N2}" -f ($q3 - $q1))
    }
    $warmMed = Get-Median $warmList
    $summary += [pscustomobject]@{
        config = $cfg.name
        env    = ($envDesc -join " ")
        reps   = $tpsList.Count
        median = if ($null -ne $med) { "{0:N2}" -f $med } else { "n/a" }
        min    = if ($sorted.Count) { "{0:N2}" -f $sorted[0] } else { "n/a" }
        max    = if ($sorted.Count) { "{0:N2}" -f $sorted[-1] } else { "n/a" }
        iqr    = $iqrTxt
        warm   = if ($null -ne $warmMed) { "{0:N2}" -f $warmMed } else { "n/a" }
    }
}

# -- emit summary --------------------------------------------------------------
$mdPath = Join-Path $OutDir "summary.md"
$lines = @("# A/B matrix -- $(Get-Date -Format s)", "",
           "Model: ``$($mf.model)``  Workload: ``$($mf.workload)``  Reps: $($mf.reps)", "",
           "Judged metric: **median lr.tps** (tokens / total wall). warm_step_rate is diagnostic only.", "",
           "| config | env | reps | median lr.tps | min | max | IQR | warm (diag) |",
           "|---|---|---|---|---|---|---|---|")
foreach ($row in $summary) {
    $lines += ("| {0} | {1} | {2} | **{3}** | {4} | {5} | {6} | {7} |" -f
        $row.config, $row.env, $row.reps, $row.median, $row.min, $row.max, $row.iqr, $row.warm)
}
$lines | Out-File -FilePath $mdPath -Encoding utf8
Write-Host ""
Write-Host ("[ab] summary written: {0}" -f $mdPath)
$summary | Format-Table -AutoSize
