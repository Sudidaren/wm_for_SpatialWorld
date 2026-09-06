# =============================================================================
# Windows 侧真实环境测试脚本：VirtualHome / CARLA / EmbodiedCity
# 每个环境跑 2 个任务，验证 Failure Detection Agent（已接入）在真实模拟器中的表现。
#
# 用法（PowerShell 7，在仓库根目录执行）：
#   $env:OPENAI_API_KEY = "<你的Key>"
#   .\scripts\test_windows_envs.ps1
#
# 说明：
#   1. 修改下面 $VH_LAUNCHER / $CARLA_LAUNCHER / $EC_LAUNCHER 为你桌面上的实际启动器路径。
#   2. API 网关默认使用 OPENAI_BASE_URL 环境变量；不设置时按 https://apic1.ohmycdn.com/v1。
#   3. EmbodiedCity 的 AirSim 服务器（Blocks.exe / TrafficSimulation.exe）需要在 Windows 上运行。
#   4. 若网关模型（如 gpt-5）不支持 top_p：VirtualHome 配置里把 model.vlm.top_p 置 null；
#      EmbodiedCity 用 /tmp/ec_run_patched.py（运行时去掉 top_p）或换支持 top_p 的模型。
#   5. CARLA 必须 Epic 画质并在每个任务前重启服务器，否则易崩（见 benchmark_carla.py）。
# =============================================================================

param(
    [string]$RepoRoot        = "C:\Users\29742\SpatialWorld",
    [string]$VHLauncher      = "",   # 例如 "C:\Users\29742\Desktop\VirtualHome.exe"
    [string]$CarlaLauncher   = "",   # 例如 "C:\Users\29742\Desktop\CarlaUE4.exe"
    [string]$ECLauncher      = "",   # 例如 "C:\Users\29742\Desktop\Blocks.exe" (EmbodiedCity AirSim 服务器)
    [string]$ApiBaseUrl      = "https://apic1.ohmycdn.com/v1",
    [string]$Model           = "gpt-5",
    [int]$MaxSteps           = 40
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

if (-not $env:OPENAI_API_KEY) {
    Write-Host "❌ 请先设置环境变量 OPENAI_API_KEY" -ForegroundColor Red
    exit 1
}
$env:OPENAI_BASE_URL = $ApiBaseUrl

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host "`n$('=' * 70)" -ForegroundColor Cyan
    Write-Host "▶ $Name" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
    & $Body
}

# ---------------------------------------------------------------------------
# 1. VirtualHome（2 个任务）
# ---------------------------------------------------------------------------
Invoke-Step "VirtualHome: virtualhome00000 + virtualhome00001" {
    if (-not $VHLauncher) { throw "请设置 -VHLauncher 参数为 VirtualHome Unity 模拟器路径" }
    if (-not (Test-Path $VHLauncher)) { throw "找不到 VirtualHome 模拟器: $VHLauncher" }

    $vh = Start-Process -FilePath $VHLauncher -ArgumentList "-windowed -screen-width 960 -screen-height 540" -PassThru
    Write-Host "VirtualHome 模拟器已启动 (PID $($vh.Id))，等待 90s 就绪..."
    Start-Sleep -Seconds 90

    try {
        .\envs\virtualhome\.venv\Scripts\python.exe -u scripts\virtualhome\work\run_task.py `
            --config experiments\configs\virtualhome\config_close_gpt-5.yaml `
            --tasks virtualhome00000 virtualhome00001 `
            --output-dir test_virtualhome
    } finally {
        Stop-Process -Id $vh.Id -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# 2. CARLA（2 个任务，Epic 画质 + 每任务重启服务器，参考 benchmark_carla.py）
# ---------------------------------------------------------------------------
Invoke-Step "CARLA: carla00000 + carla00200" {
    if (-not $CarlaLauncher) { throw "请设置 -CarlaLauncher 参数为 CARLA 服务器路径" }
    if (-not (Test-Path $CarlaLauncher)) { throw "找不到 CARLA 服务器: $CarlaLauncher" }

    # CARLA 在 Low 画质下 Town02 渲染会崩，必须 Epic；且每个任务前重启服务器
    foreach ($task in @("carla00000", "carla00200")) {
        $carla = Start-Process -FilePath $CarlaLauncher `
            -ArgumentList "-carla-rpc-port=2000", "-quality-level=Epic" -PassThru
        Write-Host "CARLA 服务器已启动 (PID $($carla.Id), Epic)，等待 90s 就绪..."
        Start-Sleep -Seconds 90

        try {
            .\envs\carla\.venv\Scripts\python.exe -u scripts\carla\work\run_task.py `
                --config experiments\configs\carla\config_close_gpt-5.yaml `
                --tasks $task `
                --output-dir test_carla
        } finally {
            Stop-Process -Id $carla.Id -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 10
        }
    }
}

# ---------------------------------------------------------------------------
# 3. EmbodiedCity（2 个任务）
# ---------------------------------------------------------------------------
Invoke-Step "EmbodiedCity: embodiedcity_2000_checked + embodiedcity_2001_checked" {
    if (-not $ECLauncher) { throw "请设置 -ECLauncher 参数为 EmbodiedCity AirSim 服务器路径 (Blocks.exe)" }
    if (-not (Test-Path $ECLauncher)) { throw "找不到 EmbodiedCity 服务器: $ECLauncher" }

    $ec = Start-Process -FilePath $ECLauncher -PassThru
    Write-Host "EmbodiedCity AirSim 服务器已启动 (PID $($ec.Id))，等待 60s 就绪..."
    Start-Sleep -Seconds 60

    try {
        $env:AIRSIM_PORT = "41451"
        .\envs\embodiedcity\.venv\Scripts\python.exe -u scripts\embodiedcity\work\run_task.py `
            --config experiments\configs\embodiedcity\vln-agent-config-gpt54.yaml `
            --model $Model `
            --api-base $ApiBaseUrl `
            --api-key $env:OPENAI_API_KEY `
            --tasks embodiedcity_2000_checked embodiedcity_2001_checked `
            --output-dir test_embodiedcity
    } finally {
        Stop-Process -Id $ec.Id -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "`n✅ 全部完成。结果与 failure detection 日志见 test_virtualhome / test_carla / test_embodiedcity 目录。" -ForegroundColor Green
