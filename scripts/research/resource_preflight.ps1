$ErrorActionPreference = 'Stop'
if (-not ('ResearchForegroundWindow' -as [type])) {
    Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class ResearchForegroundWindow {
    [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
    [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint id);
}
'@
}
[uint32]$foregroundId = 0
$null = [ResearchForegroundWindow]::GetWindowThreadProcessId(
    [ResearchForegroundWindow]::GetForegroundWindow(), [ref]$foregroundId)
$foregroundName = (Get-Process -Id $foregroundId -ErrorAction SilentlyContinue).ProcessName
$cpuSamples = @(1..4 | ForEach-Object {
    (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor -Filter "Name='_Total'").PercentProcessorTime
    Start-Sleep -Seconds 1
})
$memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
$top = @(Get-CimInstance Win32_PerfFormattedData_PerfProc_Process |
    Where-Object { $_.Name -notmatch 'Idle|_Total' } |
    Sort-Object PercentProcessorTime -Descending |
    Select-Object -First 6 Name,IDProcess,
        @{n='PercentOfSystemCPU';e={[math]::Round($_.PercentProcessorTime / 16,1)}})
$knownTasks = @(Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match 'python|blender|UnityShaderCompiler|il2cpp|Ld9BoxHeadless' -and
        ($_.CommandLine -like '*stock-recommendation*' -or
         $_.CommandLine -match 'multiprocessing.spawn|blender.*\.py|UnityShaderCompiler|il2cpp|Ld9BoxHeadless')
    } | Select-Object Name,ProcessId,ParentProcessId)
[pscustomobject]@{
    cpu_samples = $cpuSamples
    cpu_average = [math]::Round(($cpuSamples | Measure-Object -Average).Average,1)
    available_ram_gb = [math]::Round($memory.AvailableMBytes / 1024,2)
    foreground = $foregroundName
    top_cpu = $top
    relevant_jobs = $knownTasks
} | ConvertTo-Json -Depth 4
