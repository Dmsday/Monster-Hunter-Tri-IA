#Requires -Version 5.1
<#
.SYNOPSIS
    Launch multiple Dolphin instances with automatic window renaming, INI tweaks
    and User-profile management.

.DESCRIPTION
    Uses Dolphin's --user flag with sibling profiles (User, User1, User2, ...)
    placed next to Dolphin.exe. Missing profiles are auto-created from "User".

    Two modes:
      * Interactive (default) : shows a small WinForms launcher + control panel.
      * NoGUI                  : driven by Python (multi_agent_trainer), writes
                                 PID files so Python can attach to each Dolphin.
#>

[CmdletBinding()]
param(
    [int]$NumInstances     = 0,
    [switch]$NoGUI,
    [switch]$MinimizeDolphin = $false,
    [switch]$MinimizeGame    = $false,
    [string]$DolphinExePath  = "",
    [string]$UserFolderPath  = "",
    [string]$RomFilePath     = "",
    [string]$PidDirectory    = ""
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# ============================================================================
# CONSTANTS
# ============================================================================
$Script:ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Script:ConfigFile  = Join-Path $Script:ScriptDir "dolphin_launcher_config.json"
$Script:RomPatterns = '(?i)monster.*hunter|mhtri|RMHP'
$Script:RomExts     = @('*.rvz', '*.iso', '*.wbfs', '*.gcm')
$Script:WindowTimeoutSec = 15
$Script:InitialDelaySec  = 5
$Script:WindowMatchPatterns = @('*Monster Hunter Tri*', '*RMHP*')
$Script:DolphinMenuTitle    = 'Dolphin 2509'

# ============================================================================
# LOGGING HELPERS (single source of truth for output formatting)
# ============================================================================
function Write-Info  { param([string]$Msg) Write-Host $Msg -ForegroundColor Cyan }
function Write-Ok    { param([string]$Msg) Write-Host $Msg -ForegroundColor Green }
function Write-Warn2 { param([string]$Msg) Write-Host $Msg -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host $Msg -ForegroundColor Red }
function Write-Dim   { param([string]$Msg) Write-Host $Msg -ForegroundColor DarkGray }

# ============================================================================
# CONFIG PERSISTENCE
# ============================================================================
function Get-SavedConfig {
    if (-not (Test-Path $Script:ConfigFile)) { return $null }
    try {
        return Get-Content -Path $Script:ConfigFile -Raw | ConvertFrom-Json
    }
    catch [System.Exception] {
        Write-Warn2 "Failed to load saved config: $($_.Exception.Message)"
        return $null
    }
}

function Save-LauncherConfig {
    param([string]$DolphinPath, [string]$UserFolder, [string]$RomPath)
    $data = [ordered]@{
        DolphinExePath = $DolphinPath
        UserFolderPath = $UserFolder
        RomFilePath    = $RomPath
        LastUpdated    = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    }
    try {
        $data | ConvertTo-Json | Set-Content -Path $Script:ConfigFile -Encoding UTF8
        Write-Ok "Configuration saved: $Script:ConfigFile"
    }
    catch [System.Exception] {
        Write-Warn2 "Failed to save config: $($_.Exception.Message)"
    }
}

# ============================================================================
# PATH RESOLUTION
# ============================================================================
function Resolve-DolphinExe {
    param([string]$Hint)
    if ($Hint) { return [System.IO.Path]::GetFullPath($Hint) }

    $candidates = @(
        (Join-Path $Script:ScriptDir 'Dolphin.exe'),
        (Join-Path (Split-Path -Parent $Script:ScriptDir) 'Dolphin.exe')
    )
    foreach ($c in $candidates) {
        if (Test-Path $c -PathType Leaf) { return [System.IO.Path]::GetFullPath($c) }
    }
    return $null
}

function Resolve-UserFolder {
    param([string]$Hint, [string]$DolphinDir)
    if ($Hint) { return $Hint.TrimEnd('\') }

    $portable  = Join-Path $DolphinDir 'User'
    $installed = Join-Path $env:USERPROFILE 'Documents\Dolphin Emulator'
    if (Test-Path $portable  -PathType Container) { return $portable.TrimEnd('\')  }
    if (Test-Path $installed -PathType Container) { return $installed.TrimEnd('\') }
    return $null
}

function Resolve-RomFile {
    param([string]$Hint, [string]$DolphinDir)
    if ($Hint) { return $Hint }

    # Walk up to 3 ancestors of Dolphin dir, looking for Jeux/Games/ROMs subfolders.
    $parents = @($DolphinDir)
    $cur = $DolphinDir
    for ($i = 0; $i -lt 3; $i++) {
        $cur = Split-Path -Parent $cur
        if ($cur) { $parents += $cur }
    }

    $roots = foreach ($p in $parents) {
        Join-Path $p 'Jeux'; Join-Path $p 'Games'; Join-Path $p 'ROMs'
    }
    $roots = $roots | Select-Object -Unique

    foreach ($root in $roots) {
        if (-not (Test-Path $root -PathType Container)) { continue }
        foreach ($ext in $Script:RomExts) {
            $found = Get-ChildItem -Path $root -Filter $ext -Recurse -ErrorAction SilentlyContinue |
                     Where-Object { $_.Name -match $Script:RomPatterns } |
                     Select-Object -First 1
            if ($found) { return $found.FullName }
        }
    }
    return $null
}

# ============================================================================
# WINAPI for window enumeration / rename
# ============================================================================
if (-not ([System.Management.Automation.PSTypeName]'WindowManager').Type) {
    Add-Type @"
    using System;
    using System.Runtime.InteropServices;
    using System.Text;
    using System.Collections.Generic;

    public class WindowManager {
        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);
        [DllImport("user32.dll")]
        public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
        [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
        public static extern bool SetWindowText(IntPtr hWnd, string lpString);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool IsWindowVisible(IntPtr hWnd);
        [DllImport("user32.dll", SetLastError = true)]
        public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);
        [DllImport("user32.dll")]
        public static extern bool SetForegroundWindow(IntPtr hWnd);
        [DllImport("user32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

        public class WindowInfo { public IntPtr Handle; public string Title; }

        public static List<WindowInfo> GetProcessWindows(int processId) {
            List<WindowInfo> windows = new List<WindowInfo>();
            EnumWindows(delegate(IntPtr hWnd, IntPtr lParam) {
                if (IsWindowVisible(hWnd)) {
                    uint pid;
                    GetWindowThreadProcessId(hWnd, out pid);
                    if (pid == processId) {
                        StringBuilder sb = new StringBuilder(512);
                        GetWindowText(hWnd, sb, sb.Capacity);
                        string title = sb.ToString();
                        if (!string.IsNullOrEmpty(title)) {
                            windows.Add(new WindowInfo { Handle = hWnd, Title = title });
                        }
                    }
                }
                return true;
            }, IntPtr.Zero);
            return windows;
        }
    }
"@
}

# ============================================================================
# INI HELPER — single function replaces 4 copy-pasted blocks
# ============================================================================
function Set-IniValue {
    <#
    .SYNOPSIS
        Set a key=value pair inside a section of an INI file.
        Creates the section / key if missing, overwrites if present.
    #>
    param(
        [Parameter(Mandatory)] [string] $Path,
        [Parameter(Mandatory)] [string] $Section,
        [Parameter(Mandatory)] [string] $Key,
        [Parameter(Mandatory)] [string] $Value
    )

    $lines = @()
    if (Test-Path $Path) {
        $lines = Get-Content -Path $Path -Encoding UTF8
    }

    $output            = New-Object System.Collections.Generic.List[string]
    $sectionHeader     = "[$Section]"
    $newLine           = "$Key = $Value"
    $insideTarget      = $false
    $keyWritten        = $false
    $sectionFound      = $false

    foreach ($line in $lines) {
        # Section header
        if ($line -match '^\s*\[(.+)\]\s*$') {
            # If leaving the target section without having written the key, append it now
            if ($insideTarget -and -not $keyWritten) {
                $output.Add($newLine)
                $keyWritten = $true
            }
            $insideTarget = ($line.Trim() -eq $sectionHeader)
            if ($insideTarget) { $sectionFound = $true }
            $output.Add($line)
            continue
        }

        # Key inside the target section: replace
        if ($insideTarget -and ($line -match "^\s*$([regex]::Escape($Key))\s*=")) {
            if (-not $keyWritten) {
                $output.Add($newLine)
                $keyWritten = $true
            }
            continue
        }

        $output.Add($line)
    }

    # End-of-file fallbacks
    if ($insideTarget -and -not $keyWritten) {
        $output.Add($newLine)
        $keyWritten = $true
    }
    if (-not $sectionFound) {
        $output.Add('')
        $output.Add($sectionHeader)
        $output.Add($newLine)
    }

    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($Path, $output, $utf8NoBom)
}

function Remove-StaleIniPaths {
    <#
    .SYNOPSIS
        Strip absolute-path keys from Dolphin.ini whose target no longer exists.
        Prevents Dolphin from re-creating ghost folder trees on startup.
    #>
    param([Parameter(Mandatory)] [string] $Path)
    if (-not (Test-Path $Path)) { return 0 }

    $stalePatterns = @(
        '^\s*ISOPath\d+\s*=', '^\s*ISOPaths\s*=', '^\s*BootDefaultISO\s*=',
        '^\s*WiiSDCardPath\s*=', '^\s*LastFilename\s*=',
        '^\s*NANDRootPath\s*=', '^\s*DumpPath\s*='
    )
    $kept    = New-Object System.Collections.Generic.List[string]
    $removed = 0

    foreach ($line in (Get-Content -Path $Path -Encoding UTF8)) {
        $isStale = $false
        foreach ($pat in $stalePatterns) {
            if ($line -match $pat -and $line -match '=\s*"?([a-zA-Z]:[\\/].+?)"?\s*$') {
                $val = $Matches[1].Trim('"').Trim()
                if (-not (Test-Path -LiteralPath $val)) {
                    $isStale = $true
                    $removed++
                    break
                }
            }
        }
        if (-not $isStale) { $kept.Add($line) }
    }

    if ($removed -gt 0) {
        $utf8NoBom = New-Object System.Text.UTF8Encoding $false
        [System.IO.File]::WriteAllLines($Path, $kept, $utf8NoBom)
    }
    return $removed
}

# ============================================================================
# PROFILE MANAGEMENT
# ============================================================================
function Get-UserProfiles {
    param([Parameter(Mandatory)] [string] $DolphinDir, [Parameter(Mandatory)] [string] $BaseUserFolder)

    $profiles = @(
        Get-ChildItem -Path $DolphinDir -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^User(\d*)$' } |
            ForEach-Object {
                $idx = if ($_.Name -eq 'User') { 0 } else { [int]($_.Name -replace 'User','') }
                [PSCustomObject]@{ Name = $_.Name; Index = $idx; Path = $_.FullName }
            } |
            Sort-Object Index
    )

    # Fallback : non-portable install where User lives outside Dolphin dir
    if ($profiles.Count -eq 0 -and (Test-Path $BaseUserFolder -PathType Container)) {
        $profiles = @([PSCustomObject]@{
            Name  = (Split-Path -Leaf $BaseUserFolder)
            Index = 0
            Path  = $BaseUserFolder
        })
    }
    return ,$profiles
}

function Initialize-UserProfiles {
    <#
    .SYNOPSIS
        Make sure User, User1 ... User(N-1) exist next to Dolphin.exe.
        Returns $true on success, $false if any required profile failed.
    #>
    param(
        [Parameter(Mandatory)] [int]    $NumInstances,
        [Parameter(Mandatory)] [string] $BaseUserFolder,
        [Parameter(Mandatory)] [string] $DolphinDir
    )

    if (-not (Test-Path $BaseUserFolder -PathType Container)) {
        Write-Err "Base User folder not found: $BaseUserFolder"
        return $false
    }

    $created = 0; $existing = 0; $failed = 0

    for ($i = 1; $i -lt $NumInstances; $i++) {
        $target = Join-Path $DolphinDir "User$i"
        if (Test-Path $target -PathType Container) {
            $existing++
            continue
        }

        try {
            Copy-Item -Path $BaseUserFolder -Destination $target -Recurse -Force -ErrorAction Stop

            # Drop empty GBA folder Dolphin creates by default (not used by Wii games)
            $gba = Join-Path $target 'GBA'
            if ((Test-Path $gba -PathType Container) -and
                (-not (Get-ChildItem -Path $gba -Recurse -File -ErrorAction SilentlyContinue))) {
                Remove-Item -Path $gba -Recurse -Force -ErrorAction SilentlyContinue
            }

            # Make sure Config exists
            $cfg = Join-Path $target 'Config'
            if (-not (Test-Path $cfg)) { New-Item -ItemType Directory -Path $cfg -Force | Out-Null }

            Write-Ok  "  User$i : created"
            $created++
        }
        catch [System.IO.IOException], [System.UnauthorizedAccessException], [System.Exception] {
            Write-Err "  User$i : failed - $($_.Exception.Message)"
            $failed++
        }
    }

    Write-Info "Profiles: $existing existing, $created created, $failed failed"
    return ($failed -eq 0)
}

function Set-DolphinInstanceConfig {
    <#
    .SYNOPSIS
        Apply all required INI tweaks for one Dolphin instance:
          - GFX.ini : RenderToMain = False (so we can minimize without freezing render)
          - Dolphin.ini : Backend=No audio, Volume=0, CPUThread=False, PauseOnFocusLost=False
          - Strip stale absolute paths from Dolphin.ini
    #>
    param([Parameter(Mandatory)] [string] $UserFolderPath, [Parameter(Mandatory)] [int] $Index)

    if (-not (Test-Path $UserFolderPath -PathType Container)) {
        Write-Warn2 "  Instance $Index : User folder missing, skipping INI config"
        return
    }

    $cfgDir = Join-Path $UserFolderPath 'Config'
    if (-not (Test-Path $cfgDir)) { New-Item -ItemType Directory -Path $cfgDir -Force | Out-Null }

    $gfxIni     = Join-Path $cfgDir 'GFX.ini'
    $dolphinIni = Join-Path $cfgDir 'Dolphin.ini'

    try {
        # GFX: render even when minimized
        Set-IniValue -Path $gfxIni -Section 'Settings' -Key 'RenderToMain' -Value 'False'

        # Strip ghost paths BEFORE writing new keys, otherwise we'd preserve them
        $removed = Remove-StaleIniPaths -Path $dolphinIni
        if ($removed -gt 0) {
            Write-Dim "  Instance $Index : removed $removed stale path(s) from Dolphin.ini"
        }

        # Audio off
        Set-IniValue -Path $dolphinIni -Section 'DSP'       -Key 'Backend' -Value 'No audio'
        Set-IniValue -Path $dolphinIni -Section 'DSP'       -Key 'Volume'  -Value '0'
        # Single-core (avoid race condition crashes when many instances run together)
        Set-IniValue -Path $dolphinIni -Section 'Core'      -Key 'CPUThread' -Value 'False'
        # Don't pause when window loses focus (training cycles through instances)
        Set-IniValue -Path $dolphinIni -Section 'Interface' -Key 'PauseOnFocusLost' -Value 'False'

        Write-Ok "  Instance $Index : INI configured (audio muted, render-when-minimized, no focus pause)"
    }
    catch [System.Exception] {
        Write-Err "  Instance $Index : INI write failed - $($_.Exception.Message)"
    }
}

# ============================================================================
# LAUNCH + WINDOW RENAME
# ============================================================================
function Start-DolphinInstance {
    param(
        [Parameter(Mandatory)] [int]    $Index,
        [Parameter(Mandatory)] [string] $DolphinExe,
        [Parameter(Mandatory)] [string] $DolphinDir,
        [string] $RomPath
    )

    $userName = if ($Index -eq 0) { 'User' } else { "User$Index" }
    $userPath = [System.IO.Path]::GetFullPath((Join-Path $DolphinDir $userName))

    if (-not (Test-Path $userPath -PathType Container)) {
        Write-Warn2 "  User folder missing: $userPath (Dolphin will create a fresh one)"
    }

    $argLine = if ([string]::IsNullOrEmpty($RomPath)) {
        "--user `"$userPath`""
    } else {
        "--user `"$userPath`" `"$RomPath`""
    }

    try {
        $proc = Start-Process -FilePath $DolphinExe `
                              -ArgumentList $argLine `
                              -WorkingDirectory $DolphinDir `
                              -PassThru -ErrorAction Stop
        return [PSCustomObject]@{
            Process     = $proc
            Index       = $Index
            UserProfile = $userName
            Title       = "MHTri-$Index"
        }
    }
    catch [System.Exception] {
        Write-Warn2 "  Launch failed for User$Index : $($_.Exception.Message)"
        return $null
    }
}

function Rename-DolphinWindow {
    param(
        [Parameter(Mandatory)] [int]    $ProcessId,
        [Parameter(Mandatory)] [string] $NewTitle,
        [int]  $TimeoutSec     = 15,
        [bool] $MinimizeMenu   = $false,
        [bool] $MinimizeGame   = $false
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $windows = [WindowManager]::GetProcessWindows($ProcessId)

            if ($MinimizeMenu) {
                foreach ($w in $windows) {
                    if ($w.Title -eq $Script:DolphinMenuTitle) {
                        [WindowManager]::ShowWindow($w.Handle, 6) | Out-Null   # SW_MINIMIZE
                    }
                }
            }

            foreach ($w in $windows) {
                $isGameWindow = $false
                foreach ($pat in $Script:WindowMatchPatterns) {
                    if ($w.Title -like $pat) { $isGameWindow = $true; break }
                }
                if (-not $isGameWindow) { continue }

                if ([WindowManager]::SetWindowText($w.Handle, $NewTitle)) {
                    if ($MinimizeGame) {
                        [WindowManager]::ShowWindow($w.Handle, 6) | Out-Null
                    }
                    return $true
                }
            }
        }
        catch [System.Exception] {
            # Window enum can race with Dolphin startup; just retry until deadline
        }
        Start-Sleep -Milliseconds 250
    }
    return $false
}

# ============================================================================
# PID FILE I/O (NoGUI mode only)
# ============================================================================
function Write-PidFile {
    param(
        [Parameter(Mandatory)] [int] $Index,
        [Parameter(Mandatory)] [int] $ProcessId
    )
    $dir = if ([string]::IsNullOrEmpty($PidDirectory)) { $Script:ScriptDir } else { $PidDirectory }
    if (-not (Test-Path $dir -PathType Container)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $file = Join-Path $dir "dolphin_pid_$Index.tmp"
    try {
        $ProcessId | Out-File -FilePath $file -Encoding ASCII -Force
        Write-Dim "  PID file: $file"
    }
    catch [System.Exception] {
        Write-Err "  PID file write failed: $($_.Exception.Message)"
    }
}

# ============================================================================
# CLEANUP HELPERS
# ============================================================================
function Remove-EmptyGbaFolders {
    param([Parameter(Mandatory)] [string] $DolphinDir, [Parameter(Mandatory)] [int] $Count)
    for ($i = 0; $i -lt $Count; $i++) {
        $name = if ($i -eq 0) { 'User' } else { "User$i" }
        $gba  = Join-Path (Join-Path $DolphinDir $name) 'GBA'
        if (-not (Test-Path $gba -PathType Container)) { continue }
        if (Get-ChildItem -Path $gba -Recurse -File -ErrorAction SilentlyContinue) { continue }

        Remove-Item -Path $gba -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path $gba) {
            Start-Sleep -Seconds 3
            Remove-Item -Path $gba -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Remove-PhantomDesktopFolders {
    $desktop = Join-Path $env:USERPROFILE 'Desktop'
    if (-not (Test-Path $desktop -PathType Container)) { return }

    Get-ChildItem -Path $desktop -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(?i)Monster\s*Hunter' } |
        ForEach-Object {
            $files = Get-ChildItem -Path $_.FullName -Recurse -File -ErrorAction SilentlyContinue
            if (-not $files) {
                Remove-Item -Path $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
                if (-not (Test-Path $_.FullName)) {
                    Write-Dim "Cleaned phantom Desktop folder: $($_.Name)"
                }
            }
        }
}

# ============================================================================
# GUI : LAUNCHER DIALOG + CONTROL PANEL
# ============================================================================
function Show-LauncherDialog {
    param([Parameter(Mandatory)] [array] $Profiles)

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Dolphin Multi-Instance Launcher"
    $form.Size = New-Object System.Drawing.Size(450, 240)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false

    function New-Label($text, $x, $y, $w = 250) {
        $l = New-Object System.Windows.Forms.Label
        $l.Text = $text
        $l.Location = New-Object System.Drawing.Point($x, $y)
        $l.Size = New-Object System.Drawing.Size($w, 20)
        $form.Controls.Add($l)
    }

    New-Label "Number of instances (1-20):" 20 20
    $numCount = New-Object System.Windows.Forms.NumericUpDown
    $numCount.Minimum = 1; $numCount.Maximum = 20
    $numCount.Value = [Math]::Max(1, [Math]::Min(3, $Profiles.Count))
    $numCount.Location = New-Object System.Drawing.Point(280, 18)
    $numCount.Size = New-Object System.Drawing.Size(120, 20)
    $form.Controls.Add($numCount)

    New-Label "Initial delay before rename (sec):" 20 55
    $numDelay = New-Object System.Windows.Forms.NumericUpDown
    $numDelay.Minimum = 0; $numDelay.Maximum = 30
    $numDelay.Value = $Script:InitialDelaySec
    $numDelay.Location = New-Object System.Drawing.Point(280, 53)
    $numDelay.Size = New-Object System.Drawing.Size(120, 20)
    $form.Controls.Add($numDelay)

    $chkMinDolphin = New-Object System.Windows.Forms.CheckBox
    $chkMinDolphin.Text = "Minimize Dolphin menu windows"
    $chkMinDolphin.Location = New-Object System.Drawing.Point(20, 90)
    $chkMinDolphin.Size = New-Object System.Drawing.Size(380, 20)
    $chkMinDolphin.Checked = $true
    $form.Controls.Add($chkMinDolphin)

    $chkMinGame = New-Object System.Windows.Forms.CheckBox
    $chkMinGame.Text = "Minimize game windows"
    $chkMinGame.Location = New-Object System.Drawing.Point(20, 115)
    $chkMinGame.Size = New-Object System.Drawing.Size(380, 20)
    $form.Controls.Add($chkMinGame)

    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = "Launch"
    $btnOk.Location = New-Object System.Drawing.Point(150, 150)
    $btnOk.Size = New-Object System.Drawing.Size(100, 30)
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Controls.Add($btnOk); $form.AcceptButton = $btnOk

    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = "Cancel"
    $btnCancel.Location = New-Object System.Drawing.Point(260, 150)
    $btnCancel.Size = New-Object System.Drawing.Size(100, 30)
    $btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($btnCancel); $form.CancelButton = $btnCancel

    if ($form.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }

    return @{
        Count           = [int]$numCount.Value
        InitialDelay    = [int]$numDelay.Value
        MinimizeDolphin = $chkMinDolphin.Checked
        MinimizeGame    = $chkMinGame.Checked
    }
}

function Show-ControlPanel {
    param([Parameter(Mandatory)] [array] $Instances)

    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Dolphin Instance Manager"
    $form.Size = New-Object System.Drawing.Size(550, 350)
    $form.StartPosition = "CenterScreen"
    $form.TopMost = $true

    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = "Active instances (select to close):"
    $lbl.Location = New-Object System.Drawing.Point(20, 15)
    $lbl.Size = New-Object System.Drawing.Size(500, 20)
    $form.Controls.Add($lbl)

    $listBox = New-Object System.Windows.Forms.ListBox
    $listBox.Location = New-Object System.Drawing.Point(20, 40)
    $listBox.Size = New-Object System.Drawing.Size(500, 200)
    $listBox.SelectionMode = "MultiExtended"
    $form.Controls.Add($listBox)

    $refresh = {
        $listBox.Items.Clear()
        foreach ($inst in $Instances) {
            $status = if ($inst.Process.HasExited) { "[CLOSED]" } else { "[ACTIVE]" }
            $listBox.Items.Add("$status $($inst.Title) - PID:$($inst.Process.Id) - $($inst.UserProfile)") | Out-Null
        }
    }
    & $refresh

    $closeProc = {
        param($inst)
        try {
            if (-not $inst.Process.HasExited) {
                $inst.Process.CloseMainWindow() | Out-Null
                Start-Sleep -Milliseconds 300
                if (-not $inst.Process.HasExited) { $inst.Process.Kill() }
            }
        }
        catch [System.Exception] {
            Write-Warn2 "Close failed PID $($inst.Process.Id): $($_.Exception.Message)"
        }
    }

    $btnClose = New-Object System.Windows.Forms.Button
    $btnClose.Text = "Close selected"
    $btnClose.Location = New-Object System.Drawing.Point(20, 260)
    $btnClose.Size = New-Object System.Drawing.Size(150, 35)
    $btnClose.Add_Click({
        foreach ($i in $listBox.SelectedIndices) { & $closeProc $Instances[$i] }
        & $refresh
    })
    $form.Controls.Add($btnClose)

    $btnCloseAll = New-Object System.Windows.Forms.Button
    $btnCloseAll.Text = "Close all"
    $btnCloseAll.Location = New-Object System.Drawing.Point(190, 260)
    $btnCloseAll.Size = New-Object System.Drawing.Size(150, 35)
    $btnCloseAll.Add_Click({
        foreach ($inst in $Instances) { & $closeProc $inst }
        $form.Close()
    })
    $form.Controls.Add($btnCloseAll)

    $btnQuit = New-Object System.Windows.Forms.Button
    $btnQuit.Text = "Quit (leave open)"
    $btnQuit.Location = New-Object System.Drawing.Point(360, 260)
    $btnQuit.Size = New-Object System.Drawing.Size(160, 35)
    $btnQuit.Add_Click({ $form.Close() })
    $form.Controls.Add($btnQuit)

    $form.Add_Shown({
        [WindowManager]::SetForegroundWindow($form.Handle) | Out-Null
        $form.Activate()
    })
    $form.ShowDialog() | Out-Null
}

# ============================================================================
# MAIN
# ============================================================================
Write-Info "=== Dolphin Multi-Instance Launcher ==="

# ---- Step 1 : load saved config to fill missing parameters -----------------
$saved = Get-SavedConfig
if ($saved) {
    if (-not $DolphinExePath -and $saved.PSObject.Properties['DolphinExePath']) { $DolphinExePath = $saved.DolphinExePath }
    if (-not $UserFolderPath -and $saved.PSObject.Properties['UserFolderPath']) { $UserFolderPath = $saved.UserFolderPath }
    if (-not $RomFilePath    -and $saved.PSObject.Properties['RomFilePath'])    { $RomFilePath    = $saved.RomFilePath }
}

# ---- Step 2 : resolve all paths --------------------------------------------
$DolphinExePath = Resolve-DolphinExe -Hint $DolphinExePath
if (-not $DolphinExePath) {
    Write-Err "Dolphin.exe not found. Pass -DolphinExePath or place this script next to Dolphin.exe."
    exit 1
}
$DolphinDir = Split-Path -Parent $DolphinExePath

$UserFolderPath = Resolve-UserFolder -Hint $UserFolderPath -DolphinDir $DolphinDir
if (-not $UserFolderPath) {
    Write-Err "Dolphin User folder not found. Launch Dolphin once or pass -UserFolderPath."
    exit 1
}

$RomFilePath = Resolve-RomFile -Hint $RomFilePath -DolphinDir $DolphinDir
if (-not $RomFilePath) {
    Write-Warn2 "ROM auto-detection failed — Dolphin will start without a ROM."
}

# ---- Step 3 : validate -----------------------------------------------------
$errors = @()
if (-not (Test-Path $DolphinExePath -PathType Leaf))      { $errors += "Dolphin.exe not found: $DolphinExePath" }
if (-not (Test-Path $UserFolderPath -PathType Container)) { $errors += "User folder not found: $UserFolderPath" }
if ($RomFilePath -and -not (Test-Path $RomFilePath -PathType Leaf)) { $errors += "ROM not found: $RomFilePath" }

if ($errors.Count -gt 0) {
    foreach ($e in $errors) { Write-Err "  - $e" }
    exit 1
}

Write-Ok "Dolphin.exe : $DolphinExePath"
Write-Ok "User folder : $UserFolderPath"
Write-Ok "ROM file    : $(if ($RomFilePath) { $RomFilePath } else { '(none)' })"

Save-LauncherConfig -DolphinPath $DolphinExePath -UserFolder $UserFolderPath -RomPath $RomFilePath

# ---- Step 4 : enumerate profiles -------------------------------------------
$profiles = Get-UserProfiles -DolphinDir $DolphinDir -BaseUserFolder $UserFolderPath
if ($profiles.Count -eq 0) {
    Write-Err "No User profiles found in $DolphinDir"
    exit 1
}
Write-Ok "Found $($profiles.Count) profile(s)"

# ---- Step 5 : decide options (NoGUI vs interactive) ------------------------
if ($NoGUI -and $NumInstances -gt 0) {
    $options = @{
        Count           = $NumInstances
        InitialDelay    = $Script:InitialDelaySec
        MinimizeDolphin = [bool]$MinimizeDolphin
        MinimizeGame    = [bool]$MinimizeGame
    }
}
else {
    $options = Show-LauncherDialog -Profiles $profiles
    if ($null -eq $options) { Write-Warn2 "Operation canceled"; exit 0 }
}

# ---- Step 6 : create missing profiles (single unified path) ----------------
if ($options.Count -gt $profiles.Count) {
    Write-Info "Need $($options.Count) profiles, have $($profiles.Count) — creating missing ones"
    $ok = Initialize-UserProfiles -NumInstances $options.Count `
                                  -BaseUserFolder $UserFolderPath `
                                  -DolphinDir $DolphinDir
    if (-not $ok) {
        Write-Err "Failed to create required profiles"
        exit 1
    }
    $profiles = Get-UserProfiles -DolphinDir $DolphinDir -BaseUserFolder $UserFolderPath
    if ($profiles.Count -lt $options.Count) {
        Write-Err "Still missing profiles after auto-create ($($profiles.Count)/$($options.Count))"
        exit 1
    }
}

# ---- Step 7 : write INI files for every instance ---------------------------
Write-Info "Configuring Dolphin INI files..."
for ($i = 0; $i -lt $options.Count; $i++) {
    $name = if ($i -eq 0) { 'User' } else { "User$i" }
    Set-DolphinInstanceConfig -UserFolderPath (Join-Path $DolphinDir $name) -Index $i
}

# ---- Step 8 : launch instances ---------------------------------------------
Write-Info "Launching $($options.Count) instance(s)..."
$instances = New-Object System.Collections.Generic.List[object]

for ($i = 0; $i -lt $options.Count; $i++) {
    Write-Host "  [$($i+1)/$($options.Count)] $($profiles[$i].Name)... " -NoNewline

    $inst = Start-DolphinInstance -Index $i `
                                  -DolphinExe $DolphinExePath `
                                  -DolphinDir $DolphinDir `
                                  -RomPath $RomFilePath
    if ($inst) {
        Write-Host "OK (PID $($inst.Process.Id))" -ForegroundColor Green
        $instances.Add($inst)
        if ($NoGUI) { Write-PidFile -Index $i -ProcessId $inst.Process.Id }
    }
    else {
        Write-Host "FAILED" -ForegroundColor Red
        if ($NoGUI) { Write-PidFile -Index $i -ProcessId -1 }
    }
}

if ($instances.Count -eq 0) {
    Write-Err "No instance launched successfully"
    exit 1
}

# ---- Step 9 : wait + rename windows ----------------------------------------
Write-Info "Waiting $($options.InitialDelay)s for windows to appear..."
Start-Sleep -Seconds $options.InitialDelay

Write-Info "Renaming windows..."
foreach ($inst in $instances) {
    $ok = Rename-DolphinWindow -ProcessId $inst.Process.Id `
                               -NewTitle $inst.Title `
                               -TimeoutSec $Script:WindowTimeoutSec `
                               -MinimizeMenu $options.MinimizeDolphin `
                               -MinimizeGame $options.MinimizeGame
    if ($ok) {
        Write-Ok  "  $($inst.Title) (PID $($inst.Process.Id)) renamed"
    } else {
        Write-Warn2 "  $($inst.Title) (PID $($inst.Process.Id)) rename TIMEOUT"
    }
}

# ---- Step 10 : final cleanup -----------------------------------------------
Write-Info "Waiting 6s for Dolphin init to settle, then cleaning GBA folders..."
Start-Sleep -Seconds 6
Remove-EmptyGbaFolders -DolphinDir $DolphinDir -Count $options.Count
Remove-PhantomDesktopFolders

Write-Ok "Launch complete: $($instances.Count) active instance(s)"

# ---- Step 11 : control panel (GUI) or hand back to Python (NoGUI) ----------
if (-not $NoGUI) {
    Show-ControlPanel -Instances $instances

    Write-Info "Auto-close in 60s (press any key to close now)"
    $start = Get-Date
    while (((Get-Date) - $start).TotalSeconds -lt 60) {
        if ([Console]::KeyAvailable) { [Console]::ReadKey($true) | Out-Null; break }
        Start-Sleep -Milliseconds 100
    }
}
else {
    Write-Ok "NoGUI mode: handing control back to Python"
}