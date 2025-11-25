#Requires -Version 5.1
<#
.SYNOPSIS
    Launch multiple Dolphin instances with automatic window renaming
.DESCRIPTION
    Optimized script to launch and manage multiple Dolphin instances
    
    IMPORTANT - Multi-instance setup:
    This script uses Dolphin's User profiles (User, User1, User2, etc.)
    You must create these profiles BEFORE running the script:
    
    1. Locate your Dolphin User folder:
       - Portable: [Dolphin directory]\User
       - Installed: C:\Users\[YourName]\Documents\Dolphin Emulator
    
    2. Create additional profiles:
       - Copy the "User" folder
       - Rename copies to: User1, User2, User3, etc.
       - Each AI agent will use one profile
    
    3. Number of profiles needed = Number of instances
       Example: 4 instances requires User, User1, User2, User3
#>

[CmdletBinding()]
param(
    [int]$NumInstances = 0,           # number of instances to launch
    [switch]$NoGUI,                   # silent mode (no dialog)
    [switch]$MinimizeDolphin = $false, # minimize Dolphin windows
    [switch]$MinimizeGame = $false,    # minimize game windows
    [string]$DolphinExePath = "",      # Path to Dolphin.exe (optional)
    [string]$UserFolderPath = "",      # Path to User folder (optional)
    [string]$RomFilePath = ""          # Path to ROM file (optional)
)

# Auto-detect paths if not provided
if ([string]::IsNullOrEmpty($DolphinExePath)) {
    # Try to find Dolphin.exe in script directory or parent
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    
    # Check if Dolphin.exe is in same folder as script
    if (Test-Path "$ScriptDir\Dolphin.exe") {
        $DolphinExePath = "$ScriptDir\Dolphin.exe"
    }
    # Check parent folder
    elseif (Test-Path "$ScriptDir\..\Dolphin.exe") {
        $DolphinExePath = Resolve-Path "$ScriptDir\..\Dolphin.exe"
    }
    # Fallback to hardcoded (for backward compatibility)
    else {
        $DolphinExePath = "C:\Users\rocca\Desktop\MonsterHunter\Emulateur\IA_jeux\Dolphin-x64\Dolphin.exe"
    }
}

if ([string]::IsNullOrEmpty($UserFolderPath)) {
    # Normalize DolphinExePath first (critical for Split-Path)
    $DolphinExePath = [System.IO.Path]::GetFullPath($DolphinExePath)
    
    # Get directory from executable path
    $DolphinDir = Split-Path -Parent $DolphinExePath
    
    # Normalize path (remove trailing slashes)
    $DolphinDir = $DolphinDir.TrimEnd('\')
    
    # Debug output
    Write-Host "DEBUG: DolphinExePath resolved to: $DolphinExePath" -ForegroundColor Cyan
    Write-Host "DEBUG: DolphinDir extracted as: $DolphinDir" -ForegroundColor Cyan
    
    # Check for portable mode (User folder in Dolphin directory)
    $PortableUserPath = Join-Path $DolphinDir "User"
    Write-Host "DEBUG: Testing portable path: $PortableUserPath" -ForegroundColor Cyan
    
    if (Test-Path $PortableUserPath -PathType Container) {
        $UserFolderPath = $PortableUserPath
        Write-Host "Detected portable Dolphin User folder: $UserFolderPath" -ForegroundColor Green
    }
    # Check for standard install (AppData)
    else {
        $AppDataUser = Join-Path $env:USERPROFILE "Documents\Dolphin Emulator"
        if (Test-Path $AppDataUser -PathType Container) {
            $UserFolderPath = $AppDataUser
            Write-Host "Detected installed Dolphin User folder: $UserFolderPath" -ForegroundColor Green
        }
        else {
            # CRITICAL ERROR : No valid User folder found
            Write-Host "ERROR: Cannot find Dolphin User folder" -ForegroundColor Red
            Write-Host "  Checked locations:" -ForegroundColor Yellow
            Write-Host "    - Portable: $PortableUserPath" -ForegroundColor Yellow
            Write-Host "    - AppData: $AppDataUser" -ForegroundColor Yellow
            Write-Host "" -ForegroundColor Yellow
            Write-Host "SOLUTION:" -ForegroundColor Yellow
            Write-Host "  1. Launch Dolphin at least once to create User folder" -ForegroundColor Yellow
            Write-Host "  2. Or provide explicit path:" -ForegroundColor Yellow
            Write-Host "     -UserFolderPath 'C:\Path\To\Dolphin\User'" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Additional validation
if (-not [string]::IsNullOrEmpty($UserFolderPath)) {
    # Normalize the detected path
    $UserFolderPath = $UserFolderPath.TrimEnd('\')
    
    # Verify it actually exists
    if (-not (Test-Path $UserFolderPath -PathType Container)) {
        Write-Host "ERROR: User folder does not exist: $UserFolderPath" -ForegroundColor Red
        Write-Host "SOLUTION:" -ForegroundColor Yellow
        Write-Host "  1. Launch Dolphin to create the User folder" -ForegroundColor Yellow
        Write-Host "  2. Or manually create User profiles (User, User1, User2...)" -ForegroundColor Yellow
        exit 1  # ✅ FAIL INSTEAD OF AUTO-CREATE
    }
}

if ([string]::IsNullOrEmpty($RomFilePath)) {
    # Try to find ROM in common locations
    $PossibleRomPaths = @(
        "C:\Users\rocca\Desktop\MonsterHunter\Emulateur\Jeux\MHtri\MonsterHunterTri.rvz",
        "$UserFolderPath\..\Jeux\MHtri\MonsterHunterTri.rvz",
        "$UserFolderPath\Games\MonsterHunterTri.rvz"
    )
    
    foreach ($Path in $PossibleRomPaths) {
        if (Test-Path $Path) {
            $RomFilePath = $Path
            break
        }
    }
}

# Configuration
$Config = @{
    DolphinPath = $DolphinExePath
    UserFolder  = $UserFolderPath
    RomPath     = $RomFilePath
    InitialDelay = 2
    WindowTimeout = 5
}

# ==============================================================================
# DEBUG: Verify variables before validation
# ==============================================================================
Write-Host ""
Write-Host "DEBUG: Pre-validation check" -ForegroundColor Magenta
Write-Host "  DolphinExePath  = '$DolphinExePath'" -ForegroundColor Gray
Write-Host "  UserFolderPath  = '$UserFolderPath'" -ForegroundColor Gray
Write-Host "  RomFilePath     = '$RomFilePath'" -ForegroundColor Gray
Write-Host "  Config.DolphinPath = '$($Config.DolphinPath)'" -ForegroundColor Gray
Write-Host "  Config.UserFolder  = '$($Config.UserFolder)'" -ForegroundColor Gray
Write-Host "  Config.RomPath     = '$($Config.RomPath)'" -ForegroundColor Gray
Write-Host ""

# ==============================================================================
# PATH VALIDATION
# ==============================================================================
Write-Host "==================================================================" -ForegroundColor Cyan

# ==============================================================================
# CONFIGURATION PERSISTENCE (load saved config)
# ==============================================================================
# Save validated paths to local config file for future runs
# Config file location: same directory as script

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ConfigFile = Join-Path $ScriptDir "dolphin_launcher_config.json"

function Save-LauncherConfig {
    param(
        [string]$DolphinPath,
        [string]$UserFolder,
        [string]$RomPath
    )
    
    $configData = @{
        DolphinExePath = $DolphinPath
        UserFolderPath = $UserFolder
        RomFilePath = $RomPath
        LastUpdated = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    }
    
    try {
        $configData | ConvertTo-Json | Set-Content -Path $ConfigFile -Encoding UTF8
        Write-Host "Configuration saved to: $ConfigFile" -ForegroundColor Green
    }
    catch {
        Write-Warning "Failed to save configuration: $_"
    }
}

# ==============================================================================
# PATH VALIDATION
# ==============================================================================
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "CONFIGURATION VALIDATION" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Dolphin.exe: $($Config.DolphinPath)" -ForegroundColor White
Write-Host "User Folder: $($Config.UserFolder)" -ForegroundColor White
Write-Host "ROM File   : $($Config.RomPath)" -ForegroundColor White
Write-Host ""

$ValidationErrors = @()

if (-not (Test-Path $Config.DolphinPath)) {
    $ValidationErrors += "Dolphin.exe NOT FOUND: $($Config.DolphinPath)"
}

if (-not (Test-Path $Config.UserFolder -PathType Container)) {
    $ValidationErrors += "User Folder NOT FOUND: $($Config.UserFolder)"
}

if (-not (Test-Path $Config.RomPath)) {
    $ValidationErrors += "ROM File NOT FOUND: $($Config.RomPath)"
}

if ($ValidationErrors.Count -gt 0) {
    Write-Host "VALIDATION ERRORS:" -ForegroundColor Red
    foreach ($ErrorMsg in $ValidationErrors) {
        Write-Host "  - $ErrorMsg" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "SOLUTIONS:" -ForegroundColor Yellow
    Write-Host "  1. Verify paths are correct" -ForegroundColor Yellow
    Write-Host "  2. Launch Dolphin at least once to create User folder" -ForegroundColor Yellow
    Write-Host "  3. Provide paths explicitly:" -ForegroundColor Yellow
    Write-Host "     .\launch_dolphin_instances.ps1 -DolphinExePath 'C:\Path\To\Dolphin.exe' \" -ForegroundColor Yellow
    Write-Host "                                     -UserFolderPath 'C:\Path\To\UserFolder' \" -ForegroundColor Yellow
    Write-Host "                                     -RomFilePath 'C:\Path\To\ROM.rvz'" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Cyan
    exit 1
}

Write-Host "All paths validated successfully!" -ForegroundColor Green

# Save configuration for next run
Save-LauncherConfig -DolphinPath $Config.DolphinPath `
                    -UserFolder $Config.UserFolder `
                    -RomPath $Config.RomPath

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

function Get-LauncherConfig {
    if (Test-Path $ConfigFile) {
        try {
            $config = Get-Content -Path $ConfigFile -Raw | ConvertFrom-Json
            Write-Host "Loaded saved configuration from: $ConfigFile" -ForegroundColor Cyan
            return $config
        }
        catch {
            Write-Warning "Failed to load configuration: $_"
            return $null
        }
    }
    return $null
}

# Try to load saved configuration if parameters not provided
if ([string]::IsNullOrEmpty($DolphinExePath) -or 
    [string]::IsNullOrEmpty($UserFolderPath) -or 
    [string]::IsNullOrEmpty($RomFilePath)) {
    
    $savedConfig = Get-LauncherConfig
    
    if ($null -ne $savedConfig) {
        if ([string]::IsNullOrEmpty($DolphinExePath)) { 
            $DolphinExePath = $savedConfig.DolphinExePath 
            Write-Host "  Using saved DolphinExePath" -ForegroundColor Gray
        }
        if ([string]::IsNullOrEmpty($UserFolderPath)) { 
            $UserFolderPath = $savedConfig.UserFolderPath 
            Write-Host "  Using saved UserFolderPath" -ForegroundColor Gray
        }
        if ([string]::IsNullOrEmpty($RomFilePath)) { 
            $RomFilePath = $savedConfig.RomFilePath 
            Write-Host "  Using saved RomFilePath" -ForegroundColor Gray
        }
        Write-Host ""
    }
}

# Charger assemblies
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# Definir WinAPI avec methodes utilitaires
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
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

        public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);
        
        // Classe pour stocker les infos de fenetre
        public class WindowInfo {
            public IntPtr Handle;
            public string Title;
        }
        
        // Methode pour obtenir toutes les fenetres d'un processus
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

#region Functions

function Test-Prerequisites {
    $errors = @()
    
    # Validate Dolphin.exe exists (file)
    if (-not (Test-Path $Config.DolphinPath -PathType Leaf)) {
        $errors += "Dolphin introuvable : $($Config.DolphinPath)"
    }
    
    # Validate User folder exists (directory)
    if (-not (Test-Path $Config.UserFolder -PathType Container)) {
        $errors += "Dossier utilisateurs introuvable : $($Config.UserFolder)"
    }
    
    # Validate ROM file exists (file)
    if (-not (Test-Path $Config.RomPath -PathType Leaf)) {
        $errors += "ROM introuvable : $($Config.RomPath)"
    }
    
    if ($errors.Count -gt 0) {
    $errorMessage = "Configuration errors detected:`n`n" + ($errors -join "`n`n")
    $errorMessage += "`n`nDEBUG INFO:`n"
    $errorMessage += "DolphinPath exists: $(Test-Path $Config.DolphinPath -PathType Leaf)`n"
    $errorMessage += "UserFolder exists: $(Test-Path $Config.UserFolder -PathType Container)`n"
    $errorMessage += "RomPath exists: $(Test-Path $Config.RomPath -PathType Leaf)`n"
    
    [System.Windows.Forms.MessageBox]::Show(
        $errorMessage,
        "Configuration Error",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    )
    return $false
}   
    return $true
}

function Get-UserProfiles {
    # Check if User folders exist in Dolphin directory
    # Expected format: User, User1, User2, User3...
    # IMPORTANT: We search in the PARENT directory of $Config.UserFolder
    # Because User, User1, User2 are SIBLINGS, not children of User folder
    
    # Get parent directory (Dolphin-x64 folder)
    $dolphinDir = Split-Path -Parent $Config.UserFolder
    
    Write-Host "Searching for User profiles in: $dolphinDir" -ForegroundColor Cyan
    
    $profiles = Get-ChildItem -Path $dolphinDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^User(\d*)$' } |
        ForEach-Object {
            $index = if ($_.Name -eq 'User') { 0 } else { [int]($_.Name -replace 'User','') }
            [PSCustomObject]@{
                Name = $_.Name
                Index = $index
                Path = $_.FullName
            }
        } |
        Sort-Object Index
    
    if ($profiles) {
        Write-Host "Found profiles:" -ForegroundColor Green
        foreach ($p in $profiles) {
            Write-Host "  - $($p.Name) (Index: $($p.Index))" -ForegroundColor Gray
        }
    }
    else {
        Write-Host "WARNING: No User profiles found!" -ForegroundColor Yellow
        Write-Host "Expected folders like: User, User1, User2..." -ForegroundColor Yellow
    }
    Write-Host ""
    
    return $profiles
}

function Show-LauncherDialog {
    param([array]$Profiles)
    
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Lanceur Dolphin Multi-Instance"
    $form.Size = New-Object System.Drawing.Size(450, 270)
    $form.StartPosition = "CenterScreen"
    $form.FormBorderStyle = "FixedDialog"
    $form.MaximizeBox = $false
    
    # Label instances
    $lblCount = New-Object System.Windows.Forms.Label
    $lblCount.Text = "Nombre d'instances (1-$($Profiles.Count)) :"
    $lblCount.Location = New-Object System.Drawing.Point(20, 20)
    $lblCount.Size = New-Object System.Drawing.Size(250, 20)
    $form.Controls.Add($lblCount)
    
    # NumericUpDown instances
    $numCount = New-Object System.Windows.Forms.NumericUpDown
    $numCount.Minimum = 1
    $numCount.Maximum = $Profiles.Count
    $numCount.Value = [Math]::Min(3, $Profiles.Count)
    $numCount.Location = New-Object System.Drawing.Point(280, 18)
    $numCount.Size = New-Object System.Drawing.Size(120, 20)
    $form.Controls.Add($numCount)
    
    # Label delai
    $lblDelay = New-Object System.Windows.Forms.Label
    $lblDelay.Text = "Delai initial avant renommage (sec) :"
    $lblDelay.Location = New-Object System.Drawing.Point(20, 55)
    $lblDelay.Size = New-Object System.Drawing.Size(250, 20)
    $form.Controls.Add($lblDelay)
    
    # NumericUpDown delai
    $numDelay = New-Object System.Windows.Forms.NumericUpDown
    $numDelay.Minimum = 0
    $numDelay.Maximum = 30
    $numDelay.Value = $Config.InitialDelay
    $numDelay.Location = New-Object System.Drawing.Point(280, 53)
    $numDelay.Size = New-Object System.Drawing.Size(120, 20)
    $form.Controls.Add($numDelay)
    
    # Checkbox copie config
    $chkCopy = New-Object System.Windows.Forms.CheckBox
    $chkCopy.Text = "Copier la configuration du 1er profil vers les autres"
    $chkCopy.Location = New-Object System.Drawing.Point(20, 90)
    $chkCopy.Size = New-Object System.Drawing.Size(380, 20)
    $form.Controls.Add($chkCopy)
    
    # Checkbox reduire fenetre Dolphin
    $chkMinimizeDolphin = New-Object System.Windows.Forms.CheckBox
    $chkMinimizeDolphin.Text = "Reduire les fenetres Dolphin (menu)"
    $chkMinimizeDolphin.Location = New-Object System.Drawing.Point(20, 115)
    $chkMinimizeDolphin.Size = New-Object System.Drawing.Size(280, 20)
    $chkMinimizeDolphin.Checked = $true
    $form.Controls.Add($chkMinimizeDolphin)
    
    # Checkbox reduire fenetre de jeu
    $chkMinimizeGame = New-Object System.Windows.Forms.CheckBox
    $chkMinimizeGame.Text = "Reduire les fenetres de jeu"
    $chkMinimizeGame.Location = New-Object System.Drawing.Point(20, 140)
    $chkMinimizeGame.Size = New-Object System.Drawing.Size(280, 20)
    $chkMinimizeGame.Checked = $false
    $form.Controls.Add($chkMinimizeGame)
    
    # Bouton Lancer
    $btnOk = New-Object System.Windows.Forms.Button
    $btnOk.Text = "Lancer"
    $btnOk.Location = New-Object System.Drawing.Point(150, 175)
    $btnOk.Size = New-Object System.Drawing.Size(100, 30)
    $btnOk.DialogResult = [System.Windows.Forms.DialogResult]::OK
    $form.Controls.Add($btnOk)
    $form.AcceptButton = $btnOk
    
    # Bouton Annuler
    $btnCancel = New-Object System.Windows.Forms.Button
    $btnCancel.Text = "Annuler"
    $btnCancel.Location = New-Object System.Drawing.Point(260, 175)
    $btnCancel.Size = New-Object System.Drawing.Size(100, 30)
    $btnCancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
    $form.Controls.Add($btnCancel)
    $form.CancelButton = $btnCancel
    
    $result = $form.ShowDialog()
    
    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
        return @{
            Count = [int]$numCount.Value
            InitialDelay = [int]$numDelay.Value
            CopyConfig = $chkCopy.Checked
            MinimizeDolphin = $chkMinimizeDolphin.Checked
            MinimizeGame = $chkMinimizeGame.Checked
        }
    }
    return $null
}

function Copy-ConfigToProfiles {
    param([array]$Profiles)
    
    $sourceConfig = Join-Path $Profiles[0].Path "Config"
    if (-not (Test-Path $sourceConfig)) {
        Write-Warning "Configuration source introuvable : $sourceConfig"
        return
    }
    
    for ($i = 1; $i -lt $Profiles.Count; $i++) {
        $destConfig = Join-Path $Profiles[$i].Path "Config"
        if (-not (Test-Path $destConfig)) {
            try {
                Copy-Item -Path $sourceConfig -Destination $destConfig -Recurse -Force -ErrorAction Stop
                Write-Host "  Config copiee vers $($Profiles[$i].Name)" -ForegroundColor Green
            }
            catch {
                Write-Warning "Echec copie config vers $($Profiles[$i].Name) : $_"
            }
        }
    }
}

function Initialize-UserProfiles {
    param(
        [int]$NumInstances,
        [string]$BaseUserFolder
    )
    
    Write-Host "Checking User profiles..." -ForegroundColor Cyan
    
    $DolphinDir = Split-Path -Parent $BaseUserFolder
    
    # Verify base User folder exists
    if (-not (Test-Path $BaseUserFolder -PathType Container)) {
        Write-Error "Base User folder not found: $BaseUserFolder"
        Write-Error "Launch Dolphin at least once to create it"
        return $false
    }
    
    # Create User1, User2, ... if missing
    for ($i = 1; $i -lt $NumInstances; $i++) {
        $TargetFolder = Join-Path $DolphinDir "User$i"
        
        if (-not (Test-Path $TargetFolder -PathType Container)) {
            Write-Host "  Creating User$($i)..." -ForegroundColor Yellow
            
            try {
                Copy-Item -Path $BaseUserFolder -Destination $TargetFolder -Recurse -Force
                Write-Host "  User$($i) created successfully" -ForegroundColor Green
            }
            catch {
                Write-Error "  Failed to create User$($i): $($_)"
                return $false
            }
        }
        else {
            Write-Host "  User$($i) already exists" -ForegroundColor Gray
        }
    }
    
    return $true
}

function Start-DolphinInstance {
    param(
        [string]$UserProfile,
        [int]$Index
    )
    
    try {
        # Build absolute path to User folder
        $UserFolderName = if ($Index -eq 0) { "User" } else { "User$Index" }
        $DolphinDir = Split-Path -Parent $Config.DolphinPath
        $UserFolderAbsolutePath = Join-Path $DolphinDir $UserFolderName

        # CRITICAL: Convert to absolute path and verify
        $UserFolderAbsolutePath = [System.IO.Path]::GetFullPath($UserFolderAbsolutePath)

        Write-Host "  DEBUG: Using User folder: $UserFolderAbsolutePath" -ForegroundColor Cyan

        # Verify folder exists BEFORE launching
        if (-not (Test-Path $UserFolderAbsolutePath -PathType Container)) {
            Write-Warning "  User folder not found: $UserFolderAbsolutePath"
            Write-Warning "  Dolphin will create a NEW user folder instead of loading existing config"
            Write-Warning "  SOLUTION: Create User profiles manually or copy from User folder"
        }

        # IMPORTANT: Set WorkingDirectory to Dolphin folder to avoid relative path issues
        $process = Start-Process -FilePath $Config.DolphinPath `
                 -ArgumentList @('--user', $UserFolderAbsolutePath, $Config.RomPath) `
                 -WorkingDirectory $DolphinDir `
                 -PassThru `
                 -ErrorAction Stop
        
        # Title format: MHTri-0, MHTri-1, MHTri-2...
        # Matches Index (0-based) for consistency with Python
        return @{
            Process = $process
            Index = $Index
            UserProfile = $UserProfile
            Title = "MHTri-$Index"
        }
    }
    catch {
        Write-Warning "Erreur lancement instance $Index ($UserProfile) : $_"
        return $null
    }
}

function Find-AndRenameWindow {
    param(
        [int]$ProcessId,
        [string]$SearchTerm,
        [string]$NewTitle,
        [int]$TimeoutSeconds,
        [bool]$MinimizeDolphin,
        [bool]$MinimizeGame
    )
    
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $renamed = $false
    
    Write-Host "    Recherche fenetre pour PID $ProcessId..." -ForegroundColor Gray
    
    while ((Get-Date) -lt $deadline -and -not $renamed) {
        try {
            # Utiliser la methode C# pour recuperer les fenetres
            $windows = [WindowManager]::GetProcessWindows($ProcessId)
            
            if ($windows.Count -gt 0) {
                Write-Host "    Trouve $($windows.Count) fenetre(s)" -ForegroundColor Cyan
                
                # Reduire les fenetres "Dolphin 2509" (menu) si demande
                if ($MinimizeDolphin) {
                    foreach ($win in $windows) {
                        if ($win.Title -eq "Dolphin 2509") {
                            [WindowManager]::ShowWindow($win.Handle, 6) | Out-Null
                            Write-Host "    Fenetre Dolphin reduite" -ForegroundColor DarkGray
                        }
                    }
                }
                
                # Chercher et renommer la fenetre de jeu
                foreach ($win in $windows) {
                    # Chercher "Monster Hunter Tri" ou "RMHP" dans le titre
                    if ($win.Title -like "*Monster Hunter Tri*" -or $win.Title -like "*RMHP*") {
                        Write-Host "    Fenetre correspondante : '$($win.Title)'" -ForegroundColor Yellow
                        
                        # Tenter le renommage
                        $result = [WindowManager]::SetWindowText($win.Handle, $NewTitle)
                        
                        if ($result) {
                            Write-Host "    SUCCES : Renomme en '$NewTitle'" -ForegroundColor Green
                            
                            # Reduire la fenetre de jeu si demande
                            if ($MinimizeGame) {
                                [WindowManager]::ShowWindow($win.Handle, 6) | Out-Null
                                Write-Host "    Fenetre de jeu reduite" -ForegroundColor DarkGray
                            }
                            
                            $renamed = $true
                            break
                        }
                        else {
                            Write-Warning "    SetWindowText a echoue"
                        }
                    }
                }
            }
        }
        catch {
            Write-Warning "    Erreur : $_"
        }
        
        if (-not $renamed) {
            Start-Sleep -Milliseconds 250
        }
    }
    
    return $renamed
}

function Show-ControlPanel {
    param([array]$Instances)
    
    # Close instances window shape
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Gestionnaire d'instances Dolphin"
    $form.Size = New-Object System.Drawing.Size(550, 350)
    $form.StartPosition = "CenterScreen"
    $form.TopMost = $true  # Force window to front
    
    # Label
    $lblInfo = New-Object System.Windows.Forms.Label
    $lblInfo.Text = "Instances actives (selectionner pour fermer) :"
    $lblInfo.Location = New-Object System.Drawing.Point(20, 15)
    $lblInfo.Size = New-Object System.Drawing.Size(500, 20)
    $form.Controls.Add($lblInfo)
    
    # ListBox
    $listBox = New-Object System.Windows.Forms.ListBox
    $listBox.Location = New-Object System.Drawing.Point(20, 40)
    $listBox.Size = New-Object System.Drawing.Size(500, 200)
    $listBox.SelectionMode = "MultiExtended"
    $form.Controls.Add($listBox)
    
    # Remplir la liste
    foreach ($inst in $Instances) {
        $status = if ($inst.Process.HasExited) { "[FERMEE]" } else { "[ACTIVE]" }
        $text = "$status $($inst.Title) - PID:$($inst.Process.Id) - $($inst.UserProfile)"
        $listBox.Items.Add($text) | Out-Null
    }
    
    # Bouton Fermer selection
    $btnClose = New-Object System.Windows.Forms.Button
    $btnClose.Text = "Fermer selection"
    $btnClose.Location = New-Object System.Drawing.Point(20, 260)
    $btnClose.Size = New-Object System.Drawing.Size(150, 35)
    $btnClose.Add_Click({
        $selected = $listBox.SelectedIndices
        if ($selected.Count -eq 0) { return }
        
        foreach ($index in $selected) {
            $inst = $Instances[$index]
            try {
                if (-not $inst.Process.HasExited) {
                    $inst.Process.CloseMainWindow() | Out-Null
                    Start-Sleep -Milliseconds 300
                    if (-not $inst.Process.HasExited) {
                        $inst.Process.Kill()
                    }
                }
            }
            catch {
                Write-Warning "Erreur fermeture PID $($inst.Process.Id) : $_"
            }
        }
        
        # Rafraichir la liste
        $listBox.Items.Clear()
        foreach ($inst in $Instances) {
            $status = if ($inst.Process.HasExited) { "[FERMEE]" } else { "[ACTIVE]" }
            $text = "$status $($inst.Title) - PID:$($inst.Process.Id) - $($inst.UserProfile)"
            $listBox.Items.Add($text) | Out-Null
        }
    })
    $form.Controls.Add($btnClose)
    
    # Bouton Tout fermer
    $btnCloseAll = New-Object System.Windows.Forms.Button
    $btnCloseAll.Text = "Tout fermer"
    $btnCloseAll.Location = New-Object System.Drawing.Point(190, 260)
    $btnCloseAll.Size = New-Object System.Drawing.Size(150, 35)
    $btnCloseAll.Add_Click({
        foreach ($inst in $Instances) {
            try {
                if (-not $inst.Process.HasExited) {
                    $inst.Process.CloseMainWindow() | Out-Null
                    Start-Sleep -Milliseconds 300
                    if (-not $inst.Process.HasExited) {
                        $inst.Process.Kill()
                    }
                }
            }
            catch {}
        }
        $form.Close()
    })
    $form.Controls.Add($btnCloseAll)
    
    # Bouton Quitter
    $btnQuit = New-Object System.Windows.Forms.Button
    $btnQuit.Text = "Quitter (laisser ouvert)"
    $btnQuit.Location = New-Object System.Drawing.Point(360, 260)
    $btnQuit.Size = New-Object System.Drawing.Size(160, 35)
    $btnQuit.Add_Click({ $form.Close() })
    $form.Controls.Add($btnQuit)
    
    # Bring window to foreground
    Add-Type @"
        using System;
        using System.Runtime.InteropServices;
        public class WindowHelper {
            [DllImport("user32.dll")]
            [return: MarshalAs(UnmanagedType.Bool)]
            public static extern bool SetForegroundWindow(IntPtr hWnd);
            
            [DllImport("user32.dll")]
            public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
        }
"@
    
    $form.Add_Shown({
        $handle = $form.Handle
        [WindowHelper]::ShowWindow($handle, 5)  # SW_SHOW
        [WindowHelper]::SetForegroundWindow($handle)
        $form.Activate()
    })
    
    $form.ShowDialog() | Out-Null
}

#endregion

#region Main

Write-Host "=== Lanceur Multi-Instance Dolphin ===" -ForegroundColor Cyan
Write-Host ""

# Verifications
if (-not (Test-Prerequisites)) {
    exit 1
}

# Recuperer les profils
$profiles = Get-UserProfiles
if ($profiles.Count -eq 0) {
    [System.Windows.Forms.MessageBox]::Show(
        "Aucun profil utilisateur trouve dans :`n$($Config.UserFolder)",
        "Erreur",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Error
    )
    exit 1
}

Write-Host "Profils detectes : $($profiles.Count)" -ForegroundColor Green

# MODE AUTOMATIQUE (appelé depuis Python)
if ($NumInstances -gt 0 -and $NoGUI) {
    Write-Host "Mode automatique : $NumInstances instances" -ForegroundColor Cyan
    
    $options = @{
        Count = [Math]::Min($NumInstances, $profiles.Count)
        InitialDelay = 2
        CopyConfig = $false
        MinimizeDolphin = $MinimizeDolphin
        MinimizeGame = $MinimizeGame
    }
}
# MODE INTERACTIF (manuel)
else {
    # Afficher dialogue
    $options = Show-LauncherDialog -Profiles $profiles
    if ($null -eq $options) {
        Write-Host "Operation annulee" -ForegroundColor Yellow
        exit 0
    }
}

# Copier config si demande
if ($options.CopyConfig) {
    Write-Host "`nCopie de la configuration..." -ForegroundColor Cyan
    Copy-ConfigToProfiles -Profiles $profiles
}

# BEFORE launching instances, ensure all User profiles exist
if ($profiles.Count -lt $options.Count) {
    Write-Host "Not enough User profiles detected" -ForegroundColor Yellow
    Write-Host "  Requested: $($options.Count) instances" -ForegroundColor Yellow
    Write-Host "  Available: $($profiles.Count) profiles" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Auto-creating missing profiles..." -ForegroundColor Cyan
    
    $ProfilesReady = Initialize-UserProfiles -NumInstances $options.Count -BaseUserFolder $profiles[0].Path
    
    if (-not $ProfilesReady) {
        Write-Error "Profile creation failed"
        Write-Error "Create User1, User2, ... folders manually in Dolphin directory"
        exit 1
    }
    
    # Re-scan profiles after creation
    $profiles = Get-UserProfiles
    Write-Host "Profiles after creation: $($profiles.Count)" -ForegroundColor Green
}

# Copier config si demande
if ($options.CopyConfig) {
    Write-Host "`nCopie de la configuration..." -ForegroundColor Cyan
    Copy-ConfigToProfiles -Profiles $profiles
}

# Launch instances quickly
Write-Host "`nQuick launch of instances..." -ForegroundColor Cyan
$instances = @()

for ($i = 0; $i -lt $options.Count; $i++) {
    # Determine which User profile to use
    # Index 0 uses "User", others use "User1", "User2", etc.
    # Profiles must already exist (created by Dolphin or manually)
    
    if ($i -ge $profiles.Count) {
        Write-Host "  [$($i+1)/$($options.Count)] ERROR: Not enough User profiles!" -ForegroundColor Red
        Write-Host "    Requested: $($options.Count) profiles" -ForegroundColor Yellow
        Write-Host "    Available: $($profiles.Count) profiles" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "SOLUTION: Create additional User profiles manually:" -ForegroundColor Yellow
        Write-Host "  1. Copy the 'User' folder in Dolphin directory" -ForegroundColor Yellow
        Write-Host "  2. Rename copies to: User1, User2, User3..." -ForegroundColor Yellow
        Write-Host "  3. Run script again" -ForegroundColor Yellow
        continue
    }
    
    $userProfile = $profiles[$i]
    Write-Host "  [$($i+1)/$($options.Count)] Launching with $($userProfile.Name)..." -NoNewline
    
    $instance = Start-DolphinInstance -UserProfile $userProfile.Name -Index ($i)
    
    if ($null -ne $instance) {
        $instances += $instance
        Write-Host " OK (PID: $($instance.Process.Id))" -ForegroundColor Green

        # Write PID IMMEDIATELY (not at the end)
        if ($NoGUI) {
            $pidFile = "dolphin_pid_$($i).tmp"
            
            try {
                # Write PID with explicit encoding and force flush
                $instance.Process.Id | Out-File -FilePath $pidFile -Encoding ASCII -Force
                
                # Verify file was created
                if (Test-Path $pidFile) {
                    Write-Host "  PID file created: $pidFile" -ForegroundColor DarkGray
                }
                else {
                    Write-Host "  WARNING: PID file NOT created: $pidFile" -ForegroundColor Yellow
                }
            }
            catch {
                Write-Host "  ERROR writing PID file: $_" -ForegroundColor Red
            }
        }
    }
    else {
        Write-Host " FAILED" -ForegroundColor Red
        
        # Create empty PID file to signal failure
        if ($NoGUI) {
            $pidFile = "dolphin_pid_$($i).tmp"
            "-1" | Out-File -FilePath $pidFile -Encoding ASCII -Force
        }
    }
}


if ($instances.Count -eq 0) {
    Write-Host "`nAucune instance lancee avec succes" -ForegroundColor Red
    exit 1
}

# Attendre que les fenetres se creent
Write-Host "`nAttente de $($options.InitialDelay) secondes pour la creation des fenetres..." -ForegroundColor Cyan
Start-Sleep -Seconds $options.InitialDelay

# Renommer les fenetres
Write-Host "`nRenommage des fenetres..." -ForegroundColor Cyan
$searchTerm = [System.IO.Path]::GetFileNameWithoutExtension($Config.RomPath)

foreach ($instance in $instances) {
    Write-Host "`n  Instance $($instance.Index) (PID: $($instance.Process.Id)):" -ForegroundColor Yellow
    
    $success = Find-AndRenameWindow -ProcessId $instance.Process.Id `
                                     -SearchTerm $searchTerm `
                                     -NewTitle $instance.Title `
                                     -TimeoutSeconds $Config.WindowTimeout `
                                     -MinimizeDolphin $options.MinimizeDolphin `
                                     -MinimizeGame $options.MinimizeGame
    
    if (-not $success) {
        Write-Host "    TIMEOUT - Fenetre non trouvee ou non renommee" -ForegroundColor Red
    }
}

Write-Host "`nLancement termine : $($instances.Count) instance(s) active(s)" -ForegroundColor Green
Write-Host ""

# MODE INTERACTIF : Afficher panneau de controle
if (-not $NoGUI) {
    Show-ControlPanel -Instances $instances
    Write-Host "`nScript termine" -ForegroundColor Cyan
}
# MODE AUTOMATIQUE : Pas de panneau, Python prend le relais
else {
    Write-Host "Mode automatique : Python reprend le controle" -ForegroundColor Green
}

#endregion