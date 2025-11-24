#Requires -Version 5.1
<#
.SYNOPSIS
    Lance plusieurs instances de Dolphin avec renommage automatique des fenetres
.DESCRIPTION
    Script optimise pour lancer et gerer plusieurs instances Dolphin
#>

[CmdletBinding()]
param(
    [int]$NumInstances = 0,           # nombre d'instances à lancer
    [switch]$NoGUI,                    # mode silencieux (pas de dialogue)
    [switch]$MinimizeDolphin = $false, # réduire fenêtres Dolphin
    [switch]$MinimizeGame = $false   # réduire fenêtres jeu
)

# Configuration
$Config = @{
    DolphinPath = "C:\Users\rocca\Desktop\MonsterHunter\Emulateur\IA_jeux\Dolphin-x64\Dolphin.exe"
    UserFolder  = "C:\Users\rocca\Desktop\MonsterHunter\Emulateur\IA_jeux\Dolphin-x64"
    RomPath     = "C:\Users\rocca\Desktop\MonsterHunter\Emulateur\Jeux\MHtri\MonsterHunterTri.rvz"
    InitialDelay = 2
    WindowTimeout = 5
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
    
    if (-not (Test-Path $Config.DolphinPath)) {
        $errors += "Dolphin introuvable : $($Config.DolphinPath)"
    }
    if (-not (Test-Path $Config.UserFolder)) {
        $errors += "Dossier utilisateurs introuvable : $($Config.UserFolder)"
    }
    if (-not (Test-Path $Config.RomPath)) {
        $errors += "ROM introuvable : $($Config.RomPath)"
    }
    
    if ($errors.Count -gt 0) {
        [System.Windows.Forms.MessageBox]::Show(
            ($errors -join "`n"),
            "Erreur de configuration",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        )
        return $false
    }
    return $true
}

function Get-UserProfiles {
    $profiles = Get-ChildItem -Path $Config.UserFolder -Directory -ErrorAction SilentlyContinue |
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

function Start-DolphinInstance {
    param(
        [string]$UserProfile,
        [int]$Index
    )
    
    try {
        $process = Start-Process -FilePath $Config.DolphinPath `
                                 -ArgumentList @('--user', $UserProfile, $Config.RomPath) `
                                 -PassThru `
                                 -ErrorAction Stop
        
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
    
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Gestionnaire d'instances Dolphin"
    $form.Size = New-Object System.Drawing.Size(550, 350)
    $form.StartPosition = "CenterScreen"
    
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

# Lancer les instances RAPIDEMENT
Write-Host "`nLancement rapide des instances..." -ForegroundColor Cyan
$instances = @()

for ($i = 0; $i -lt $options.Count; $i++) {
    $profile = $profiles[$i]
    Write-Host "  [$($i+1)/$($options.Count)] Lancement $($profile.Name)..." -NoNewline
    
    $instance = Start-DolphinInstance -UserProfile $profile.Name -Index ($i + 1)
    
    if ($null -ne $instance) {
        $instances += $instance
        Write-Host " OK (PID: $($instance.Process.Id))" -ForegroundColor Green

         # Écrire le PID dans un fichier pour Python
        if ($NoGUI) {
            $instance.Process.Id | Out-File -FilePath "dolphin_pid_$($i).tmp" -Encoding ASCII
        }
    }
    else {
        Write-Host " ECHEC" -ForegroundColor Red
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

    # Écrire les PIDs dans des fichiers
    foreach ($instance in $instances) {
        $pidFile = "dolphin_pid_$($instance.Index - 1).tmp"
        $instance.Process.Id | Out-File -FilePath $pidFile -Encoding ASCII
        Write-Host "PID écrit : $pidFile → $($instance.Process.Id)" -ForegroundColor Cyan
    }
}

#endregion