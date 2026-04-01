"""
Automatic DLL builder for DolphinCapture
Compiles C++ code to DLL using available compiler

NOTE: Both DolphinCapture.dll (C++) and dolphin_input_hook.dll (Rust) are
already pre-compiled and included in the repository under vision/.
You only need this script if the DLL was accidentally deleted or you want
to rebuild from source.

- DolphinCapture.dll  → this script (requires Visual Studio C++ desktop tools)
- dolphin_input_hook.dll → cd hook && cargo build --release (requires cargo/Rust)
  See: https://rustup.rs
"""
import subprocess
import sys
from pathlib import Path
import shutil


def find_visual_studio():
    """Find Visual Studio installation (including Build Tools and Insiders)"""
    print("\n🔍 Searching for Visual Studio or Build Tools...")

    # Standard VS installations
    possible_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
        r"C:\Program Files\Microsoft Visual Studio\2022\Preview",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
    ]

    # Check standard paths first
    for path in possible_paths:
        vcvars = Path(path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if vcvars.exists():
            print(f"✅ Found at: {path}")
            return vcvars

    # Check for numbered version folders (e.g., "18" for VS 2022)
    print("   Checking for numbered version folders...")
    vs_base = Path(r"C:\Program Files\Microsoft Visual Studio")

    if vs_base.exists():
        # Look for numbered folders (17, 18, etc.)
        for version_folder in vs_base.iterdir():
            if version_folder.is_dir() and version_folder.name.isdigit():
                print(f"   Found version folder: {version_folder.name}")

                # Check all subdirectories (Community, Professional, Insiders, etc.)
                for edition_folder in version_folder.iterdir():
                    if edition_folder.is_dir():
                        vcvars = edition_folder / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                        if vcvars.exists():
                            print(f"✅ Found at: {edition_folder} ({edition_folder.name} edition)")
                            return vcvars

    print("❌ Not found in any standard location")
    return None


def find_windows_sdk():
    """Find Windows SDK"""
    sdk_base = Path(r"C:\Program Files (x86)\Windows Kits\10")
    if not sdk_base.exists():
        return None

    include_dir = sdk_base / "Include"
    if not include_dir.exists():
        return None

    # Find latest version
    versions = [d.name for d in include_dir.iterdir() if d.is_dir()]
    if not versions:
        return None

    latest = sorted(versions)[-1]
    return sdk_base, latest


def compile_dll():
    """
    Compile DolphinCapture.dll from C++ source.

    NOTE: A pre-compiled DLL is already included in vision/DolphinCapture.dll.
    Only run this script if the file is missing or you want to rebuild.
    Requires Visual Studio with 'Desktop development with C++' workload.
    """
    print("=" * 70)
    print("DOLPHINCAPTURE.DLL - AUTOMATIC BUILDER")
    print("=" * 70)
    print()
    print("📁 Source file detection:")
    print("   1. Check for DolphinCapture.cpp (C++ source)")
    print("   2. Check for DolphinCapture.dll (if it's source code)")
    print("   3. Auto-copy to .cpp if needed")
    print()

    # Check for source file - try .cpp first, then .dll
    cpp_file = Path("DolphinCapture.cpp")
    dll_source = Path("DolphinCapture.dll")

    source_found = False

    # Priority 1: Check if .cpp exists
    if cpp_file.exists():
        # Check if it's actually source code (text file, not binary)
        try:
            with open(cpp_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                if '/*' in first_line or '#include' in first_line or '//' in first_line:
                    print(f"\n✅ Found C++ source: {cpp_file}")
                    source_found = True
                else:
                    print(f"\n⚠️ {cpp_file} exists but doesn't look like C++ source")
        except:
            print(f"\n⚠️ {cpp_file} exists but cannot be read as text")

    # Priority 2: Check if .dll file is actually source code
    if not source_found and dll_source.exists():
        # Check if it's source code (small file, text content)
        file_size = dll_source.stat().st_size

        if file_size < 100000:  # Less than 100KB = probably source code
            print(f"\n📝 Found source code in: {dll_source}")
            print(f"   (File size: {file_size:,} bytes - too small to be compiled DLL)")
            print(f"   Copying to {cpp_file.name}...")

            try:
                shutil.copy(dll_source, cpp_file)
                print(f"✅ Copied {dll_source.name} → {cpp_file.name}")
                source_found = True
            except Exception as e:
                print(f"❌ Failed to copy: {e}")
        else:
            print(f"\n⚠️ {dll_source} exists and is large ({file_size:,} bytes)")
            print(f"   This is probably a compiled DLL, not source code")

    # Check if we found valid source
    if not source_found:
        print(f"\n❌ No C++ source file found")
        print(f"\n💡 Expected files:")
        print(f"   - DolphinCapture.cpp (preferred)")
        print(f"   - DolphinCapture.dll (if it contains source code)")
        print(f"\n📥 Make sure you have the C++ source code before building")
        return False

    # Final verification that .cpp exists
    if not cpp_file.exists():
        print(f"\n❌ C++ source file missing: {cpp_file}")
        return False

    # Check for compiler
    print("\n🔍 Looking for Visual Studio...")
    vcvars = find_visual_studio()

    if not vcvars:
        print("\n" + "=" * 70)
        print("❌ VISUAL STUDIO C++ TOOLS NOT FOUND")
        print("=" * 70)
        print("\n📋 Troubleshooting checklist:")
        print("\n1️⃣ Is Visual Studio installed?")
        print("   Check: C:\\Program Files\\Microsoft Visual Studio\\2022\\")
        print("\n2️⃣ Did you install 'Desktop development with C++'?")
        print("   This is REQUIRED - not just 'Visual Studio'")
        print("\n3️⃣ Installation steps:")
        print("   a. Download Visual Studio Installer")
        print("      https://visualstudio.microsoft.com/downloads/")
        print("   b. Run installer")
        print("   c. Select 'Desktop development with C++' workload")
        print("   d. Click Install (may take 30-60 minutes)")
        print("\n4️⃣ After installation:")
        print("   Restart this script")
        print("\n💡 Alternative: Use MinGW (no installation needed)")
        print("   python build_dll_mingw.py")
        print("=" * 70)

        # Try to detect partial installation
        vs_base = Path(r"C:\Program Files\Microsoft Visual Studio\2022")
        if vs_base.exists():
            editions = list(vs_base.iterdir())
            if editions:
                print(f"\n⚠️ Found VS 2022 installed at: {editions[0]}")
                print("   But C++ tools are missing!")
                print("   → Re-run VS Installer and add 'Desktop development with C++'")

        return False

    print(f"✅ Found: {vcvars.parent.parent.parent.parent}")

    # Check for Windows SDK
    print("\n🔍 Looking for Windows SDK...")
    sdk_info = find_windows_sdk()

    if not sdk_info:
        print("❌ Windows SDK not found!")
        print("\n💡 Install Windows 10 SDK:")
        print("   https://developer.microsoft.com/windows/downloads/windows-sdk/")
        return False

    sdk_base, sdk_version = sdk_info
    print(f"✅ Found: Version {sdk_version}")

    # Create build script
    print("\n🔨 Creating build script...")

    build_script = Path("build_dolphin_capture.bat")

    script_content = f'''@echo off
echo ================================================
echo Building DolphinCapture.dll (x64 Release)
echo ================================================

call "{vcvars}"

echo.
echo Compiling...
cl.exe /LD /O2 /MD /EHsc ^
    /I"{sdk_base}\\Include\\{sdk_version}\\um" ^
    /I"{sdk_base}\\Include\\{sdk_version}\\shared" ^
    /I"{sdk_base}\\Include\\{sdk_version}\\ucrt" ^
    /D "NDEBUG" ^
    /D "WIN32" ^
    /D "_WINDOWS" ^
    /D "_USRDLL" ^
    DolphinCapture.cpp ^
    /link ^
    /DLL ^
    /MACHINE:X64 ^
    /OUT:DolphinCapture_compiled.dll ^
    user32.lib gdi32.lib dwmapi.lib

REM NOTE: To enable DirectX hook fallback, uncomment the line below
REM and install Microsoft Detours from https://github.com/microsoft/Detours
REM /D "ENABLE_DIRECTX_HOOK" ^

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build FAILED
    exit /b 1
)

echo.
echo ✅ Build successful: DolphinCapture_compiled.dll
exit /b 0
'''

    build_script.write_text(script_content, encoding='utf-8')
    print(f"✅ Created: {build_script}")

    # Run build
    print("\n🔨 Compiling DLL...")
    print("=" * 70)

    try:
        result = subprocess.run(
            [str(build_script)],
            shell=True,
            capture_output=True,
            text=True,
            encoding='cp850'  # Windows console encoding
        )

        # Show output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        if result.returncode != 0:
            print("\n❌ Compilation failed!")
            print("\n💡 Common issues:")
            print("   1. Missing Visual Studio C++ tools")
            print("   2. Missing Windows SDK")
            print("   3. Syntax errors in C++ code")
            return False

        # Check if DLL was created
        compiled_dll = Path("DolphinCapture_compiled.dll")
        if not compiled_dll.exists():
            print("\n❌ DLL file not created")
            return False

        # Backup old DLL if exists
        old_dll = Path("DolphinCapture_old.dll")
        target_dll = Path("DolphinCapture.dll")

        if target_dll.exists() and target_dll.stat().st_size > 10000:  # If it's a real DLL
            if old_dll.exists():
                old_dll.unlink()
            shutil.copy(target_dll, old_dll)
            print(f"📦 Backed up old DLL: {old_dll}")

        # Replace with new DLL
        if target_dll.exists():
            target_dll.unlink()
        shutil.copy(compiled_dll, target_dll)

        print("\n" + "=" * 70)
        print("✅ SUCCESS!")
        print("=" * 70)
        print(f"📦 DLL compiled: {target_dll.absolute()}")
        print(f"📏 Size: {target_dll.stat().st_size:,} bytes")

        # Cleanup intermediate files
        print("\n🧹 Cleaning up build files...")
        for pattern in ["*.obj", "*.exp", "*.lib", "DolphinCapture_compiled.dll"]:
            for file in Path(".").glob(pattern):
                file.unlink()
                print(f"   Deleted: {file}")

        return True

    except Exception as e:
        print(f"\n❌ Build error: {e}")
        return False


def main():
    success = compile_dll()

    if success:
        print("\n" + "=" * 70)
        print("🎉 Build complete! You can now run:")
        print("   python test_dolphin_capture_dll.py")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Build failed - see errors above")
        print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())