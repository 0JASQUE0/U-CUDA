@echo off
rem Tests of U-CUDA (MSBuild project, no CMake). Each test is compiled by cl.exe and linked against
rem the Release objects (build U-CUDA.sln Release|x64 first), then run from tests\u-cuda\build with a
rem fresh copy of x64\Release\kernels next to it (the engines load templates from the exe dir).
rem   run_tests.bat [test ...]     tests: gpu_smoke krs_cpu adaptive architecture dae spice (none = all)
rem Exit code of a test: 0 pass, 77 skip (no CUDA device), anything else fail.
setlocal enabledelayedexpansion
call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
set HERE=%~dp0
set SRC=%HERE%..\..\U-CUDA
set OBJ=%SRC%\x64\Release
set OUT=%HERE%build
set VCPKG=%HERE%..\..\vcpkg_installed\x64-windows\x64-windows
if not exist "%OUT%" mkdir "%OUT%"
if exist "%OUT%\kernels" rmdir /s /q "%OUT%\kernels"
xcopy /e /i /q /y "%HERE%..\..\x64\Release\kernels" "%OUT%\kernels" >nul

rem Engine tests: the computation objects only.
set ENG=
for %%o in (parametric_engine data_export krs_cpu codegen sysparse system_library nvrtc_engine integrator spice_parse circuit_synth circuit_solver circuit_netlist) do set ENG=!ENG! "%OBJ%\%%o.obj"
rem Full application (sessions, AppModel): every object except the entry point and the circuit
rem runner (both define exe_dir, which the test provides).
set APP=
for %%f in ("%OBJ%\*.obj") do if /i not "%%~nf"=="app_main" if /i not "%%~nf"=="circuit_nvrtc" set APP=!APP! "%%f"
set LIBS=/LIBPATH:"%CUDA_PATH%\lib\x64" cudart_static.lib nvrtc.lib cuda.lib
set APPLIBS=/LIBPATH:"%VCPKG%\lib" imgui.lib implot.lib implot3d.lib glfw3dll.lib opengl32.lib user32.lib gdi32.lib shell32.lib ole32.lib comdlg32.lib
set CF=/nologo /EHsc /std:c++17 /O2 /MD /utf-8 /DWIN32 /DWIN64 /DNDEBUG /I"%HERE%." /I"%SRC%" /I"%SRC%\include" /I"%CUDA_PATH%\include"

set TESTS=%*
if "%TESTS%"=="" set TESTS=gpu_smoke krs_cpu adaptive architecture dae spice
set FAILED=0
for %%t in (%TESTS%) do (
  set FILE=%%t_test.cpp
  if "%%t"=="gpu_smoke" set FILE=gpu_smoke.cpp
  set OBJS=!ENG!
  set EXTRA=%LIBS%
  set DEF=
  if "%%t"=="architecture" (
    set OBJS=!APP!
    set EXTRA=%LIBS% %APPLIBS%
    set DEF=/DUT_FULL_APP /I"%VCPKG%\include"
  )
  echo === %%t
  pushd "%OUT%"
  cl %CF% !DEF! "%HERE%!FILE!" /Fe"%OUT%\%%t.exe" !OBJS! /link !EXTRA! > "%OUT%\%%t.build.log" 2>&1
  set BRC=!errorlevel!
  popd
  if not "!BRC!"=="0" (
    echo BUILD FAILED %%t: %OUT%\%%t.build.log
    findstr /i /r /c:"error" "%OUT%\%%t.build.log"
    set FAILED=1
  ) else (
    pushd "%OUT%"
    "%OUT%\%%t.exe"
    set RC=!errorlevel!
    popd
    if "!RC!"=="0" echo PASS %%t
    if "!RC!"=="77" echo SKIP %%t
    if not "!RC!"=="0" if not "!RC!"=="77" ( echo FAIL %%t ^(exit !RC!^) & set FAILED=1 )
  )
)
if "%FAILED%"=="0" (echo ALL PASSED) else (echo SOME TESTS FAILED)
exit /b %FAILED%
