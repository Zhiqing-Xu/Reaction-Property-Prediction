@echo off
setlocal enabledelayedexpansion

:: Set the size limit in bytes (100MB = 104857600 bytes)
set "sizeLimit=104857600"

:: Create a temporary file to store large files
echo|set /p=>temp.txt

:: Get the current directory
set "currentDir=%CD%"

:: Loop through each file in the current directory and its subdirectories
for /R %%F in (*) do (
    :: If the file size is greater than the limit, append its path to temp.txt
    if %%~zF GTR %sizeLimit% (
        set "fullPath=%%~dpnxF"
        set "relativePath=!fullPath:%currentDir%\=!"
        set "relativePath=!relativePath:\=/!"
        echo !relativePath! >> temp.txt
    )
)

:: Check and update .gitignore
if not exist .gitignore (
    echo #--------------------------------------------------# > .gitignore
    echo # Large Files >> .gitignore
) else (
    set "addMarkers=1"
    set "lastLine="
    for /f "usebackq delims=" %%A in (".gitignore") do (
        set "lastLine=%%A"
        if "%%A"=="# Large Files" set "addMarkers=0"
    )
    if !addMarkers! equ 1 (
        if not "!lastLine!"=="" (
            echo.>> .gitignore
        )
        echo #--------------------------------------------------# >> .gitignore
        echo # Large Files >> .gitignore
    )
)

:: Append large file paths to .gitignore
type temp.txt >> .gitignore

:: Remove duplicate lines from .gitignore
> unique_gitignore.txt (
    setlocal enabledelayedexpansion
    for /f "usebackq delims=" %%A in (".gitignore") do (
        if not defined unique_line[%%A] (
            set unique_line[%%A]=1
            echo %%A
        )
    )
    endlocal
)
move /Y unique_gitignore.txt .gitignore

:: Remove temporary file
del temp.txt

endlocal
