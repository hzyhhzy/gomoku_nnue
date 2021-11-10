@echo off
:: ########################################################
:: Modify this part to your needs!

:: General configs
set ResultDir=result
set BoardSize=15
set Rule=0
set Concurrency=11
set DrawAfter=215
set LoseOnly=
set PGNFile=result.pgn
set SGFFile=result.sgf
set MSGFile=
set Logging=
set Tolerance=50000
set FatalError=

:: Time Control
set TC=10000/0.4
set Depth=
set Nodes=

:: Opening configs
set Round=1
set Games=3000
set Gauntlet=
set Repeat=
set Transform=yes
set Openings=openings/balance-freestyle15-50k.txt
set OpeningType=pos
set OpeningOrder=random
set OpeningSrand=

:: Engine list
set Engines=nnue1 nnue1
set nnue1=./pbrain-nnue1.exe

:: Bayeselo configs
set EloFile=elo.txt
set EloOffset=1600
set Advantage=0
set DrawElo=0.01

:: Sample configs
set SampleFile=sample.bin.lz4
set SampleFormat=bin_lz4
set SampleFreq=1.0

:: ########################################################
setlocal enabledelayedexpansion

:: Choose a result directory if not specified
if "%ResultDir%" == "" (
    set /p ResultDir="Enter Result Directory (empty for cwd): "
    if "!ResultDir!" == "" set ResultDir=.
)
:: Check tournament options
:PRINT
echo Sample Options:
echo Result Directory = %ResultDir%
echo Sample Format = %SampleFormat%
if defined SampleFreq (
    echo Sample Freq = %SampleFreq%
) else (
    echo Sample Freq = 1.0
)
if defined TC echo Time Control = %TC%
if defined Depth echo Depth = %Depth%
if defined Nodes echo Nodes = %Nodes%
if %Rule% == 0 (
    echo Rule = Freestyle
) else if %Rule% == 1 (
    echo Rule = Standard
) else if %Rule% == 4 (
    echo Rule = Renju
) else (
    echo Rule = Unknown Rule %Rule%
)
echo Round = %Round%
echo Games = %Games%
if defined Openings (
    echo Openings = %Openings%
) else (
    echo Openings = (empty board)
)
if defined Concurrency (
    echo Concurrency = %Concurrency%
) else (
    echo Concurrency = 1
)

:: Run the tournament
echo Start tournament...
set startTime=%TIME%

:: Create result directory
if not exist "%ResultDir%" mkdir "%ResultDir%"

:: Create engine argument
for %%e in (%Engines%) do (
    for %%i in (%%e) do set cmd=!%%i!
    set args=!args! -engine name=%%e "cmd=!cmd!"
)

:: Create global arguments
set args=%args% -each
if defined TC set args=%args% tc=%TC%
if defined Depth set args=%args% depth=%Depth%
if defined Nodes set args=%args% nodes=%Nodes%
if defined Tolerance set args=%args% tolerance=%Tolerance%
set args=%args% -boardsize %BoardSize% -rule %Rule%
set args=%args% -rounds %Round% -games %Games%
if defined Openings (
    set args=%args% -openings file=%Openings%
    if defined OpeningType set args=!args! type=%OpeningType%
    if defined OpeningOrder set args=!args! order=%OpeningOrder%
    if defined OpeningSrand set args=!args! srand=%OpeningSrand%
)
if defined Gauntlet set args=%args% -gauntlet
if defined Repeat set args=%args% -repeat
if defined Transform set args=%args% -transform
if defined LoseOnly set args=%args% -loseonly
if defined DrawAfter set args=%args% -drawafter %DrawAfter%
if defined Concurrency set args=%args% -concurrency %Concurrency%
if defined Logging set args=%args% -log
if defined FatalError set args=%args% -fatalerror
if not "%PGNFile%" == "" set args=%args% -pgn %ResultDir%/%PGNFile%
if not "%SGFFile%" == "" set args=%args% -sgf %ResultDir%/%SGFFile%
if not "%MSGFile%" == "" set args=%args% -msg %ResultDir%/%MSGFile%
if defined SampleFile (
    set args=%args% -sample file=%ResultDir%/%SampleFile%
    if defined SampleFormat set args=!args! format=%SampleFormat%
    if defined SampleFreq set args=!args! freq=%SampleFreq%
)

:: Run c-gomoku-cli
c-gomoku-cli.exe %args%
if errorlevel 1 goto ERR

:FINISHED
echo.
echo Tournament Finished.

if "%EloFile%" == "" goto END
echo Calcuating Bayeselo...
set LF=^


rem TWO empty lines are required
:: Setup bayeselo inputs
set in=readpgn %ResultDir%/%PGNFile%!LF! elo!LF!
if defined EloOffset set in=!in! offset %EloOffset%!LF!
if defined Advantage set in=!in! advantage %Advantage%!LF!
if defined DrawElo set in=!in! drawelo %DrawElo%!LF!
set in=!in! mm
if defined Advantage (set in=!in! 1) else (set in=!in! 0)
if defined DrawElo (set in=!in! 1) else (set in=!in! 0)
set "in=!in!!LF! ratings >%ResultDir%/%EloFile%!LF!"
set "in=!in! echo ------------------------->>%ResultDir%/%EloFile%!LF!"
set "in=!in! los >>%ResultDir%/%EloFile%!LF!"
set "in=!in! echo ------------------------->>%ResultDir%/%EloFile%!LF!"
set "in=!in! details >>%ResultDir%/%EloFile%!LF!"
set in=!in! x!LF! x!LF!

(cmd /v:on /c echo !in!) | bayeselo.exe 1>nul

:END
set endTime=%TIME%
FOR /F "tokens=1-4 delims=:.," %%a IN ("%startTime%") DO (
   set /A "start=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)
FOR /F "tokens=1-4 delims=:.," %%a IN ("%endTime%") DO (
   set /A "end=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)

set /A elapsed=end-start
set /A hh=elapsed/(60*60*100), rest=elapsed%%(60*60*100), mm=rest/(60*100), rest%%=60*100, ss=rest/100, cc=rest%%100
if %hh% lss 10 set hh=0%hh%
if %mm% lss 10 set mm=0%mm%
if %ss% lss 10 set ss=0%ss%
if %cc% lss 10 set cc=0%cc%
set duration=%hh%:%mm%:%ss%,%cc%

echo.
echo Tournament Ended.
echo Start    : %startTime%
echo Finish   : %endTime%
echo          ---------------
echo Duration : %duration%
goto EXIT

:ERR
echo Tournament failed with error code %errorlevel%

:EXIT