@echo off
md vdata
md tdata
cd origin
for /d %%i in (*) do move %%i\vdata\*.npz ..\vdata\
for /d %%i in (*) do move %%i\tdata\*.npz ..\tdata\