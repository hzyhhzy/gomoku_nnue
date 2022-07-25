分离tdata和vdata.bat
call conda.bat activate t
python cut_split.py
python merge_shuffle.py

python choose_blackNormal.py
python choose_whiteNormal.py