
:loop
cd cli
call selfplay.bat
cd ..
python train/process_bin.py cli/result/sample.bin.lz4 current
cd train
python train.py --save mainline --size 128 16 32 30 --type mix6 --epoch 1 --data ../current.npz --bs 128
python export_mix6.py --copy --model mainline
cd ..
copy /y train\export\mainline.txt cli\model.txt
goto loop