CONFIG=projects/configs/$1.py
CHECKPOINT=work_dirs/$1/latest.pth
RESULT_DIR=work_dirs/$1/eval/

python tools/test.py $CONFIG $CHECKPOINT --launcher none --eval bbox --tmpdir tmp --show-dir $RESULT_DIR 