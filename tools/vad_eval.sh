CONFIG=$1
CHECKPOINT=$2

python tools/test.py $CONFIG $CHECKPOINT --launcher none --eval bbox --tmpdir tmp