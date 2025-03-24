EXECUTABLE_PATH=./bin/micro-benchmarks
U_TYPES="u32 u64"
F_TYPES="f32 f64"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateful-branchless"
PATCHERS="dummy naive naive-branchless prefetch-all prefetch-all-branchless"
VECTOR_COUNT=256
VBW_START=0
VBW_END=8
EC_START=20
EC_END=20

LOG_FILE=/tmp/log
echo "============================================="
echo "Old-fls decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH u32 query-multi-column 1 32 old-fls none $VBW_START $VBW_END 0 0 $VECTOR_COUNT {} ::: 0
./test-scripts/print-joblog.sh
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH f32 query-multi-column 1 32 old-fls {1} $VBW_START $VBW_END $EC_START $EC_END $VECTOR_COUNT 0 ::: $PATCHERS
./test-scripts/print-joblog.sh

echo "============================================="
echo "Main kernels"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH {1} query-multi-column {2} {3} {4} none $VBW_START $VBW_END 0 0 $VECTOR_COUNT 0 ::: $U_TYPES ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS 
./test-scripts/print-joblog.sh
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH {1} query-multi-column {2} {3} {4} {5} $VBW_START $VBW_END $EC_START $EC_END $VECTOR_COUNT 0 ::: $F_TYPES ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS ::: $PATCHERS
./test-scripts/print-joblog.sh
