EXECUTABLE_PATH=./bin/micro-benchmarks
TYPES="f32 f64"
KERNELS="decompress query"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateful-branchless"
PATCHERS="stateless stateful naive naive-branchless prefetch-all prefetch-all-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
echo "============================================="
echo "Old-fls decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH f32 {1} 1 32 old-fls {2} 0 16 0 10 $VECTOR_COUNT 1 0 ::: $KERNELS ::: $PATCHERS
./test-scripts/print-joblog.sh
echo "============================================="
echo "Dummy patcher"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH {1} {2} {3} {4} {5} dummy 0 16 0 0 $VECTOR_COUNT 1 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS 
./test-scripts/print-joblog.sh
echo "============================================="
echo "Main decompressors"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH {1} {2} {3} {4} {5} {6} 0 16 0 10 $VECTOR_COUNT 1 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS ::: $PATCHERS
./test-scripts/print-joblog.sh
