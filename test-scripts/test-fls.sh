EXECUTABLE_PATH=./bin/micro-benchmarks
TYPES="32 64"
KERNELS="decompress query"
UNPACK_N_VECS="1 4"
UNPACK_N_VALS="1"
UNPACKERS="stateful-branchless"
VECTOR_COUNT=256

LOG_FILE=/tmp/log
echo "============================================="
echo "Dummy decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH u{1} {2} {3} {4} dummy none {1} {1} 0 0 $VECTOR_COUNT 1 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS 
./test-scripts/print-joblog.sh
echo "============================================="
echo "Old-fls decompressor"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH u32 {1} 1 32 old-fls none 0 32 0 0 $VECTOR_COUNT 1 0 ::: $KERNELS 
./test-scripts/print-joblog.sh
echo "============================================="
echo "Main decompressors"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH u{1} {2} {3} {4} {5} none 0 {1} 0 0 $VECTOR_COUNT 1 0 ::: $TYPES ::: $KERNELS ::: $UNPACK_N_VECS ::: $UNPACK_N_VALS ::: $UNPACKERS
./test-scripts/print-joblog.sh
