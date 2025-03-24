EXECUTABLE_PATH=./bin/compressors-benchmarks
FLOAT_FILES="$(ls data/float/*.bin) 
DOUBLE_FILES="$(ls data/double/*.bin) 
COMPARISON="decompression decompression_query"
COMPRESSORS="ALP GALP Bitcomp BitcompSparse LZ4 zstd Deflate GDeflate Snappy"
VECTOR_COUNT=25600

LOG_FILE=/tmp/log
echo "============================================="
echo "Thrust"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH f32 decompression_query Thrust {1} $VECTOR_COUNT ::: $FLOAT_FILES
./test-scripts/print-joblog.sh
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH f64 decompression_query Thrust {1} $VECTOR_COUNT ::: $DOUBLE_FILES
./test-scripts/print-joblog.sh
echo "============================================="
echo "Floats"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH f32 {1} {2} {3} $VECTOR_COUNT ::: $COMPARISON ::: $COMPRESSORS ::: $FLOAT_FILES
./test-scripts/print-joblog.sh

echo "============================================="
echo "Double"
echo "============================================="
parallel --progress --joblog $LOG_FILE $EXECUTABLE_PATH f64 {1} {2} {3} $VECTOR_COUNT ::: $COMPARISON ::: $COMPRESSORS ::: $DOUBLE_FILES
./test-scripts/print-joblog.sh
