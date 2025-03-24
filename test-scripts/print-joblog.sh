LOG_FILE=/tmp/log

TOTAL=$(cat $LOG_FILE | tail -n +2 | wc -l)

echo "$(cat $LOG_FILE | head -n 1)"

OUTPUT=$(cat $LOG_FILE | tail -n +2 | perl -ane 'print if $F[7] == 11' | sort -n)
if [ -n "$OUTPUT" ]; then
	echo Received SEGFAULT: $(echo "$OUTPUT" | wc -l) / $TOTAL 
	echo "$OUTPUT"
fi

OUTPUT=$(cat $LOG_FILE | tail -n +2 | perl -ane 'print if $F[7] == 6' | sort -n)
if [ -n "$OUTPUT" ]; then
	echo Received SIGABORT: $(echo "$OUTPUT" | wc -l) / $TOTAL 
	echo "$OUTPUT"
fi

OUTPUT=$(cat $LOG_FILE | tail -n +2 | perl -ane 'print if $F[7] != 0 and $F[7] != 6 and $F[7] != 11' | sort -n)
if [ -n "$OUTPUT" ]; then
	echo Received other signal: $(echo "$OUTPUT" | wc -l) / $TOTAL 
	echo "$OUTPUT"
fi

OUTPUT=$(cat $LOG_FILE | tail -n +2 | perl -ane '$F[6] and print' | sort -n)
if [ -n "$OUTPUT" ]; then
	echo Failed: $(echo "$OUTPUT" | wc -l) / $TOTAL 
	echo "$OUTPUT"
fi

OUTPUT=$(cat $LOG_FILE | tail -n +2 | perl -ane 'print if $F[6] == 0' | sort -n)
if [ -n "$OUTPUT" ]; then
	echo Success: $(echo "$OUTPUT" | wc -l) / $TOTAL 
	echo "$OUTPUT"
fi

