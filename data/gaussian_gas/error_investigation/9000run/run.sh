#!/bin/bash
start=`date +%s`
echo $start

# submit job with caffeinate -s to run with computer closed, as long as it is plugged in
# caffeinate -s bash run.sh >> bashoutput.txt 2>&1

julia ./main.jl >> julia_out.txt


end=`date +%s`
runtime=$((end-start))

echo "Runtime: $runtime seconds"
