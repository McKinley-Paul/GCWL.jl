#!/bin/bash
start=`date +%s`
echo $start
# submit with caffeinate -s to run while lid closes
# caffeinate -i is another option
# caffeinate -s bash run.sh >> bashoutput.txt 2>&1

julia ./main.jl >> julia_out.txt


end=`date +%s`
runtime=$((end-start))

echo "Runtime: $runtime seconds"
