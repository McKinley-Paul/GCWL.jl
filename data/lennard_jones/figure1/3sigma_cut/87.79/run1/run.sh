#!/bin/bash
start=`date +%s`
echo $start
# submit with caffeinate -s to run while lid closes
# caffeinate -s bash run.sh >> bashoutput.txt 2>&1 
# to run in the background so you can close the terminal window:
# caffeinate -s bash run.sh >> bashoutput.txt 2>&1 & 

julia ./restart.jl >> julia_out.txt


end=`date +%s`
runtime=$((end-start))

echo "Runtime: $runtime seconds"
