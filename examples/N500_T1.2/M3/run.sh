start=`date +%s`
echo $start
# submit with caffeinate -s to run while lid closes
# caffeinate -i is another option
# caffeinate -s bash run.sh >> bashoutput.txt 2>&1

# julia won't naturally output to julia_out.txt until program is over 
# so i added progress log to run_simulation!() to flush progress

julia ./main.jl >> julia_out.txt


end=`date +%s`
runtime=$((end-start))
echo "Runtime: $runtime seconds"
