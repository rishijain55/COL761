#!/bin/bash

# extracting times from output.txt
file_fsg="output_fsg.txt"

# extracting times for fsg algorithm
time_list_fsg=$(grep "Elapsed User CPU Time:" "$file_fsg" | awk '{gsub(/\[sec\]/, ""); print $5}')
echo $time_list_fsg >> plot_point.txt

# extracting times for gSpan algorithm
time_gspan=$(grep "sec" "output_gSpan.txt" | awk '{gsub(/sec/, ""); print $1}')
echo $time_gspan >> plot_point.txt

# extracting times for Gaston algorithm
time_gaston=$(grep "Approximate total runtime:" "output_gaston.txt" | awk '{gsub(/s/,"") ;print $4}')
echo $time_gaston >> plot_point.txt