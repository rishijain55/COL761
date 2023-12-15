#!/bin/sh

rm plot_point.txt
input_file="../167.txt_graph"
cd FSG/ 
g++ --std=c++11 -o input Input_conversion_fsg.cpp -O3
./input $input_file FSG_input
chmod +x fsg
./fsg -s 5 FSG_input >> ../output_fsg.txt
./fsg -s 10 FSG_input >> ../output_fsg.txt
./fsg -s 25 FSG_input >> ../output_fsg.txt
./fsg -s 50 FSG_input >> ../output_fsg.txt
./fsg -s 95 FSG_input >> ../output_fsg.txt

cd ../gSpan
g++ --std=c++11 -o input Input_conversion_gSpan.cpp -O3
./input $input_file gSpan_input
chmod +x gSpan-64
./gSpan-64 -f gSpan_input -s 0.05 -o -i >> ../output_gSpan.txt
./gSpan-64 -f gSpan_input -s 0.10 -o -i >> ../output_gSpan.txt
./gSpan-64 -f gSpan_input -s 0.25 -o -i >> ../output_gSpan.txt
./gSpan-64 -f gSpan_input -s 0.50 -o -i >> ../output_gSpan.txt
./gSpan-64 -f gSpan_input -s 0.95 -o -i >> ../output_gSpan.txt

cd ../Gaston
make
g++ --std=c++11 -o input Input_conversion_gaston.cpp -O3
./input $input_file Gaston_input
chmod +x gaston
lines=$(grep "#" "$input_file")
line_count=$(echo "$lines" | wc -l)
l5=$((line_count* 5/100))
l10=$((line_count* 10/100))
l25=$((line_count* 25/100))
l50=$((line_count* 50/100))
l95=$((line_count* 95/100))
./gaston $l5 Gaston_input >> ../output_gaston.txt
./gaston $l10 Gaston_input >> ../output_gaston.txt
./gaston $l25 Gaston_input >> ../output_gaston.txt
./gaston $l50 Gaston_input >> ../output_gaston.txt
./gaston $l95 Gaston_input >> ../output_gaston.txt
make clean
cd ..

sh plot_gen.sh
rm output_fsg.txt output_gSpan.txt output_gaston.txt 
rm FSG/FSG_input gSpan/gSpan_input Gaston/Gaston_input
rm FSG/input gSpan/input Gaston/input