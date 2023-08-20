rm dec
g++ -o dec decompressor.cpp 
./dec ../tests/D_medium_out.dat ../tests/D_medium_out_d.dat
rm comp
g++ -o comp comparator.cpp -O3
./comp ../tests/D_medium.dat ../tests/D_medium_out_d.dat