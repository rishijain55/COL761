if [ "$1" = "C" ]; then
    ./src/main $2 $3
fi

if [ "$1" = "D" ]; then
    ./src/decompressor $2 $3
fi
