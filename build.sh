start_time=$(date +%s.%N)

p_flag=false

# search for relevent flags
while getopts ":p" opt; do
    case $opt in
        p) p_flag=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2 ;;
    esac
done

# all the flags
flags1="-std=c++20 -O3 -ftree-vectorize -flto=auto -fomit-frame-pointer -funroll-loops -DNDEBUG -fipa-pta -fdevirtualize-speculatively"
flags2="-march=native -fopenmp -mavx2 -mfma -Ofast -frandom-seed=123"
FLAGS="$flags1 $flags2"


# .cpp dependency files
nndependencies="NeuralNetwork/DotProds.cpp NeuralNetwork/Training.cpp NeuralNetwork/Utils.cpp NeuralNetwork/StaticUtils.cpp NeuralNetwork/Initialization.cpp"
dldependencies="DataLoader/DataLoader.cpp"
stdependencies="State/State.cpp State/StaticUtils.cpp"
DEPENDENCIES="$nndependencies $dldependencies $stdependencies"

declare file_size

# compile based on args passed
if [ "$p_flag" = true ]; then

    printf "Compiling pch\n"
    ccache g++ -std=c++20 -x c++-header $FLAGS ./Dependencies/pch.h -o ./Dependencies/pch.h.gch

    file_size=$(stat -c %s "./Dependencies/pch.h.gch")
else

    printf "Compiling program\n"
    ccache g++ --static $FLAGS $DEPENDENCIES -include ./Dependencies/pch.h main.cpp -o MLEngine
    strip ./MLEngine

    file_size=$(stat -c %s "MLEngine")
fi

# output information about build process
size_human=$(numfmt --to=iec --suffix=B $file_size)
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

printf "Build completed in %.2f seconds (%s)\n" $elapsed $size_human