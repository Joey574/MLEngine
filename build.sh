start_time=$(date +%s.%N)

p_flag=false
d_flag=false

# search for relevent flags
while getopts ":pd" opt; do
    case $opt in
        p) p_flag=true ;;
        d) d_flag=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2 ;;
    esac
done

# all the flags
flagsopt="-std=c++20 -O3 -ftree-vectorize -flto=auto -fomit-frame-pointer -funroll-loops -DNDEBUG -fipa-pta -fdevirtualize-speculatively -march=native -fopenmp -mavx2 -mfma -Ofast -frandom-seed=123 -Wno-unused-result"
flagsdeb="-std=c++20 -O0 -g -DDEBUG -fno-omit-frame-pointer -fno-lto -mavx2 -mfma -fopenmp"


# .cpp dependency files
nndependencies="NeuralNetwork/NNActivations.cpp NeuralNetwork/NNDerivatives.cpp NeuralNetwork/NNDotProds.cpp NeuralNetwork/NNTraining.cpp NeuralNetwork/NNUtils.cpp NeuralNetwork/NNStaticUtils.cpp NeuralNetwork/NNInitializations.cpp"
dldependencies="DataLoader/DataLoader.cpp"
stdependencies="State/State.cpp State/StaticUtils.cpp"
DEPENDENCIES="$nndependencies $dldependencies $stdependencies"

declare file_size
declare FLAGS
declare build

if [ "$d_flag" = true ]; then
    build="DEBUG"
    FLAGS="$flagsdeb"
else
    build="RELEASE"
    FLAGS="$flagsopt"
fi

# compile based on args passed
if [ "$p_flag" = true ]; then

    printf "Compiling pch (%s)\n" $build
    ccache g++ -x c++-header $FLAGS ./Dependencies/pch.h -o ./Dependencies/pch.h.gch

    file_size=$(stat -c %s "./Dependencies/pch.h.gch")
else

    printf "Compiling program (%s)\n" $build
    # compiling phase
    for src in $DEPENDENCIES main.cpp; do
        ccache g++ -c $FLAGS "$src" -include ./Dependencies/pch.h -o "${src%.cpp}.o"
    done

    # link phase
    ccache g++ --static $(find . -name "*.o") -o MLEngine
    strip ./MLEngine

    # cleanup object files
    find . -name "*.o" -delete
    
    file_size=$(stat -c %s "MLEngine")
fi

# output information about build process
size_human=$(numfmt --to=iec --suffix=B $file_size)
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

printf "Build completed in %.2f seconds (%s)\n" $elapsed $size_human