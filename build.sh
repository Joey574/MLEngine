start_time=$(date +%s.%N)

p_flag=false
d_flag=false
t_flag=false

# search for relevent flags
while getopts ":pdt" opt; do
    case $opt in
        p) p_flag=true ;;
        d) d_flag=true ;;
        t) t_flag=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2 ;;
    esac
done

# all the flags
flagsopt="\
    -std=c++20 -march=native -mtune=native \
    -flto=auto -fomit-frame-pointer -funroll-loops -fipa-pta -fdevirtualize-speculatively \
    -O3 -Ofast \
    -fopenmp \
    -mavx2 -mfma -mprefer-vector-width=256 \
    -DNDEBUG \
    -falign-functions=32 -falign-loops=32 \
    -ffast-math -fno-math-errno -fassociative-math -freciprocal-math -fno-signed-zeros -fno-trapping-math \
    -fmodulo-sched -fmodulo-sched-allow-regmoves \
    -fpredictive-commoning -fhoist-adjacent-loads \
    -ftree-loop-distribution -ftree-loop-vectorize -ftree-slp-vectorize -ftree-vectorize \
    -Wno-unused-result \
    -frename-registers -fschedule-insns -fschedule-insns2 -fweb -fno-semantic-interposition \
    -frandom-seed=123 -s\
"
flagsdeb="-std=c++20 -O0 -g -DDEBUG -fno-omit-frame-pointer -fno-lto -mavx2 -mfma -fopenmp"

# .cpp dependency files
nndependencies="\
    NeuralNetwork/NNActivations.cpp NeuralNetwork/NNDerivatives.cpp \
    NeuralNetwork/NNDotProds.cpp NeuralNetwork/NNInitializations.cpp \
    NeuralNetwork/NNLogging.cpp NeuralNetwork/NNMetrics.cpp \
    NeuralNetwork/NNStaticUtils.cpp NeuralNetwork/NNTraining.cpp \
    NeuralNetwork/NNUtils.cpp"

dldependencies="DataLoader/DataLoader.cpp"
stdependencies="State/State.cpp State/StaticUtils.cpp State/StateUtils.cpp"
TESTFILES="TestNetwork/ActivationTests.cpp TestNetwork/DerivativeTests.cpp TestNetwork/Initializations.cpp TestNetwork/MathUtilTests.cpp TestNetwork/TNUtils.cpp"
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
    ccache g++ -x c++-header $FLAGS -Wno-pragmas ./Dependencies/pch.h -o ./Dependencies/pch.h.gch

    file_size=$(stat -c %s "./Dependencies/pch.h.gch")
else

    if [ "$t_flag" = true ]; then
        printf "Compiling test suite (%s)\n" $build

        # compiling phase
        for src in $DEPENDENCIES $TESTFILES "test.cpp"; do
            ccache g++ -c $FLAGS "$src" -include ./Dependencies/pch.h -o "${src%.cpp}.o"
        done

        # link phase
        ccache g++ -static-libgcc -static-libstdc++ -Wl,-Bdynamic -lgomp -Wl,-Bstatic -lstdc++ -lpthread -lm -ldl $FLAGS $(find . -name "*.o") -o MLTestEngine
        strip ./MLTestEngine

        # cleanup object files
        find . -name "*.o" -delete

        file_size=$(stat -c %s "MLTestEngine")
    else
        printf "Compiling program (%s)\n" $build

        # compiling phase
        for src in $DEPENDENCIES main.cpp; do
            ccache g++ -c $FLAGS "$src" -include ./Dependencies/pch.h -o "${src%.cpp}.o"
        done

        # link phase
        ccache g++ -static-libgcc -static-libstdc++ -Wl,-Bdynamic -lgomp -Wl,-Bstatic -lstdc++ -lpthread -lm -ldl $FLAGS $(find . -name "*.o") -o MLEngine
        strip ./MLEngine

        # cleanup object files
        find . -name "*.o" -delete
    
        file_size=$(stat -c %s "MLEngine")
    fi

fi

# output information about build process
size_human=$(numfmt --to=iec --suffix=B $file_size)
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

printf "Build completed in %.2f seconds (%s)\n" $elapsed $size_human