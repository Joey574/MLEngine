start_time=$(date +%s.%N)

p_flag=false

while getopts ":p" opt; do
    case $opt in
        p) p_flag=true ;;
        \?) echo "Invalid option: -$OPTARG" >&2 ;;
    esac
done

# all the flags
BUILDFLAGS="-std=c++20 -O3 -ftree-vectorize -flto=auto -fomit-frame-pointer -funroll-loops -DNDEBUG -fipa-pta -fdevirtualize-speculatively"
MCFLAGS="-march=native -fopenmp -mavx2 -mfma -Ofast"

NNDEPENDENCIES="NeuralNetwork/DotProds.cpp NeuralNetwork/Training.cpp NeuralNetwork/Utils.cpp NeuralNetwork/StaticUtils.cpp"
DLDEPENDENCIES="DataLoader/DataLoader.cpp"

DEPENDENCIES="$NNDEPENDENCIES $DLDEPENDENCIES"

# compile based on args passed
if [ "$p_flag" = true ]; then

    printf "Compiling pch\n"
    g++ -std=c++20 -x c++-header $BUILDFLAGS $MCFLAGS ./Dependencies/pch.h -o ./Dependencies/pch.h.gch
else

    printf "Compiling program\n"
    g++ --static $BUILDFLAGS $MCFLAGS $DEPENDENCIES -I ./Dependencies/ main.cpp -o MLEngine
    strip ./MLEngine
fi


# output elapsed time
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)
printf "Build completed in %.2f seconds\n" $elapsed