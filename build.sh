BUILDFLAGS="-O3 -ftree-vectorize -flto -fomit-frame-pointer -funroll-loops -fno-exceptions -DNDEBUG -fipa-pta -fdevirtualize-speculatively"
MCFLAGS="-march=native -fopenmp -mavx2 -mfma -Ofast"

NNDEPENDENCIES="NeuralNetwork/DotProds.cpp NeuralNetwork/Training.cpp NeuralNetwork/Utils.cpp"
DLDEPENDENCIES="DataLoader/DataLoader.cpp"

DEPENDENCIES="$NNDEPENDENCIES $DLDEPENDENCIES"

g++ -static $BUILDFLAGS $MCFLAGS $DEPENDENCIES main.cpp -o MLEngine

# make doubly sure we strip all debug information
strip ./MLEngine

# clean display
echo "MLEngine:"
ls -lh ./MLEngine
echo ""