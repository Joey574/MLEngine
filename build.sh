start_time=$(date +%s.%N)

# values
build_type="Release"

# Parse flags
while getopts ":d" opt; do
  case $opt in
    d) build_type="Debug" ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

# build directory
mkdir -p build
cd build

# configure
export OMP_WAIT_POLICY=active
cmake .. -DCMAKE_BUILD_TYPE="$build_type" -G Ninja

# build program
echo "-- Building program ($build_type)"
cmake --build . -j

# output
file_size=$(stat -c %s "./MLEngine")
size_human=$(numfmt --to=iec --suffix=B "$file_size")
end_time=$(date +%s.%N)
elapsed=$(echo "$end_time - $start_time" | bc)

printf "Build completed in %.2f seconds (%s)\n" "$elapsed" "$size_human"
