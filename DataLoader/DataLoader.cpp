#include "DataLoader.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

/// @brief Returns the passed dataset as constrained by args
void DataLoader::LoadDataset(YAML::Node& config, const std::string& path) {
    std::string dataset = config[Y_DATASET].as<std::string>();
    args = config[Y_DATASETARGS];

    refresh_every = args[Y_AUGMENT_REF_INTERVAL].as<size_t>(Y_AUGMENT_REF_INTERVAL_DEFAULT);

    if (dataset == "mnist") {
        LoadMNIST();
    } else if (dataset == "fmnist") {
        LoadFMNIST();
    } else if (dataset == "mandlebrot") {
        LoadMandlebrot(path);
    } else {
        std::cerr << "Failed to initialize dataset\n";
    }
}

void DataLoader::LoadMNIST() {
    name = "mnist";
    type = Type::mnist;
    hasTestData = true;

    // training dataset path
    std::string trainingImages = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TrainingData/train-images.idx3-ubyte");
    std::string trainingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TrainingData/train-labels.idx1-ubyte");

    // testing dataset path
    std::string testingImages = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TestingData/t10k-images.idx3-ubyte");
    std::string testingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/MNIST/TestingData/t10k-labels.idx1-ubyte");

    // open files
    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        std::cerr << "Failed to open dataset file(s)\n";
        exit(1);
    }

    LoadMNISTStyleDataset(traind, trainl, testd, testl);

    // close files
    traind.close();
    trainl.close();
    testd.close();
    testl.close();
}
void DataLoader::LoadFMNIST() {
    name = "fmnist";
    type = Type::fmnist;
    hasTestData = true;

    // training dataset path
    std::string trainingImages = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TrainingData/train-images-idx3-ubyte");
    std::string trainingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TrainingData/train-labels-idx1-ubyte");

    // testing dataset path
    std::string testingImages = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TestingData/t10k-images-idx3-ubyte");
    std::string testingLabels = ExpandPath("~/.local/share/MLEngine/Datasets/FMNIST/TestingData/t10k-labels-idx1-ubyte");

    // open files
    std::ifstream traind(trainingImages, std::ios::binary);
    std::ifstream trainl(trainingLabels, std::ios::binary);
    std::ifstream testd(testingImages, std::ios::binary);
    std::ifstream testl(testingLabels, std::ios::binary);

    if (!traind.is_open() || !trainl.is_open() || !testd.is_open() || !testl.is_open()) {
        std::cerr << "Failed to open dataset file(s)\n";
        exit(1);
    }

    LoadMNISTStyleDataset(traind, trainl, testd, testl);

    // close files
    traind.close();
    trainl.close();
    testd.close();
    testl.close();
}
void DataLoader::LoadMandlebrot(const std::string& path) {
    if (pipe != nullptr) {
        pclose(pipe);
        pipe = nullptr;
    }

    name = "mandlebrot";
    type = Type::mandlebrot;
    hasTestData = true;

    const size_t n = args[Y_SAMPLES].as<size_t>(Y_SAMPLE_DEFAULT);
    const size_t depth = args[Y_MANDLE_DEPTH].as<size_t>(Y_MANDLEDEPTH_DEFAULT);
    const size_t fourier = args[Y_FOURIER_SERIES].as<size_t>(Y_FOURIER_DEFAULT);

    const size_t img_width = args[Y_MANDLE_IMG_WIDTH].as<size_t>(Y_MANDLE_IMG_WIDTH_DEFAULT);
    const size_t img_height = args[Y_MANDLE_IMG_HEIGHT].as<size_t>(Y_MANDLE_IMG_HEIGHT_DEFAULT);
    test_dims = std::vector<size_t>(2); test_dims[0] = img_width; test_dims[1] = img_height;

    const size_t test_elements = img_width*img_height;

    const double xMin = -2.5;
    const double xMax = 1.0;
    const double yMin = -1.1;
    const double yMax = 1.1;

    std::mt19937 gen(SEED-82);

    trainData.rows = n;
    trainData.cols = 2 + (fourier*4);
    testData.rows = test_elements;
    testData.cols = 2 + (fourier*4);

    trainLabels.rows = n;
    trainLabels.cols = 1;
    testLabels.rows = test_elements;
    testLabels.cols = 1;

    trainData.data = std::vector<float>(trainData.rows*trainData.cols);
    testData.data = std::vector<float>(testData.rows*testData.cols);

    trainLabels.data = std::vector<float>(n);
    testLabels.data = std::vector<float>(test_elements);

    LoadMandlebrotTrainData(-1, path);

    // build testing dataset
    double scaleX = (std::abs(xMin-xMax)/((double)img_width-1.0));
    double scaleY = (std::abs(yMin-yMax)/((double)img_height-1.0));

    #pragma omp parallel for simd schedule(static)
    for (size_t h = 0; h < img_height; h++) {
        for (size_t w = 0; w < img_width; w++) {
            const size_t i = h*img_width+w;

            double x = xMin+(double)w*scaleX;
            double y = yMin+(double)h*scaleY;

            float m = InMandlebrot(x, y, depth);

            testData.data[i*testData.cols] = x;
            testData.data[i*testData.cols+1] = y;
            testLabels.data[i] = m;
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < test_elements; i++) {
        ComputeFourier(&testData.data[i*testData.cols], xMin, xMax, yMin, yMax, fourier);
    }

    // ensure run gets a unique video file path
    std::string mp4Path = path+"images/output";
    std::string ext = ".mp4";
    if (std::filesystem::exists(mp4Path+ext)) {
        mp4Path = mp4Path.append(". ");
        int xt = 0;

        do {
            xt++;
            mp4Path[mp4Path.length()-1] = std::to_string(xt)[0];
        } while (std::filesystem::exists(mp4Path+ext));
    }

    std::string cmd = "ffmpeg -y -loglevel error -nostats -f rawvideo -vcodec rawvideo -s ";
    cmd += std::to_string(test_dims[0]) + "x" + std::to_string(test_dims[1]);
    cmd +=  " -pix_fmt rgb24 -r 30 -i pipe:0 -c:v libx264 -pix_fmt yuv420p " + mp4Path+ext;
    this->pipe = popen(cmd.c_str(), "w");
}

void DataLoader::LoadMandlebrotTrainData(size_t e, const std::string& path) {
    const size_t n = args[Y_SAMPLES].as<size_t>(Y_SAMPLE_DEFAULT);
    const size_t depth = args[Y_MANDLE_DEPTH].as<size_t>(Y_MANDLEDEPTH_DEFAULT);
    const size_t fourier = args[Y_FOURIER_SERIES].as<size_t>(Y_FOURIER_DEFAULT);

    const double xMin = -2.5;
    const double xMax = 1.0;
    const double yMin = -1.1;
    const double yMax = 1.1;

    std::mt19937 gen(SEED-82+e);

    std::uniform_real_distribution<double> xrand(xMin, xMax);
    std::uniform_real_distribution<double> yrand(yMin, yMax);

    const bool makeQuad = args[Y_MANDLE_SAMPLE_QUAD_TREE].as<bool>(Y_MANDLE_SAMPLE_QUAD_TREE_DEFAULT);
    const size_t quadDepth = args[Y_MANDLE_SAMPLE_QUAD_DEPTH].as<size_t>(Y_MANDLE_SAMPLE_QUAD_DEPTH_DEFAULT);

    const double uniformPercent = args[Y_MANDLE_UNIFORM_PERCENT].as<float>(Y_MANDLE_UNIFORM_PERCENT_DEFAULT);
    const size_t xRes = args[Y_MANDLE_SAMPLE_X_RES].as<size_t>(Y_MANDLE_SAMPLE_X_RES_DEFAULT);
    const size_t yRes = args[Y_MANDLE_SAMPLE_Y_RES].as<size_t>(Y_MANDLE_SAMPLE_Y_RES_DEFAULT);

    auto cells = GenerateBoundaryCells(xMin, xMax, yMin, yMax, xRes, yRes, depth);
    if (makeQuad) {
        cells = GenerateBoundaryCellsQuadTree(cells, xMin, xMax, yMin, yMax, depth, quadDepth);
    }

    std::uniform_real_distribution<double> prob(0.0, 1.0);

    // build training dataset
    for (size_t i = 0; i < n; i++) {
        double x;
        double y;

        if (cells.empty() || prob(gen) < uniformPercent) {
            x = xrand(gen);
            y = yrand(gen);
        } else {
            std::tie(x, y) = SampleBoundary(gen, cells);
        }

        trainData.data[i*trainData.cols] = x;
        trainData.data[i*trainData.cols+1] = y;
        trainLabels.data[i] = InMandlebrot(x, y, depth);
    }

    const bool dump = args[Y_MANDLE_DUMP_RAW_DATASET].as<bool>(Y_MANDLE_DUMP_RAW_DATASET_DEFUALT);
    if (dump) {
        const bool shouldExit = args[Y_MANDLE_EXIT_ON_DUMP].as<bool>(Y_MANDLE_EXIT_ON_DUMP_DEFAULT);

        std::vector<float> xy(2 * trainData.rows);
        for (size_t i = 0; i < trainData.rows; i++) {
            xy[2*i]     = trainData.data[i*trainData.cols];
            xy[2*i + 1] = trainData.data[i*trainData.cols + 1];
        }

        std::ofstream f(path+"dataset.bin", std::ios::binary);
        if (f.is_open()) {
            f.write(reinterpret_cast<const char*>(xy.data()), xy.size() * sizeof(float));
            f.close();
        }

        if (shouldExit) {
            exit(0);
        }
    }

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        ComputeFourier(&trainData.data[i*trainData.cols], xMin, xMax, yMin, yMax, fourier);
    }
}

void DataLoader::LoadMNISTStyleDataset(std::ifstream& traind, std::ifstream& trainl, std::ifstream& testd, std::ifstream& testl) {
    ReadBigInt(&trainl);
    ReadBigInt(&trainl);

    ReadBigInt(&traind);
    int imagenum = ReadBigInt(&traind);
    int width = ReadBigInt(&traind);
    int height = ReadBigInt(&traind);

    // set up vector sizes
    originalData.data = std::vector<float>(imagenum*width*height, 0.0f);
    originalLabels.data = std::vector<float>(imagenum);

    originalData.rows = imagenum;
    originalData.cols = width*height;

    originalLabels.rows = imagenum;
    originalLabels.cols = 1;

    // parse out training data
    for (int i = 0; i < imagenum; i++) {
        // read one image from the file
        std::vector<uint8_t> bytes(width * height);
        traind.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        // convert data to float array
        std::vector<float> floatdata(bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatdata.begin(), [](uint8_t val) { return (float)val / 255.0f; });

        // insert data into dataset
        std::memcpy(&originalData.data[i*width*height], &floatdata[0], width*height*sizeof(float));

        // get label for data
        char byte;
        trainl.read(&byte, 1);
        int label = static_cast<int>(static_cast<unsigned char>(byte));
        originalLabels.data[i] = label;
    }

    ReadBigInt(&testl);
    ReadBigInt(&testl);

    ReadBigInt(&testd);
    imagenum = ReadBigInt(&testd);
    width = ReadBigInt(&testd);
    height = ReadBigInt(&testd);

    // set up vector sizes
    testData.data = std::vector<float>(imagenum*width*height, 0.0f);
    testLabels.data = std::vector<float>(imagenum);

    testData.rows = imagenum;
    testData.cols = width*height;

    testLabels.rows = imagenum;
    testLabels.cols = 1;

    // parse out test data
    for (int i = 0; i < imagenum; i++) {
        // read one image from the file
        std::vector<uint8_t> bytes(width * height);
        testd.read(reinterpret_cast<char*>(bytes.data()), bytes.size());

        // convert data to float array
        std::vector<float> floatdata(bytes.size());
        std::transform(bytes.begin(), bytes.end(), floatdata.begin(), [](uint8_t val) { return (float)val / 255.0f; });

        // insert data into dataset
        std::memcpy(&testData.data[i*width*height], &floatdata[0], width*height*sizeof(float));

        // get label for data
        char byte;
        testl.read(&byte, 1);
        int label = static_cast<int>(static_cast<unsigned char>(byte));
        testLabels.data[i] = label;
    }

    trainData.rows = originalData.rows;
    trainData.cols = originalData.cols;

    // size in augmentations
    trainData.rows += originalData.rows*args[Y_ROT_VARIANTS].as<size_t>(Y_ROT_VAR_DEFAULT);
    trainData.rows += originalData.rows*args[Y_SCALE_VARIANTS].as<size_t>(Y_SCALE_VAR_DEFAULT);
    trainData.rows += originalData.rows*args[Y_SHEAR_VARIANTS].as<size_t>(Y_SHEAR_VAR_DEFAULT);
    trainData.rows += originalData.rows*args[Y_ELASTIC_DEFORM][Y_ELASTIC_VARIANTS].as<size_t>(Y_ELASTIC_VAR_DEFAULT);
    trainLabels.rows = trainData.rows; trainLabels.cols = 1;

    // reserve size for augmentations
    trainData.data = std::vector<float>(trainData.rows*trainData.cols, 0.0f);
    trainLabels.data = std::vector<float>(trainLabels.rows*trainLabels.cols, 0.0f);

    // set dataset dimensions
    dims = std::vector<size_t>(2, 28);

    // check enabled augmentations
    uint8_t augments = 0;

    if (args[Y_ROTATION]) { augments |= 0b1; }
    if (args[Y_SCALE]) { augments |= 0b10; }
    if (args[Y_SHEAR]) { augments |= 0b100; }
    if (args[Y_ELASTIC_DEFORM]) { augments |= 0b1000; }

    // select correct augment function
    if (augments == 0b00000001) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000001>);
    } else if (augments == 0b00000010) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000010>);
    } else if (augments == 0b00000011) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000011>);
    } else if (augments == 0b00000100) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000100>);
    } else if (augments == 0b00000101) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000101>);
    } else if (augments == 0b00000110) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000110>);
    } else if (augments == 0b00000111) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00000111>);
    } else if (augments == 0b00001000) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001000>);
    } else if (augments == 0b00001001) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001001>);
    } else if (augments == 0b00001010) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001010>);
    } else if (augments == 0b00001011) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001011>);
    } else if (augments == 0b00001100) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001100>);
    } else if (augments == 0b00001101) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001101>);
    } else if (augments == 0b00001110) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001110>);
    } else if (augments == 0b00001111) {
        augment = static_cast<AugmentFn>(&DataLoader::Augment<0b00001111>);
    } else {
        // default state, no augmentations
        augment = nullptr;
    }

    // if we just pre generate augmentations, build trainData set now and set augment to nullptr
    if (refresh_every == -1 && augment) {
        (this->*augment)(1234);
        augment = nullptr;
    }
}

int DataLoader::ReadBigInt(std::ifstream* f) {
    int lint;
    f->read(reinterpret_cast<char*>(&lint), sizeof(int));

    unsigned char* bytes = reinterpret_cast<unsigned char*>(&lint);
    std::swap(bytes[0], bytes[3]);
    std::swap(bytes[1], bytes[2]);

    return lint;
}

float DataLoader::InMandlebrot(double x, double y, size_t it) {
    std::complex<double> c(x, y);
        std::complex<double> z = 0;

        for (size_t i = 0; i < it; i++) {
            z = z * z + c;
            if (std::abs(z) > 2.0) {
                return (float)((double)i / ((double)i + 50.0));            }
        }
        return 1.0f;
}

std::string DataLoader::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}
