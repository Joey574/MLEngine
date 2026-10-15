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

    constexpr double xMin = -2.1;
    constexpr double xMax = 1.0;
    constexpr double yMin = -1.3;
    constexpr double yMax = 1.3;

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
    FillMandlebrotTest(img_width, img_height, depth, fourier, xMin, xMax, yMin, yMax);

    // ensure run gets a unique video file path
    std::string base = path+"images/output";
    std::string unique = DataLoader::UniqueMp4Path(base);

    std::string cmd = "ffmpeg -y -loglevel error -nostats -f rawvideo -vcodec rawvideo -s ";
    cmd += std::to_string(test_dims[0]) + "x" + std::to_string(test_dims[1]);
    cmd +=  " -pix_fmt rgb24 -r 30 -i pipe:0 -c:v libx264 -pix_fmt yuv420p " + unique;
    this->pipe = popen(cmd.c_str(), "w");
}

void DataLoader::FillMandlebrotTest(
    size_t img_width,
    size_t img_height,
    size_t depth,
    size_t fourier,
    double xMin,
    double xMax,
    double yMin,
    double yMax
) {
    const double scaleX = (xMax - xMin) / ((double)img_width - 1.0);
    const double scaleY = (yMax - yMin) / ((double)img_height - 1.0);
    const size_t cols = testData.cols;
    const size_t half = (img_height + 1) / 2;       // requires yMin == -yMax

    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t h = 0; h < half; h++) {
        const double y = yMin + (double)h * scaleY;
        const size_t hm = img_height - 1 - h;

        for (size_t w = 0; w < img_width; w++) {
            const double x = xMin + (double)w * scaleX;
            const float m = InMandlebrot(x, y, depth);

            const size_t i = h * img_width + w;
            testData.data[i * cols] = x;
            testData.data[i * cols + 1] = y;
            testLabels.data[i] = m;

            if (hm != h) {
                const size_t j = hm * img_width + w;
                testData.data[j * cols] = x;
                testData.data[j * cols + 1] = -y;
                testLabels.data[j] = m;
            }
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < testData.rows; i++) {
        ComputeFourier(&testData.data[i * cols], xMin, xMax, yMin, yMax, fourier);
    }
}
void DataLoader::LoadMandlebrotTrainData(size_t e, const std::string& path) {
    const size_t n = args[Y_SAMPLES].as<size_t>(Y_SAMPLE_DEFAULT);
    const size_t depth = args[Y_MANDLE_DEPTH].as<size_t>(Y_MANDLEDEPTH_DEFAULT);
    const size_t fourier = args[Y_FOURIER_SERIES].as<size_t>(Y_FOURIER_DEFAULT);

    constexpr double xMin = -2.1;
    constexpr double xMax = 1.0;
    constexpr double yMin = -1.3;
    constexpr double yMax = 1.3;

    std::mt19937 gen(SEED-82+e);

    std::uniform_real_distribution<double> xrand(xMin, xMax);
    std::uniform_real_distribution<double> yrand(yMin, yMax);

    const bool makeQuad = args[Y_MANDLE_SAMPLE_QUAD_TREE].as<bool>(Y_MANDLE_SAMPLE_QUAD_TREE_DEFAULT);
    const size_t quadDepth = args[Y_MANDLE_SAMPLE_QUAD_DEPTH].as<size_t>(Y_MANDLE_SAMPLE_QUAD_DEPTH_DEFAULT);

    const double uniformPercent = args[Y_MANDLE_UNIFORM_PERCENT].as<float>(Y_MANDLE_UNIFORM_PERCENT_DEFAULT);
    const size_t xRes = args[Y_MANDLE_SAMPLE_X_RES].as<size_t>(Y_MANDLE_SAMPLE_X_RES_DEFAULT);
    const size_t yRes = args[Y_MANDLE_SAMPLE_Y_RES].as<size_t>(Y_MANDLE_SAMPLE_Y_RES_DEFAULT);

    const float kJumpEps = args[Y_MANDLE_SAMPLE_JUMP_EQS].as<float>(Y_MANDLE_SAMPLE_JUMP_EQS_DEFAULT);

    if (!boundaryBuilt) {
        boundaryCells = GenerateBoundaryCells(xMin, xMax, yMax, xRes, yRes, depth, kJumpEps);
        if (makeQuad) {
            boundaryCells = GenerateBoundaryCellsQuadTree(boundaryCells, depth, quadDepth, kJumpEps, false);
        }

        boundaryBuilt = true;
    }

    const size_t nCells = boundaryCells.size();
    std::vector<uint32_t> perm(nCells);
    std::iota(perm.begin(), perm.end(), 0u);

    const size_t need = std::min(n, nCells);
    for (size_t k = 0; k < need; k++) {
        std::uniform_int_distribution<size_t> pick(k, nCells - 1);
        std::swap(perm[k], perm[pick(gen)]);
    }

    std::uniform_real_distribution<double> prob(0.0, 1.0);
    const double r0 = prob(gen), r1 = prob(gen);
    const auto frac = [](double v) {
        return v - std::floor(v);
    };

    constexpr double A1 = 0.7548776662466927;
    constexpr double A2 = 0.5698402909980532;

    size_t kb = 0, ku = 0;
    for (size_t i = 0; i < n; i++) {
        double x, y;

        if (nCells == 0 || prob(gen) < uniformPercent) {
            const double k = (double)(++ku);
            x = xMin + frac(r0 + k*A1) * (xMax-xMin);
            y = yMin + frac(r1 + k*A2) * (yMax-yMin);
        } else {
            const BoundaryCell c = boundaryCells[perm[kb++ % need]];
            x = c.x + prob(gen) * c.width;
            y = c.y + prob(gen) * c.height;
            if (gen() & 1u) y = -y;
        }

        trainData.data[i * trainData.cols] = x;
        trainData.data[i * trainData.cols+1] = y;
    }

    const bool dump = args[Y_MANDLE_DUMP_RAW_DATASET].as<bool>(Y_MANDLE_DUMP_RAW_DATASET_DEFUALT);
    if (dump) {
        std::filesystem::create_directories(path);

        const bool shouldExit = args[Y_MANDLE_EXIT_ON_DUMP].as<bool>(Y_MANDLE_EXIT_ON_DUMP_DEFAULT);

        std::vector<float> xy(2 * trainData.rows);
        for (size_t i = 0; i < trainData.rows; i++) {
            xy[2*i] = trainData.data[i*trainData.cols];
            xy[2*i + 1] = trainData.data[i*trainData.cols + 1];
        }

        const std::string datasetDump = path+"dataset.bin";
        const std::string boundaryDump = path+"boundaries.bin";

        std::ofstream pointsF(datasetDump, std::ios::binary);
        if (pointsF.is_open()) {
            pointsF.write(reinterpret_cast<const char*>(xy.data()), xy.size() * sizeof(float));
            pointsF.close();
        } else {
            std::cerr << "failed to open file at " << datasetDump << "\n";
        }

        std::ofstream cellF(boundaryDump, std::ios::binary);
        if (cellF.is_open()) {
            cellF.write(reinterpret_cast<const char*>(boundaryCells.data()), boundaryCells.size()*sizeof(BoundaryCell));
            cellF.close();
        } else {
            std::cerr << "failed to open file at " << boundaryDump << "\n";
        }

        if (shouldExit) exit(0);
    }

    #pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < n; i++) {
        float* r = &trainData.data[i * trainData.cols];
        trainLabels.data[i] = InMandlebrot(r[0], r[1], depth);
        ComputeFourier(r, xMin, xMax, yMin, yMax, fourier);
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
    const double y2 = y*y;

    const double xq = x - 0.25;
    const double q = xq*xq + y2;
    if (q * (q+xq) <= 0.25 * y2) {
        return 1.0f;
    }

    const double xp = x + 1.0;
    if (xp*xp + y2 < 0.0625) {
        return 1.0f;
    }

    double zr = 0.0, zi = 0.0, zr2 = 0.0, zi2 = 0.0;
    double sr = 0.0, si = 0.0;
    size_t period = 1, steps = 0;

    for (size_t i = 0; i < it; i++) {
        zi = 2.0 * zr * zi + y;
        zr = zr2 - zi2 + x;
        zr2 = zr * zr;
        zi2 = zi * zi;

        if (zr2 + zi2 > 4.0) {
            return (float)((double)i / ((double)i + 50.0));
        }

        if (zr == sr && zi == si) {
            return 1.0f;
        }

        if (++steps == period) {
            sr = zr;
            si = zi;
            period *= 2;
            steps = 0;
        }
    }

    return 1.0f;
}

double DataLoader::MandleDistanceLowerBound(double cx, double cy, size_t it) {
    double zr = 0.0, zi = 0.0, dr = 0.0, di = 0.0;
    for (size_t i = 0; i < it; i++) {
        const double ndr = 2.0 * (zr * dr - zi * di) + 1.0;
        di = 2.0 * (zr * di + zi * dr);
        dr = ndr;

        const double nzi = 2.0 * zr * zi + cy;
        zr = zr * zr - zi * zi + cx;
        zi = nzi;

        const double m2 = zr * zr + zi * zi;
        if (m2 > 1.0e10) {
            const double m = std::sqrt(m2);
            return 0.5 * m * std::log(m) / std::sqrt(dr * dr + di * di);
        }
    }

    return -1.0;
}

std::string DataLoader::ExpandPath(const std::string& path) {
    if (path.empty() || path[0] != '~') {
        return path;
    }

    const char* home = getenv("HOME");
    return home + path.substr(1);
}

std::string DataLoader::UniqueMp4Path(const std::string& base) {
    namespace fs = std::filesystem;
    if (!fs::exists(base + ".mp4")) return base + ".mp4";
    for (int xt = 1;; xt++) {
        std::string p = base + "." + std::to_string(xt) + ".mp4";
        if (!fs::exists(p)) return p;
    }
}
