#include "NeuralNetwork.hpp"

std::string NeuralNetwork::CompactDimensions() const {
    std::string compact = "";
    
    for (size_t i = 0; i < m_layers.size() - 1; i++) {
        compact += std::to_string(m_layers[i].nodes).append("-");
    }

    compact += std::to_string(m_layers.back().nodes);
    return compact;
}
std::string NeuralNetwork::CompactActvations() const {
    std::string compact = "";

    for (size_t i = 1; i < m_layers.size() - 1; i++) {
        compact += ActivationString(m_layers[i].actvtype).append("-");
    }

    compact += ActivationString(m_layers.back().actvtype);
    return compact;
}

nlohmann::json NeuralNetwork::Metadata() const {
    nlohmann::json metadata;
    metadata["Loss"] = LossMetricString(m_loss.type);
    metadata["Metric"] = LossMetricString(m_metric.type);
    metadata["Weights"] = WeightString(m_weight_init);
    metadata["Dimensions"] = CompactDimensions();
    metadata["Activations"] = CompactActvations();
    metadata["Parameters"] = m_network_size;
    metadata["Seed"] = m_seed;

    return metadata;
}

int NeuralNetwork::Save(int fd) const {
    ssize_t n = write(fd, m_network, m_network_size*sizeof(float));
    if (n != m_network_size*sizeof(float)) {
        return 1;
    }
    return 0;
}
int NeuralNetwork::Load(int fd, WeightInitialization trueweight) {
    m_weight_init = trueweight;

    ssize_t n = read(fd, m_network, m_network_size*sizeof(float));
    if (n != m_network_size*sizeof(float)) {
        return 1;
    }
    return 0;
}

void NeuralNetwork::SaveBest(nlohmann::json& history, float score, size_t e) const {
    // save best score this training run
    if (!history.contains("Best Score")) {
        history["Best Score"] = score;
        history["Best Epoch"] = e;
    } else {
        float best = history["Best Score"];

        if ((m_metric.highestIsBest && score > best) || (!m_metric.highestIsBest && score < best)) {
			history["Best Score"] = score;
			history["Best Epoch"] = e;
		}
    }

    // get best of all time score
    std::ifstream f(m_path+"state.meta");
    if (!f.is_open()) {
        std::cerr << m_path+"state.meta not found\n";
        return;
    }

    nlohmann::json meta = nlohmann::json::parse(f);
    f.close();

    if ((!meta.contains("Best Ever Score")) || (m_metric.highestIsBest && score > meta["Best Ever Score"]) || (!m_metric.highestIsBest && score < meta["Best Ever Score"])) {
        meta["Best Ever Score"] = score;
    } else {
        return;
    }

    // score has been updated, save model and metadata
    int fd = open((m_path+m_name+".model").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    Save(fd);
    close(fd);

    fd = open((m_path+"state.meta").c_str(), O_WRONLY | O_TRUNC, 0644);
    std::string dump = meta.dump(4) + "\n";
    write(fd, dump.c_str(), dump.length());
    close(fd);
}

void NeuralNetwork::AssignActvFunctions(const std::vector<ActivationFunctions>& actvs) {
    m_layers[0].actvtype = ActivationFunctions::none;

    for (size_t i = 1; i < m_layers.size(); i++) {
        m_layers[i].actvtype = actvs[i-1];

        switch (actvs[i-1]) {
            case ActivationFunctions::sigmoid:
                m_layers[i].activation = &Sigmoid;
                m_layers[i].derivative = &SigmoidDerivative;
                break;
            case ActivationFunctions::relu:
                m_layers[i].activation = &ReLU;
                m_layers[i].derivative = &ReLUDerivative;
                break;
            case ActivationFunctions::leakyrelu:
                m_layers[i].activation = &LeakyReLU;
                m_layers[i].derivative = &LeakyReLUDerivative;
                break;
            case ActivationFunctions::elu:
                m_layers[i].activation = &ELU;
                m_layers[i].derivative = &ELUDerivative;
                break;
            case ActivationFunctions::softmax:
                m_layers[i].activation = &Softmax;
                m_layers[i].derivative = nullptr;
                break;
            default:
                break;
        }
    }
}
void NeuralNetwork::AssignLossFunctions(LossMetric loss, LossMetric metric) {
    switch (loss) {
        case LossMetric::mae:
            m_loss.type = LossMetric::mae;
            m_loss.loss = &MaeLoss;
            break;
        case LossMetric::mse:
            m_loss.type = LossMetric::mse;
            m_loss.loss = &MseLoss;
            break;
        case LossMetric::onehot:
            m_loss.type = LossMetric::onehot;
            m_loss.loss = &OneHotLoss;
            break;
        default:
            m_loss.type = LossMetric::none;
            m_loss.loss = nullptr;
    }

    switch (metric) {
        case LossMetric::mae:
            m_metric.type = LossMetric::mae;
            m_metric.highestIsBest = false;
            m_metric.metric = &MaeScore;
            break;
        case LossMetric::mse:
            m_metric.type = LossMetric::mse;
            m_metric.highestIsBest = false;
            m_metric.metric = &MseScore;
            break;
        case LossMetric::accuracy:
            m_metric.type = LossMetric::accuracy;
            m_metric.highestIsBest = true;
            m_metric.metric = &AccuracyScore;
            break;
        default:
            m_metric.type = LossMetric::none;
            m_metric.metric = nullptr;
    }
}

float NeuralNetwork::Sum256(__m256 _x) {
	__m256 _sum1 = _mm256_hadd_ps(_x, _x);
    __m256 _sum2 = _mm256_hadd_ps(_sum1, _sum1);

    __m128 _low  = _mm256_castps256_ps128(_sum2);
    __m128 _high = _mm256_extractf128_ps(_sum2, 1);
    __m128 _res  = _mm_add_ps(_low, _high);

    return _mm_cvtss_f32(_res);
}

__m256 NeuralNetwork::Exp256(__m256 _x) {
    __m256 _a = _mm256_set1_ps(12102203.0f); 
    __m256 _b = _mm256_set1_ps(127.0f * (1 << 23));
    __m256 _c = _mm256_fmadd_ps(_x, _a, _b);

    __m256i _res = _mm256_cvtps_epi32(_c);

    return _mm256_castsi256_ps(_res);
}