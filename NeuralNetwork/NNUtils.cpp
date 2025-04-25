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
    /*
        loss
        metric
        weight_init
        dimensions
        activations
    */

    nlohmann::json metadata;
    metadata["loss"] = LossMetricString(m_loss.type);
    metadata["metric"] = LossMetricString(m_metric.type);
    metadata["weights"] = WeightString(m_weight_init);
    metadata["dimensions"] = CompactDimensions();
    metadata["activations"] = CompactActvations();
    metadata["parameters"] = m_network_size;

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

float NeuralNetwork::Sum256(const __m256& x) {
	__m256 sum1 = _mm256_hadd_ps(x, x);
    __m256 sum2 = _mm256_hadd_ps(sum1, sum1);

    __m128 low  = _mm256_castps256_ps128(sum2);
    __m128 high = _mm256_extractf128_ps(sum2, 1);
    __m128 res  = _mm_add_ps(low, high);

    return _mm_cvtss_f32(res);
}