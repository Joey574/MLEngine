#include "NeuralNetwork.hpp"

void NeuralNetwork::SaveBest(nlohmann::json& history, float score, size_t e) {
    // save best score this training run
    if (!history.contains(BESTSCORE)) {
        history[BESTSCORE] = score;
        history[BESTEPOCH] = e;
    } else {
        float best = history[BESTSCORE];

        if ((m_layers.back().lossmetric.highestIsBest && score > best) || (!m_layers.back().lossmetric.highestIsBest && score < best)) {
			history[BESTSCORE] = score;
			history[BESTEPOCH] = e;
		}
    }

    // update best of all time score
    if ((!m_meta.contains(BESTEVSCORE)) || (m_layers.back().lossmetric.highestIsBest && score > m_meta[BESTEVSCORE]) || (!m_layers.back().lossmetric.highestIsBest && score < m_meta[BESTEVSCORE])) {
        m_meta[BESTEVSCORE] = score;
    } else {
        return;
    }

    // score has been updated, save model immediately
    int fd = open((m_path+m_name+".model").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    Save(fd);
    close(fd);
}

NeuralNetwork::WeightInitialization NeuralNetwork::ParseWeight(const std::string& w) {
    if (w == "he") {
        return WeightInitialization::he;
    } else if (w == "xavier") {
        return WeightInitialization::xavier;
    } else if (w == "normalize") {
        return WeightInitialization::normalize;
    }

    return WeightInitialization::none;
}
std::vector<size_t> NeuralNetwork::ParseCompact(const std::vector<std::string>& dims) {
    std::vector<size_t> dimensions;

    for (size_t i = 0; i < dims.size(); i++) {
        // get number of layers
        size_t n = 1;
        std::string token = dims[i];
        if (dims[i].find('X') != std::string::npos) {
            n = std::stoi(dims[i].substr(dims[i].find('X')+1));
            token = dims[i].substr(0, dims[i].find('X'));

        }

        // append n layers of t
        size_t t = std::stoi(token);
        for (size_t i = 0; i < n; i++) {
            dimensions.push_back(t);
        }
    }

    return dimensions;
}

std::string NeuralNetwork::WeightName(WeightInitialization w) {
    switch (w) {
        case WeightInitialization::he:
            return "he";
        case WeightInitialization::xavier:
            return "xavier";
        case WeightInitialization::normalize:
            return "normalize";
        default:
            return "none";
    }
}
std::vector<std::string> NeuralNetwork::CompactDimensions() const {
    std::vector<std::string> compact;
    for (size_t i = 0; i < m_layers.size(); i++) {

        size_t n = 1;
        size_t token = m_layers[i].nodes;

        // collect number of same dimensions
        for (; m_layers[i+1].nodes == token && i < m_layers.size()-1; n++, i++){}

        if (n > 1) {
            compact.push_back(std::to_string(token).append("X").append(std::to_string(n)));
        } else {
            compact.push_back(std::to_string(token));
        }
    }

    return compact;
}
std::vector<std::string> NeuralNetwork::CompactActivations() const {
     std::vector<std::string> compact;
    for (size_t i = 1; i < m_layers.size(); i++) {

        size_t n = 1;
        Activation::Type token = m_layers[i].activation.type;

        // collect number of same activations
        for (; m_layers[i+1].activation.type == token && i < m_layers.size()-1; n++, i++){}

        if (n > 1) {
            compact.push_back(Activation::ParseName(token).append("X").append(std::to_string(n)));
        } else {
            compact.push_back(Activation::ParseName(token));
        }
    }

    return compact;
}

int NeuralNetwork::Save(int fd) const {
    ssize_t n = write(fd, m_network, m_network_size*sizeof(float));
    return n != m_network_size*sizeof(float);    
}
int NeuralNetwork::Load(int fd, WeightInitialization trueweight) {
    m_weightinit = trueweight;

    ssize_t n = read(fd, m_network, m_network_size*sizeof(float));
    return n != m_network_size*sizeof(float);
}

nlohmann::json NeuralNetwork::Metadata() {
    if (!m_meta.contains(LOSS)) { m_meta[LOSS] = LossMetric::ParseName(m_layers.back().lossmetric.ltype); }
    if (!m_meta.contains(METRIC)) { m_meta[METRIC] = LossMetric::ParseName(m_layers.back().lossmetric.mtype); }
    if (!m_meta.contains(WEIGHTS)) { m_meta[WEIGHTS] = WeightName(m_weightinit); }
    if (!m_meta.contains(DIMENSIONS)) { m_meta[DIMENSIONS] = CompactDimensions(); }
    if (!m_meta.contains(ACTIVATIONS)) { m_meta[ACTIVATIONS] = CompactActivations(); }
    if (!m_meta.contains(PARAMETERS)) { m_meta[PARAMETERS] = m_network_size; }
    if (!m_meta.contains(SEED)) { m_meta[SEED] = m_seed; }

    return m_meta;
}