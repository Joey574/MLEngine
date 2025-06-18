#include "NeuralNetwork.hpp"

void NeuralNetwork::SaveBest(nlohmann::json& history, nlohmann::json& storedhistory, float score, size_t e) {
    // save best score this training run
    if (!history.contains(J_BESTSCORE)) {
        history[J_BESTSCORE] = score;
        history[J_BESTEPOCH] = e;
    } else {
        float best = history[J_BESTSCORE];

        if ((m_layers.back().lossmetric.highestIsBest && score > best) || (!m_layers.back().lossmetric.highestIsBest && score < best)) {
			history[J_BESTSCORE] = score;
			history[J_BESTEPOCH] = e;
		}
    }

    // update best of all time score
    if ((!storedhistory.contains(J_BESTEVSCORE)) || (m_layers.back().lossmetric.highestIsBest && score > storedhistory[J_BESTEVSCORE]) || (!m_layers.back().lossmetric.highestIsBest && score < storedhistory[J_BESTEVSCORE])) {
        storedhistory[J_BESTEVSCORE] = score;
        m_epoch_since_improvement = 0;
    } else {
        m_epoch_since_improvement++;
    }

    // score has been updated, save the model
    if (m_epoch_since_improvement == 0) {
        int fd = open((m_path+m_name+".model").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        Save(fd);
        close(fd);
    }
}

Layer::WeightInitialization NeuralNetwork::ParseWeight(const std::string& w) {
    if (w == "he") {
        return Layer::WeightInitialization::he;
    } else if (w == "xavier") {
        return Layer::WeightInitialization::xavier;
    } else if (w == "normalize") {
        return Layer::WeightInitialization::normalize;
    }

    return Layer::WeightInitialization::none;
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

std::string NeuralNetwork::WeightName(Layer::WeightInitialization w) {
    switch (w) {
        case Layer::WeightInitialization::he:
            return "he";
        case Layer::WeightInitialization::xavier:
            return "xavier";
        case Layer::WeightInitialization::normalize:
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
int NeuralNetwork::Load(int fd, Layer::WeightInitialization trueweight) {
    m_weightinit = trueweight;

    ssize_t n = read(fd, m_network, m_network_size*sizeof(float));
    return n != m_network_size*sizeof(float);
}
