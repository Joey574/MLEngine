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

int NeuralNetwork::Save(int fd) const {
    ssize_t n = write(fd, m_network, m_network_size*sizeof(float));
    return n != m_network_size*sizeof(float);    
}
int NeuralNetwork::Load(int fd, Layer::WeightInitialization trueweight) {
    m_weightinit = trueweight;

    ssize_t n = read(fd, m_network, m_network_size*sizeof(float));
    return n != m_network_size*sizeof(float);
}
