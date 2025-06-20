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
    ssize_t n = write(fd, m_network, m_network_bytes);
    return n != m_network_bytes;    
}
int NeuralNetwork::Load(int fd) {

    ssize_t n = read(fd, m_network, m_network_bytes);
    return n != m_network_bytes;
}

std::string NeuralNetwork::Visualize() {
    std::string net_size = CleanSize(m_network_bytes);
    std::string batch_size = CleanSize(m_batch_data_bytes);
    std::string test_size = CleanSize(m_test_data_bytes);


    size_t start = 0;
    std::string res = "Network Size: " + net_size + "\n";
    for (Layer& layer : m_layers) {
        std::string layer_start = CleanSize(start);
        std::string layer_end = CleanSize(start+layer.layer_bytes);
        std::string layer_size = CleanSize(layer.layer_bytes);

        res += "\tLayer "+std::to_string(layer.m_layer_idx)+" ("+Layer::ParseName(layer.type)+"): "+layer_start+" - "+layer_end+" ("+layer_size+")";
        res += layer.VisualizeNet() + "\n\n";

        start += layer.layer_bytes;
    }

    start = 0;
    res += "\nBatch Size: " + batch_size + "\n";
    for (Layer& layer: m_layers) {
        std::string layer_start = CleanSize(start);
        std::string layer_end = CleanSize(start+layer.layer_batch_bytes);
        std::string layer_size = CleanSize(layer.layer_batch_bytes);

        res += "\tLayer "+std::to_string(layer.m_layer_idx)+" ("+Layer::ParseName(layer.type)+"): "+layer_start+" - "+layer_end+" ("+layer_size+")\n";

        start += layer.layer_batch_bytes;
    }

    start = 0;
    res += "\nTest Size: " + test_size + "\n";
    for (Layer& layer: m_layers) {
        std::string layer_start = CleanSize(start);
        std::string layer_end = CleanSize(start+layer.layer_test_bytes);
        std::string layer_size = CleanSize(layer.layer_test_bytes);

        res += "\tLayer "+std::to_string(layer.m_layer_idx)+" ("+Layer::ParseName(layer.type)+"): "+layer_start+" - "+layer_end+" ("+layer_size+")\n";

        start += layer.layer_test_bytes;
    }

    return res;
}