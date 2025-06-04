#include "NeuralNetwork.hpp"

nlohmann::json NeuralNetwork::Fit(Dataset& dataset, size_t batch_size, size_t epochs, float learning_rate, int validation_freq, float validation_split, bool shuffle) {
	auto fitstart = std::chrono::high_resolution_clock::now();

	std::cout << m_meta.dump(4) << "\n";
    
	nlohmann::json history;
	FitStart(history, epochs, batch_size, learning_rate);

	InitializeLayerData(batch_size, dataset.testDataRows);
	InitializeLayerPointers(batch_size, dataset.testDataRows);

	const size_t iterations = std::ceil((double)dataset.trainDataRows/(double)batch_size);

	for (size_t e = 0; e < epochs && KEEPRUNNING; e++) {
		auto epochstart = std::chrono::high_resolution_clock::now();

		// shuffle dataset each epoch
		dataset.Shuffle();

		for (size_t i = 0; i < iterations; i++) {
			float* __restrict x = &dataset.trainData[(i * batch_size) * dataset.trainDataCols];
			float* __restrict y = &dataset.trainLabels[(i * batch_size) * dataset.trainLabelCols];

			// set batch size here to be either batch size or number of elements remaining
			size_t remaining_elements = (dataset.trainDataRows - (i * batch_size));
			size_t effective_size = batch_size > remaining_elements ? remaining_elements : batch_size;

			ForwardProp(true, x, effective_size);
			BackProp(x, y, learning_rate, effective_size);
		}

		std::string res = "";
		if ((e+1) % validation_freq == 0) {
			res = TestNetwork(dataset, history, e);
		}

		double epochns = (std::chrono::high_resolution_clock::now() - epochstart).count();
		EpochEnd(history, res, epochns, e);
	}

	// forced network test to make sure we get at least one save if model wasn't validated during training
	TestNetwork(dataset, history, epochs);
	
	FitEnd(history, fitstart);
	return history;
}

std::string NeuralNetwork::TestNetwork(Dataset& dataset, nlohmann::json& history, size_t e) {
	ForwardProp(false, dataset.testData.data(), dataset.testDataRows);
	const float* predictions = m_layers.back().Output(false);

	float score = m_layers.back().lossmetric.metric(predictions, &dataset.testLabels[0], dataset.testDataRows, m_layers.back().nodes);

	SaveBest(history, score, e);
	std::string curs = "Score: " + std::to_string(score);
	std::string sesb = "Session Best: " + std::to_string((float)history[BESTSCORE]);
	std::string eveb = "Best Ever: " + std::to_string((float)m_meta[BESTEVSCORE]);

	int size = snprintf(nullptr, 0, "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());

	std::string fmt(size+1, ' ');
	sprintf(fmt.data(), "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());	
	return fmt;
}

void NeuralNetwork::ForwardProp(bool training, float* __restrict x, size_t n) {

    for (size_t i = 0; i < m_layers.size(); i++) {
		// input will always just be previous layers output
		float* __restrict input = i == 0 ? x : m_layers[i-1].Output(training);

		// does all the fun math stuff for us
		m_layers[i].forward(training, input, n);
    }
}
void NeuralNetwork::BackProp(const float* __restrict x, const float* __restrict y, float lr, size_t n) {

	// compute gradient
	for (ssize_t i = m_layers.size()-1; i >= 0; i--) {

		const float* truth = i == m_layers.size()-1 ? y : m_layers[i+1].Truth();
		const float* input = i == 0 ? x : m_layers[i-1].Output(true);

		m_layers[i].backward(truth, input, n);
	}

	// update layers
	for (size_t i = 0; i < m_layers.size(); i++) {
		m_layers[i].update(lr, n);
	}
}
