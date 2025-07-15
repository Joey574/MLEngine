#include "NeuralNetwork.hpp"

nlohmann::json NeuralNetwork::Fit(DataLoader& dataset, nlohmann::json& storedhistory) {
	auto fitstart = std::chrono::high_resolution_clock::now();

	std::cout << "\n" << config << "\n\n";

	size_t epochs = config[Y_EPOCHS].as<size_t>(Y_EPOCH_DEFAULT);
	size_t batch_size = config[Y_BATCHSIZE].as<size_t>(Y_BATCH_DEFAULT);
	size_t valid_freq = config[Y_VALIDFREQ].as<size_t>(Y_VALID_DEFAULT);
	float learning_rate = config[Y_OPT_OPTIMIZER][Y_OPT_LEARNINGRATE].as<float>(Y_LEARNRATE_DEFAULT);


	nlohmann::json history;
	FitStart(history, epochs, batch_size, learning_rate);

	InitializeLayerData(batch_size, dataset.testData.rows);
	InitializeLayerPointers(batch_size, dataset.testData.rows);

	LoadOptimizers();

	const size_t iterations = std::ceil((double)dataset.trainData.rows/(double)batch_size);
	
	for (size_t e = 0; e < epochs && KEEPRUNNING; e++) {
		auto epochstart = std::chrono::high_resolution_clock::now();

		// apply dataset deformations and shuffle
		dataset.Deform(e);

		for (size_t i = 0; i < iterations; i++) {
			float* __restrict x = &dataset.trainData.data[(i * batch_size) * dataset.trainData.cols];
			float* __restrict y = &dataset.trainLabels.data[(i * batch_size) * dataset.trainLabels.cols];

			// set batch size here to be either batch size or number of elements remaining
			ssize_t remaining_elements = (dataset.trainData.rows - (i * batch_size));
			ssize_t effective_size = batch_size > remaining_elements ? remaining_elements : batch_size;

			ForwardProp<true>(x, effective_size);
			BackProp(x, y, effective_size);
		}

		std::string res = "";
		if ((e+1) % valid_freq == 0) {
			res = TestNetwork(dataset, history, storedhistory, e);
		}

		double epochns = (std::chrono::high_resolution_clock::now() - epochstart).count();
		EpochEnd(history, res, epochns, e);
	}

	// forced network test to make sure we get at least one save if model wasn't validated during training
	TestNetwork(dataset, history, storedhistory, epochs);

	SaveOptimizers();
	
	FitEnd(history, fitstart);
	storedhistory[J_RUNS].push_back(history);
	return storedhistory;
}
float* NeuralNetwork::Predict(DataLoader& dataset) {
	ForwardProp<false>(dataset.testData.data.data(), dataset.testData.rows);
	return m_layers.back().Output<false>();
}

std::string NeuralNetwork::TestNetwork(DataLoader& dataset, nlohmann::json& history, nlohmann::json& storedhistory, size_t e) {
	ForwardProp<false>(dataset.testData.data.data(), dataset.testData.rows);
	const float* predictions = m_layers.back().Output<false>();

	float score = m_layers.back().lossmetric.metric(predictions, &dataset.testLabels.data[0], dataset.testData.rows, m_layers.back().nodes);

	SaveBest(history, storedhistory, score, e);
	std::string curs = "Score: " + std::to_string(score);
	std::string sesb = "Session Best: " + std::to_string((float)history[J_BESTSCORE]);
	std::string eveb = "Best Ever: " + std::to_string((float)storedhistory[J_BESTEVSCORE]);

	// TODO: do better
	if (dataset.type == DataLoader::Type::mandlebrot) {
		size_t pre = storedhistory[J_RUNS].size();
		DataLoader::SaveMandleImage(m_path+"images/"+std::to_string(pre)+"_"+std::to_string(e)+".png", predictions, dataset.test_dims[0], dataset.test_dims[1]);
	}

	int size = snprintf(nullptr, 0, "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());

	std::string fmt(size+1, ' ');
	sprintf(fmt.data(), "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());	
	return fmt;
}

template <bool training>
void NeuralNetwork::ForwardProp(float* __restrict x, size_t n) {
    for (size_t i = 0; i < m_layers.size(); i++) {
		// input will always just be previous layers output
		float* __restrict input = i == 0 ? x : m_layers[i-1].Output<training>();

		// does all the fun math stuff for us
		m_layers[i].forward<training>(input, n);
    }
}
void NeuralNetwork::BackProp(const float* __restrict x, const float* __restrict y, size_t n) {

	// compute gradient
	for (ssize_t i = m_layers.size()-1; i >= 0; i--) {

		const float* truth = i == m_layers.size()-1 ? y : m_layers[i+1].Truth();
		const float* input = i == 0 ? x : m_layers[i-1].Output<true>();

		m_layers[i].backward(truth, input, n);
	}

	// update layers
	for (size_t i = 0; i < m_layers.size(); i++) {
		m_layers[i].update(n);
	}
}
