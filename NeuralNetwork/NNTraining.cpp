#include "NeuralNetwork.hpp"

nlohmann::json NeuralNetwork::Fit(Dataset& dataset, size_t batch_size, size_t epochs, float learning_rate, int validation_freq, float validation_split, bool shuffle) {
	auto fitstart = std::chrono::high_resolution_clock::now();

	std::cout << m_meta.dump(4) << "\n";
    
	nlohmann::json history;
	FitStart(history, epochs, batch_size, learning_rate);

	InitializeBatchData(batch_size);
	InitializeTestData(dataset.testDataRows);

	const size_t iterations = (dataset.trainDataRows + (batch_size-1)) / batch_size;

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

			ForwardProp(true, x, m_batch_data, m_batch_actv, effective_size);
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
	ForwardProp(false, dataset.testData.data(), m_test_data, m_test_actv, dataset.testDataRows);
	const float* predications = &m_test_actv[m_test_actv_size - (m_layers.back().nodes*dataset.testDataRows)];

	float score = m_layers.back().lossmetric.metric(predications, &dataset.testLabels[0], dataset.testDataRows, m_layers.back().nodes);

	SaveBest(history, score, e);
	std::string curs = "Score: " + std::to_string(score);
	std::string sesb = "Session Best: " + std::to_string((float)history[BESTSCORE]);
	std::string eveb = "Best Ever: " + std::to_string((float)m_meta[BESTEVSCORE]);

	int size = snprintf(nullptr, 0, "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());

	std::string fmt(size+1, ' ');
	sprintf(fmt.data(), "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());	
	return fmt;
}

void NeuralNetwork::ForwardProp(bool training, float* __restrict x, float* __restrict z, float* __restrict a, size_t n) {
	
	size_t ouidx = 0;

	float* __restrict tz = x;
	float* __restrict ta = x;

    for (size_t i = 0; i < m_layers.size(); i++) {
		// input will always just be previous layers activation
		const float* __restrict input = ta;

		// set new outputs
		tz = i == 0 ? x : &z[ouidx];
		ta = i == 0 ? x : &a[ouidx];

		// does all the fun math stuff for us
		m_layers[i].forward(training, input, tz, ta, n);

		// update offsets
		ouidx += i == 0 ? 0 : n*m_layers[i].nodes;
    }
}
void NeuralNetwork::BackProp(const float* __restrict x, const float* __restrict y, float lr, size_t n) {

	size_t aidx = m_batch_actv_size-(n*m_layers.back().nodes);
	size_t widx = m_weights_size-(m_layers.back().size);
	size_t bidx = m_biases_size-(m_layers.back().nodes);

	for (size_t i = m_layers.size()-1; i > 0; i--) {

		// build pointers to relevent data
		const float* __restrict a = &m_batch_actv[aidx];
		const float* __restrict z = &m_batch_data[aidx];

		const float* __restrict truth = i == m_layers.size()-1 ? y : &m_d_total[aidx+n*m_layers[i].nodes];
		const float* __restrict pa = i <= 1 ? x : &m_batch_actv[aidx-n*m_layers[i-1].nodes];
		const float* __restrict nw = i == m_layers.size()-1 ? nullptr : &m_network[widx+m_layers[i].size];
		
		float* __restrict dt = &m_d_total[aidx];
		float* __restrict dw = &m_d_weights[widx];
		float* __restrict db = &m_d_biases[bidx];

		m_layers[i].backward(truth, pa, z, a, nw, dt, dw, db, n);

		// update offsets
		aidx -= n*m_layers[i-1].nodes;
		widx -= m_layers[i-1].size;
		bidx -= m_layers[i-1].nodes;
	}

	// adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);

	// update network (bias and weights currently use same formula to update, so both happen here)
	#pragma omp parallel for
	for (size_t i = 0; i <= m_network_size-8; i += 8) {
		const __m256 _a = _mm256_loadu_ps(&m_d_weights[i]);
		const __m256 _b = _mm256_loadu_ps(&m_network[i]);
		const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _b);

		_mm256_storeu_ps(&m_network[i], _res);
	}

	for (size_t i = m_network_size-(m_network_size%8); i < m_network_size; i++) {
		m_network[i] -= m_d_weights[i] * factor;
	}
}
