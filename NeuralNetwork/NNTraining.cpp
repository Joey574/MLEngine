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
			const float* x = &dataset.trainData[(i * batch_size) * dataset.trainDataCols];
			const float* y = &dataset.trainLabels[(i * batch_size) * dataset.trainLabelCols];

			// set batch size here to be either batch size or number of elements remaining
			size_t remaining_elements = (dataset.trainDataRows - (i * batch_size));
			size_t effective_size = batch_size > remaining_elements ? remaining_elements : batch_size;

			ForwardProp(true, x, m_batch_data, m_batch_activation_size, effective_size);
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

std::string NeuralNetwork::TestNetwork(const Dataset& dataset, nlohmann::json& history, size_t e) {
	ForwardProp(false, &dataset.testData[0], m_test_data, m_test_activation_size, dataset.testDataRows);
	const float* predications = &m_test_activation[m_test_activation_size - (m_layers.back().nodes*dataset.testDataRows)];

	float score = (*m_metric.metric)(predications, &dataset.testLabels[0], dataset.testDataRows, m_layers.back().nodes);


	SaveBest(history, score, e);
	std::string curs = "Score: " + std::to_string(score);
	std::string sesb = "Session Best: " + std::to_string((float)history[BESTSCORE]);
	std::string eveb = "Best Ever: " + std::to_string((float)m_meta[BESTEVSCORE]);

	int size = snprintf(nullptr, 0, "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());

	std::string fmt(size+1, ' ');
	sprintf(fmt.data(), "%-25s %-30s %-30s", curs.data(), sesb.data(), eveb.data());	
	return fmt;
}

void NeuralNetwork::ForwardProp(bool training, const float* __restrict x, float* __restrict results, size_t actvsize, size_t n) {

	size_t weight_idx = 0;
    size_t bias_idx = 0;

    size_t input_idx = 0;
    size_t output_idx = 0;

    for (size_t i = 0; i < m_layers.size()-1; i++) {

        // set network pointers
        float* weights_start = &m_network[weight_idx];
        float* bias_start = &m_biases[bias_idx];

        // set data pointers
        const float* input_start = i == 0 ? &x[0] : &results[input_idx+actvsize];
        float* output_start = &results[output_idx];

        // initialize memory to bias values to avoid computation and clear existing values
		for (size_t r = 0; r < n; r++) {
			FastCopy(bias_start, &output_start[r*m_layers[i+1].nodes], m_layers[i+1].nodes);
		}	
		
		DotProd(input_start, weights_start, output_start, n, m_layers[i].nodes, m_layers[i].nodes, m_layers[i+1].nodes, false);


        // apply activation, apply next one since input layer doesn't technically have an activation
        (*m_layers[i+1].activation)(output_start, &output_start[actvsize], m_layers[i+1].nodes*n);


        // update pointers
        weight_idx += m_layers[i].nodes * m_layers[i+1].nodes;
        bias_idx += m_layers[i+1].nodes;

        input_idx += i == 0 ? 0 : (m_layers[i].nodes * n);
        output_idx += m_layers[i+1].nodes * n;
    }
}
void NeuralNetwork::BackProp(const float* __restrict x, const float* __restrict y, float lr, size_t n) {

	// adjust learning rate to factor in number of elements
    const float factor = lr / (float)n;
    const __m256 _factor = _mm256_set1_ps(factor);

    // get pointers to last set of activation and dt
    float* last_activation = &m_activation[m_batch_activation_size - (m_layers.back().nodes * n)];
	float* last_d_total = &m_d_total[m_batch_activation_size - (m_layers.back().nodes * n)];


	// compute loss
	(*m_loss.loss)(last_activation, y, last_d_total, n, m_layers.back().nodes);


	int weight_idx = m_weights_size - (m_layers.back().nodes * m_layers[m_layers.size()-2].nodes);
	int d_total_idx = m_batch_activation_size - (m_layers.back().nodes * n);

	// compute d_total
	for (size_t i = m_layers.size() - 2; i > 0; i--) {

		float* weight = &m_network[weight_idx];
		float* prev_total = &m_batch_data[d_total_idx - (m_layers[i].nodes * n)];

		float* cur_d_total = &m_d_total[d_total_idx];
		float* prev_d_total = &m_d_total[d_total_idx - (m_layers[i].nodes * n)];

		DotProdTB(cur_d_total, weight, prev_d_total, n, m_layers[i+1].nodes, m_layers[i].nodes, m_layers[i+1].nodes, true);

		// multiply by derivative of activation function
		(*m_layers[i].derivative)(prev_total, prev_d_total, m_layers[i].nodes*n);

		d_total_idx -= m_layers[i].nodes * n;
		weight_idx -= m_layers[i].nodes * m_layers[i-1].nodes;
	}


	int activation_idx = 0;
	int d_weight_idx = 0;
	int d_bias_idx = 0;

	d_total_idx = 0;

	// compute d_weights
	for (size_t i = 0; i < m_layers.size()-1; i++) {

		const float* prev_activ = i == 0 ? &x[0] : &m_activation[activation_idx];

		float* d_total = &m_d_total[d_total_idx];
		float* d_weights = &m_d_weights[d_weight_idx];
		float* d_bias = &m_d_biases[d_bias_idx];
		
		DotProdTA(prev_activ, d_total, d_weights, n, m_layers[i].nodes, n, m_layers[i+1].nodes, true);

		// compute d_biases
		#pragma omp parallel for
		for (size_t j = 0; j < m_layers[i+1].nodes; j++) {
			__m256 _sum = _mm256_setzero_ps();

			size_t k = 0;
			for (; k <= n-8; k += 8) {
				const __m256 _a = _mm256_loadu_ps(&d_total[j*n+k]);
				_sum = _mm256_add_ps(_sum, _a);
			}

			d_bias[j] = Sum256(_sum);

			for (; k < n; k++) {
				d_bias[j] += d_total[j * n + k];
			}
		}

		d_bias_idx += m_layers[i+1].nodes;
		d_total_idx += m_layers[i+1].nodes * n;
		d_weight_idx += m_layers[i].nodes * m_layers[i+1].nodes;
		activation_idx += i == 0 ? 0 : (m_layers[i].nodes * n);
	}

	// update network (methods for update are the same so we just do weights and biases here)
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

void NeuralNetwork::DropoutFP(uint8_t* __restrict mask, float* __restrict x, size_t n, float dropout) {
		if (dropout <= 0.0f) { return; }
		const float scale = 1.0f / (1.0f - dropout);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::bernoulli_distribution dist(1.0f - dropout);

		#pragma omp parallel for simd
		for (size_t i = 0; i < n; i++) {
			bool k = dist(gen);

			if (k) {
				x[i] *= scale;
				size_t byteidx = i/8;
				size_t bitidx = i%8;

				// store bit mask of dropout results
				#pragma omp atomic update
				mask[byteidx] |= (1 << bitidx);
			} else {
				x[i] = 0.0f;
			}
		}
}
void NeuralNetwork::DropoutBP(uint8_t* __restrict mask, float* __restrict x, size_t n, float dropout) {
	if (dropout <= 0.0f) { return; }
	
	#pragma omp parallel for simd
	for (size_t i = 0; i < n; i++) {
		size_t byteidx = i/8;
		size_t bitidx = i%8;

		// grab bit of data from mask
		bool k = (mask[byteidx] >> bitidx) & 1;
		if (!k) {
			x[i] = 0.0f;
		}
	}
}