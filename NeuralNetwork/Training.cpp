#include "NeuralNetwork.hpp"

void NeuralNetwork::initialize(std::vector<size_t> dimensions, std::vector<ActivationFunctions> activations, LossMetric loss, LossMetric metric, WeightInitialization weightInit) {
    error = "";
    m_weight_init = weightInit;
    m_loss.type = loss;
    m_metric.type = metric;

    // main initialization of all the internal goodies
    if (dimensions.size() != activations.size()) {
        error = "Dimensions and Activations must match in length";
        return;
    }

    m_layers = std::vector<Layer>(dimensions.size());
    for (size_t i = 0; i < m_layers.size(); i++) {
        m_layers[i].nodes = dimensions[i];

    }

    initialize_network();
}

NeuralNetwork::History NeuralNetwork::fit(float* x_train, float* y_train, float* x_valid, float* y_valid, size_t training_elements, size_t valid_elements, size_t batch_size, size_t epochs, float learning_rate, int validation_freq, float validation_split, bool shuffle) {
    error = "";
    
    History history;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> time;

    // TODO data preprocess

    const size_t iterations = training_elements / batch_size;
    // initialize batch and test pointers to match batch size

    for (size_t e = 0; e < epochs; e++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < iterations; i++) {
            
            // adjust pointer to relevent data
            // TODO complex math bs
            // figure out how to structure matrix to avoid dotprod_t_b
            // will require different values for constructing indexes
            // pretty sure we just need to flip the matrix but getting data would be more difficult :/
            // might be best to create a block big enough for the batch, copy in, then transpose there
            // would avoid modifying originally passed data
            float* x = &x_train[(i * batch_size) * 0];
            float* y = &y_train[(i * batch_size) * 0];

            forward_prop(x, m_batch_data, m_batch_activation_size, batch_size);
            //back_prop(x, y, learning_rate, batch_size);
        }
    }

    return history;
}

void NeuralNetwork::forward_prop(float* x_data, float* result_data, size_t activation_size, size_t num_elements) {
    size_t weight_idx = 0;
    size_t bias_idx = 0;

    size_t input_idx = 0;
    size_t output_idx = 0;

    for (size_t i = 0; i < m_layers.size()-1; i++) {

        // set network pointers
        float* weights_start = &m_network[weight_idx];
        float* bias_start = &m_biases[bias_idx];

        // set data pointers
        float* input_start = i == 0 ? &x_data[0] : &result_data[input_idx+activation_size];
        float* output_start = &result_data[output_idx];

        // initialize memory to bias values to avoid computation and clear existing values
        #pragma omp parallel for
        for (size_t r = 0; r < m_layers[i+1].nodes; r++) {
            std::fill(&output_start[r * num_elements], &output_start[r*num_elements+num_elements], bias_start[r]);
        }

        dot_prod(weights_start, input_start, output_start, m_layers[i+1].nodes, m_layers[i].nodes, m_layers[i].nodes, num_elements, false);

        // apply activation
        (this->*m_layers[i].activation)(output_start, &output_start[activation_size], m_layers[i+1].nodes*num_elements);

        // update pointers
        weight_idx += m_layers[i].nodes * m_layers[i+1].nodes;
        bias_idx += m_layers[i+1].nodes;

        input_idx += i == 0 ? 0 : (m_layers[i].nodes * num_elements);
        output_idx += m_layers[i+1].nodes * num_elements;
    }
}
void NeuralNetwork::back_prop(float* x_data, float* y_data, float learning_rate, size_t num_elements) {
    // adjust learning rate tp factor in number of elements
    const float factor = learning_rate / (float)num_elements;
    const __m256 _factor = _mm256_set1_ps(factor);

    /*
		d_total[i - 1] := weight[i].T.dot(d_total[i]) * total[i - 1].activ_derivative
		d_weights[i] := d_total[i].dot(x || activation[i - 1].T)
		d_biases[i] := d_total[i].row_sums
	*/

    // get pointers to last set of activation and dt
    float* last_activation = &m_activation[m_batch_activation_size - (m_layers.back().nodes * num_elements)];
	float* last_d_total = &m_d_total[m_batch_activation_size - (m_layers.back().nodes * num_elements)];

	// compute loss
	(this->*m_loss.loss)(last_activation, y_data, last_d_total, m_layers.back().nodes, num_elements);

	int weight_idx = m_weights_size - (m_layers.back().nodes * m_layers[m_layers.size() - 2].nodes);
	int d_total_idx = m_batch_activation_size - (m_layers.back().nodes * num_elements);

	// compute d_total
	for (size_t i = m_layers.size() - 2; i > 0; i--) {

		float* weight = &m_network[weight_idx];
		float* prev_total = &m_batch_data[d_total_idx - (m_layers[i].nodes * num_elements)];

		float* cur_d_total = &m_d_total[d_total_idx];
		float* prev_d_total = &m_d_total[d_total_idx - (m_layers[i].nodes * num_elements)];

		dot_prod_t_a(weight, cur_d_total, prev_d_total, m_layers[i + 1].nodes, m_layers[i].nodes, m_layers[i+1].nodes, num_elements, true);

		(this->*m_layers[i - 1].derivative)(prev_total, prev_d_total, m_layers[i].nodes * num_elements);

		d_total_idx -= m_layers[i].nodes * num_elements;
		weight_idx -= m_layers[i].nodes * m_layers[i-1].nodes;
	}

	int activation_idx = 0;
	int d_weight_idx = 0;
	int d_bias_idx = 0;

	d_total_idx = 0;

	// compute d_weights
	for (size_t i = 0; i < m_layers.size() - 1; i++) {

		float* prev_activ = i == 0 ? &x_data[0] : &m_activation[activation_idx];

		float* d_total = &m_d_total[d_total_idx];
		float* d_weights = &m_d_weights[d_weight_idx];
		float* d_bias = &m_d_biases[d_bias_idx];

		i == 0 ?
			dot_prod(d_total, prev_activ, d_weights, m_layers[i+1].nodes, num_elements, num_elements, m_layers[i].nodes, true) :
			dot_prod_t_b(d_total, prev_activ, d_weights, m_layers[i+1].nodes, num_elements, m_layers[i].nodes, num_elements, true);

		// compute d_biases
		#pragma omp parallel for
		for (size_t j = 0; j < m_layers[i+1].nodes; j++) {
			__m256 sum = _mm256_setzero_ps();

			size_t k = 0;
			for (; k <= num_elements - 8; k += 8) {
				sum = _mm256_add_ps(sum, _mm256_load_ps(&d_total[j * num_elements + k]));
			}

			float t[8];
			_mm256_store_ps(t, sum);
			d_bias[j] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];

			for (; k < num_elements; k++) {
				d_bias[j] += d_total[j * num_elements + k];
			}
		}

		d_bias_idx += m_layers[i + 1].nodes;
		d_total_idx += m_layers[i+1].nodes * num_elements;
		d_weight_idx += m_layers[i].nodes * m_layers[i+1].nodes;
		activation_idx += i == 0 ? 0 : (m_layers[i].nodes * num_elements);
	}


	// update weights
	#pragma omp parallel for
	for (size_t i = 0; i <= m_weights_size - 8; i += 8) {
        const __m256 _a = _mm256_load_ps(&m_d_weights[i]);
        const __m256 _c = _mm256_load_ps(&m_network[i]);
        const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _c);

        _mm256_store_ps(&m_network[i], _res);
	}

	for (size_t i = m_weights_size - (m_weights_size % 8); i < m_weights_size; i++) {
		m_network[i] -= m_d_weights[i] * factor;
	}


	// update biases
	#pragma omp parallel for
	for (size_t i = 0; i <= m_biases_size - 8; i += 8) {
        const __m256 _a = _mm256_load_ps(&m_d_biases[i]);
        const __m256 _c = _mm256_load_ps(&m_biases[i]);
        const __m256 _res = _mm256_fnmadd_ps(_a, _factor, _c);

        _mm256_store_ps(&m_biases[i], _res);
	}

	for (size_t i = m_biases_size - (m_biases_size % 8); i < m_biases_size; i++) {
		m_biases[i] -= m_d_biases[i] * factor;
	}
}
