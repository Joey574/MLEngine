#pragma once

struct Activation {
  public:
    enum class Type { None, Linear, Sigmoid, ReLU, LeakyReLU, ELU, Softmax };

    static inline Type ParseType(const std::string& name) {
        auto lower = std::string(name.size(), ' ');
        std::transform(name.begin(), name.end(), lower.begin(), tolower);

        if (lower == "linear") {
            return Type::Linear;
        } else if (lower == "sigmoid") {
            return Type::Sigmoid;
        } else if (lower == "relu") {
            return Type::ReLU;
        } else if (lower == "leakyrelu") {
            return Type::LeakyReLU;
        } else if (lower == "elu") {
            return Type::ELU;
        } else if (lower == "softmax") {
            return Type::Softmax;
        } else {
            return Type::None;
        }
    }
    static inline std::string ParseName(const Type type) {
        switch (type) {
            case Type::None:
                return "None";
            case Type::Linear:
                return "Linear";
            case Type::Sigmoid:
                return "Sigmoid";
            case Type::ReLU:
                return "ReLU";
            case Type::LeakyReLU:
                return "LeakyReLU";
            case Type::ELU:
                return "ELU";
            case Type::Softmax:
                return "Softmax";
            default:
                return "";
        }
    }

    inline Type GetType() const { return type; }
    inline void AssignPointers(const std::string& name) { AssignPointers(ParseType(name)); }
    inline void AssignPointers(const Type type) {
        this->type = type;

        switch (type) {
            case Type::Linear:
                activation = Linear;
                derivative = LinearDerivative;
                return;
            case Type::Sigmoid:
                activation = Sigmoid;
                derivative = SigmoidDerivative;
                return;
            case Type::ReLU:
                activation = ReLU;
                derivative = ReLUDerivative;
                return;
            case Type::LeakyReLU:
                activation = LeakyReLU;
                derivative = LeakyReLUDerivative;
                return;
            case Type::ELU:
                activation = ELU;
                derivative = ELUDerivative;
                return;
            case Type::Softmax:
                activation = Softmax;
                derivative = nullptr;
                return;
            default:
                activation = nullptr;
                derivative = nullptr;
                return;
        }
    }

    void (*activation)(const Tensor<float>&, Tensor<float>&);
    void (*derivative)(const Tensor<float>&, Tensor<float>&);

  private:
    Type type;

    /* ----------
    activation functions
    ---------- */
    static void Linear(const Tensor<float>& x, Tensor<float>& y);
    static void Sigmoid(const Tensor<float>& x, Tensor<float>& y);
    static void ReLU(const Tensor<float>& x, Tensor<float>& y);
    static void LeakyReLU(const Tensor<float>& x, Tensor<float>& y);
    static void ELU(const Tensor<float>& x, Tensor<float>& y);
    static void Softmax(const Tensor<float>& x, Tensor<float>& y);

    /* ----------
    derivative functions
    ---------- */
    static void LinearDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void SigmoidDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void ReLUDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void LeakyReLUDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void ELUDerivative(const Tensor<float>& x, Tensor<float>& y);
};