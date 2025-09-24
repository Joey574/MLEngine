#pragma once

struct Activation {
public:
    enum Type {
        None, Linear, Sigmoid, ReLU, LeakyReLU, ELU, Softmax
    };

    void (*ApplyActivation)(const Tensor<float>&, Tensor<float>&);
    void (*ApplyDerivative)(const Tensor<float>&, Tensor<float>&);


    Activation(Type type = Type::None) {
        this->type = type;
    }

    inline Type ActivationType() const { return type; }

    static Type ParseType(const std::string& name);
    static std::string ParseName(Type type);

private:
    Type type;

    
    static void Linear(const Tensor<float>& x, Tensor<float>& y);
    static void Sigmoid(const Tensor<float>& x, Tensor<float>& y);
    static void ReLU(const Tensor<float>& x, Tensor<float>& y);
    static void LeakyReLU(const Tensor<float>& x, Tensor<float>& y);
    static void ELU(const Tensor<float>& x, Tensor<float>& y);
    static void Softmax(const Tensor<float>& x, Tensor<float>& y);

    static void LinearDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void SigmoidDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void ReLUDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void LeakyReLUDerivative(const Tensor<float>& x, Tensor<float>& y);
    static void ELUDerivative(const Tensor<float>& x, Tensor<float>& y);
};