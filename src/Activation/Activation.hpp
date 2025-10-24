#pragma once

struct Activation {
public:
    enum Type {
        None, Linear, Sigmoid, ReLU, LeakyReLU, ELU, Softmax
    };

private:
    Type type;
};