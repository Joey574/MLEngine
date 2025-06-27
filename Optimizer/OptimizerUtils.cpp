#include "Optimizer.hpp"

std::string Optimizer::ParseRegName(Regularization reg) {
    switch (reg) {
        case Regularization::l1:
            return "l1";
        case Regularization::l2:
            return "l2";
        default:
            return "none";
    }
}
std::string Optimizer::ParseUpdName(Update upd) {
    switch (upd) {
        case Update::sgd:
            return "sgd";
        case Update::momentumsgd:
            return "momentumsgd";
        case Update::rmsprop:
            return "rmsprop";
        case Update::adam:
            return "adam";
        default:
            return "none";
    }
}

Optimizer::Regularization Optimizer::ParseRegType(const std::string& reg) {
    if (reg == "l1") {
        return Regularization::l1;
    } else if (reg == "l2") {
        return Regularization::l2;
    }

    return Regularization::none;
}
Optimizer::Update Optimizer::ParseUpdType(const std::string& upd) {
    if (upd == "sgd") {
        return Update::sgd;
    } else if (upd == "momentumsgd") {
        return Update::momentumsgd;
    } else if (upd == "rmsprop") {
        return Update::rmsprop;
    } else if (upd == "adam") {
        return Update::adam;
    }

    return Update::none;
}

void Optimizer::Define(YAML::Node config) {
    std::string upd = config[Y_OPT_TYPE].as<std::string>("sgd");
    m_update = ParseUpdType(upd);

    m_lr = config[Y_OPT_LEARNINGRATE].as<float>(0.1f);

    if (config[Y_OPT_REGULARIZATION]) {
        std::string reg = config[Y_OPT_REGULARIZATION].as<std::string>();
        
        m_reg_lambda = config[Y_OPT_REGLAMBDA].as<float>(0.0001f);
        m_reg = ParseRegType(reg);
    }

    switch (m_update) {
        case Update::momentumsgd:
            m_m_coef = config[Y_OPT_MOMENTUM].as<float>(0.9f);
            break;
        case Update::rmsprop:
            m_r_decay = config[Y_OPT_DECAY].as<float>(0.9f);
            m_r_epsl = config[Y_OPT_EPSL].as<float>(0.000001f);
            break;
        case Update::adam:
            m_a_t = 1;
            m_a_b1 = config[Y_OPT_B1].as<float>(0.9f);
            m_a_b2 = config[Y_OPT_B2].as<float>(0.999f);
            m_a_epsl = config[Y_OPT_EPSL].as<float>(0.000001f);
            break;
    }

    AssignPtr();
}
void Optimizer::Initialize(float* dw, float* db, char* data, size_t wsize, size_t bsize) {
    size_t offset = 0;
    m_s_dw = dw;
    m_s_db = db;

    switch (m_update) {
        case Update::sgd:
            break;
        case Update::momentumsgd:
            m_m_vw = (float*)(data+offset);
            offset += RoundTo(32, wsize*sizeof(float));
            
            m_m_vb = (float*)(data+offset);
            offset += RoundTo(32, bsize*sizeof(float));
            break;

        case Update::rmsprop:
            m_r_gw = (float*)(data+offset);
            offset += RoundTo(32, wsize*sizeof(float));

            m_r_gb = (float*)(data+offset);
            offset += RoundTo(32, bsize*sizeof(float));
            break;

        case Update::adam:
            m_a_wm = (float*)(data+offset);
            offset += RoundTo(32, wsize*sizeof(float));

            m_a_wv = (float*)(data+offset);
            offset += RoundTo(32, wsize*sizeof(float));

            m_a_bm = (float*)(data+offset);
            offset += RoundTo(32, bsize*sizeof(float));

            m_a_bv = (float*)(data+offset);
            offset += RoundTo(32, bsize*sizeof(float));
    }
}

void Optimizer::AssignPtr() {
    if (m_update == Update::sgd) {
        if (m_reg == Regularization::l1) {
            update = static_cast<UpdateFn>(&Optimizer::SGD<Regularization::l1>);
        } else if (m_reg == Regularization::l2) {
            update = static_cast<UpdateFn>(&Optimizer::SGD<Regularization::l2>);
        } else {
            update = static_cast<UpdateFn>(&Optimizer::SGD<Regularization::none>);
        }
    } else if (m_update == Update::momentumsgd) {
        if (m_reg == Regularization::l1) {
            update = static_cast<UpdateFn>(&Optimizer::MomentumSGD<Regularization::l1>);
        } else if (m_reg == Regularization::l2) {
            update = static_cast<UpdateFn>(&Optimizer::MomentumSGD<Regularization::l2>);
        } else {
            update = static_cast<UpdateFn>(&Optimizer::MomentumSGD<Regularization::none>);
        }
    } else if (m_update == Update::rmsprop) {
        update = static_cast<UpdateFn>(&Optimizer::RMSProp);
    } else if (m_update == Update::adam) {
        update = static_cast<UpdateFn>(&Optimizer::Adam);
    }
}

size_t Optimizer::Size(size_t wsize, size_t bsize) {
    size_t size = 0;

    switch (m_update) {
        case Update::sgd:
            break;
        case Update::momentumsgd:
            size += RoundTo(32, wsize*sizeof(float));
            size += RoundTo(32, bsize*sizeof(float));
            break;
        
        case Update::rmsprop:
            size += RoundTo(32, wsize*sizeof(float));
            size += RoundTo(32, bsize*sizeof(float));
            break;

        case Update::adam:
            size += RoundTo(32, wsize*sizeof(float));
            size += RoundTo(32, wsize*sizeof(float));

            size += RoundTo(32, bsize*sizeof(float));
            size += RoundTo(32, bsize*sizeof(float));
            break;
    }

    return size;
}

/// @brief only works with powers of 2
size_t Optimizer::RoundTo(size_t alignment, size_t n) {
        alignment--;
        return (n+alignment) & ~alignment;
}