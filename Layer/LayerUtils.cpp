#include "Layer.hpp"

std::string Layer::ParseName(LayerType type) {
    switch (type) {
        case LayerType::input:
            return "input";
        case LayerType::hidden:
            return "hidden";
        case LayerType::output:
            return "output";
        case LayerType::conv2D:
            return "conv2D";
        case LayerType::conv3D:
            return "conv3D";
        case LayerType::none:
            return "none";
        default:
            std::cerr << "[-] Invalid Layer Type: " << (int)type << "\n";
            return "none";
    }
}
Layer::LayerType Layer::ParseType(const std::string& type) {
    if (type == "input") {
        return LayerType::input;
    } else if (type == "hidden") {
        return LayerType::hidden;
    } else if (type == "output") {
        return LayerType::output;
    } else if (type == "conv2D") {
        return LayerType::conv2D;
    } else if (type == "conv3D") {
        return LayerType::conv3D;
    } else if (type == "none") {
        return LayerType::none;
    } else {
        std::cerr << "[-] Invalid Layer Type: " << type << "\n";
        return LayerType::none;
    }
}

std::string Layer::CleanSize(size_t bytes) {
    long double dbytes = bytes;
    const double gb = 1e9;
    const double mb = 1e6;
    const double kb = 1e3;

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);

    if (dbytes / gb > 1.00) {
        oss << dbytes / gb << " gb";
    } else if (dbytes / mb > 1.00) {
        oss << dbytes / mb << " mb";
    } else if (dbytes / kb > 1.00) {
        oss << dbytes / kb << " kb";
    } else {
        oss << dbytes << " b";
    }

    return oss.str();
}
std::string Layer::StartEndTotal(size_t offset, size_t start, size_t end) {
    std::string sstart = CleanSize(start-offset);
    std::string send = CleanSize(end-offset);
    std::string ssize = CleanSize(end-start);

    return sstart+" - "+send+" ("+ssize+")";
}

std::string Layer::VisualizeNet() {
    std::string res = "\n\t\t"+std::to_string(inodes)+"x"+std::to_string(nodes);

    if (m_w) {
        res += "\n\t\tm_w: "+StartEndTotal((size_t)m_net, (size_t)m_w, (size_t)m_w+m_w_bytes);
    }

    if (m_b) {
        res += "\n\t\tm_b: "+StartEndTotal((size_t)m_net, (size_t)m_b, (size_t)m_b+m_b_bytes);
    }

    return res;
}
std::string Layer::VisualizeBatch() {
    std::string res = "";

    if (m_z) {
        res += "\n\t\tm_z: "+StartEndTotal((size_t)m_batch, (size_t)m_z, (size_t)m_z+m_z_bytes);
    }

    if (m_a) {
        res += "\n\t\tm_a: "+StartEndTotal((size_t)m_batch, (size_t)m_a, (size_t)m_a+m_a_bytes);
    }

    if (m_dt) {
        res += "\n\t\tm_dt: "+StartEndTotal((size_t)m_batch, (size_t)m_dt, (size_t)m_dt+m_dt_bytes);
    }

    if (m_dw) {
        res += "\n\t\tm_dw: "+StartEndTotal((size_t)m_batch, (size_t)m_dw, (size_t)m_dw+m_dw_bytes);
    }

    if (m_db) {
        res += "\n\t\tm_db: "+StartEndTotal((size_t)m_batch, (size_t)m_db, (size_t)m_db+m_db_bytes);
    }

    return res;
}
std::string Layer::VisualizeTest() {
    std::string res = "";

    if (m_tz) {
        res += "\n\t\tm_tz: "+StartEndTotal((size_t)m_test, (size_t)m_tz, (size_t)m_tz+m_tz_bytes);
    }

    if (m_ta) {
        res += "\n\t\tm_ta: "+StartEndTotal((size_t)m_test, (size_t)m_ta, (size_t)m_ta+m_ta_bytes);
    }

    return res;
}
