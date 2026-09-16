#pragma once // File: cuda_dynamics_lib/include/config.h
//
// A deliberately minimal, dependency-free "key = value" config file reader,
// used to drive the generic runner (runner.cuh) so simulation parameters
// don't have to be hardcoded/recompiled per run. No external JSON/TOML
// library is required.
//
// File format:
//   # comments start with '#' and run to end of line
//   key = value
//   A2 = 0.0, 0.5, 1.0          <- a comma-separated value makes this a
//                                   sweep axis (see enumerate_sweeps below)
//   grid_dims = 1024, 1024      <- EXCEPT grid_dims/grid_min/grid_max,
//                                   which are always fixed-length
//                                   per-dimension vectors, never swept
//
// Duplicate keys: the later occurrence wins, but the key keeps its
// original position for ordering purposes (sweep-suffix order, entries()).

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

class Config {
public:
    Config() = default;

    static Config load(const std::string& filename) {
        Config cfg;
        std::ifstream in(filename);
        if (!in) {
            throw std::runtime_error("Config: could not open '" + filename + "'");
        }
        std::string line;
        int lineno = 0;
        while (std::getline(in, line)) {
            ++lineno;
            size_t hash = line.find('#');
            if (hash != std::string::npos) line = line.substr(0, hash);
            line = trim(line);
            if (line.empty()) continue;

            size_t eq = line.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("Config: " + filename + ":" + std::to_string(lineno) +
                                          ": expected 'key = value', got: '" + line + "'");
            }
            std::string key = trim(line.substr(0, eq));
            std::string value = trim(line.substr(eq + 1));
            if (key.empty()) {
                throw std::runtime_error("Config: " + filename + ":" + std::to_string(lineno) + ": empty key");
            }
            cfg.set(key, value);
        }
        return cfg;
    }

    bool has(const std::string& key) const {
        return find(key) != entries_.end();
    }

    // Overwrites the key's value if present (keeping its original
    // position), otherwise appends it.
    void set(const std::string& key, const std::string& value) {
        auto it = find(key);
        if (it != entries_.end()) it->second = value;
        else entries_.emplace_back(key, value);
    }

    std::string get_string(const std::string& key) const { return raw(key); }
    std::string get_string(const std::string& key, const std::string& default_value) const {
        return has(key) ? raw(key) : default_value;
    }

    double get_double(const std::string& key) const {
        return parse_double(key, raw(key));
    }
    double get_double(const std::string& key, double default_value) const {
        return has(key) ? get_double(key) : default_value;
    }

    int get_int(const std::string& key) const {
        return parse_int(key, raw(key));
    }
    int get_int(const std::string& key, int default_value) const {
        return has(key) ? get_int(key) : default_value;
    }

    // Splits a comma-separated value, e.g. for grid_dims/grid_min/grid_max.
    std::vector<double> get_double_list(const std::string& key) const {
        std::vector<double> out;
        for (const auto& tok : split_csv(raw(key))) out.push_back(parse_double(key, tok));
        return out;
    }
    std::vector<int> get_int_list(const std::string& key) const {
        std::vector<int> out;
        for (const auto& tok : split_csv(raw(key))) out.push_back(parse_int(key, tok));
        return out;
    }

    // Ordered (key, raw_value) pairs, in file order (first-occurrence
    // position, last-occurrence value). Used by enumerate_sweeps().
    const std::vector<std::pair<std::string, std::string>>& entries() const { return entries_; }

    static std::string trim(const std::string& s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return "";
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    }

    static std::vector<std::string> split_csv(const std::string& s) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, ',')) out.push_back(trim(tok));
        return out;
    }

private:
    std::vector<std::pair<std::string, std::string>> entries_;

    std::vector<std::pair<std::string, std::string>>::iterator find(const std::string& key) {
        return std::find_if(entries_.begin(), entries_.end(),
                             [&](const auto& e) { return e.first == key; });
    }
    std::vector<std::pair<std::string, std::string>>::const_iterator find(const std::string& key) const {
        return std::find_if(entries_.begin(), entries_.end(),
                             [&](const auto& e) { return e.first == key; });
    }

    std::string raw(const std::string& key) const {
        auto it = find(key);
        if (it == entries_.end()) {
            throw std::runtime_error("Config: missing required key '" + key + "'");
        }
        return it->second;
    }

    static double parse_double(const std::string& key, const std::string& value) {
        try {
            size_t pos;
            double v = std::stod(value, &pos);
            if (pos != value.size()) throw std::invalid_argument("trailing characters");
            return v;
        } catch (const std::exception&) {
            throw std::runtime_error("Config: '" + key + "' is not a number: '" + value + "'");
        }
    }
    static int parse_int(const std::string& key, const std::string& value) {
        try {
            size_t pos;
            int v = std::stoi(value, &pos);
            if (pos != value.size()) throw std::invalid_argument("trailing characters");
            return v;
        } catch (const std::exception&) {
            throw std::runtime_error("Config: '" + key + "' is not an integer: '" + value + "'");
        }
    }
};

// One resolved combination from a parameter sweep: `config` has a single
// value for every key (ready for get_double()/get_string()/load_params_for),
// and `suffix` names which combination this is (e.g. "A2_0.5000_A3_0.0000"),
// or "" if nothing was swept.
struct SweepResult {
    Config config;
    std::string suffix;
};

// Any key whose value contains a comma becomes a sweep axis (its values
// tried one at a time), EXCEPT grid_dims/grid_min/grid_max, which are
// always fixed-length per-dimension vectors rather than sweep lists.
// Returns the cartesian product of all sweep axes, in the axes' file
// order -- a single-element vector (suffix "") if nothing is swept.
inline std::vector<SweepResult> enumerate_sweeps(const Config& cfg) {
    static const std::vector<std::string> vector_keys = {"grid_dims", "grid_min", "grid_max"};

    struct Axis {
        std::string key;
        std::vector<std::string> values;
    };
    std::vector<Axis> axes;
    for (const auto& e : cfg.entries()) {
        if (std::find(vector_keys.begin(), vector_keys.end(), e.first) != vector_keys.end()) continue;
        if (e.second.find(',') == std::string::npos) continue;
        axes.push_back({e.first, Config::split_csv(e.second)});
    }

    std::vector<SweepResult> results;
    if (axes.empty()) {
        results.push_back({cfg, ""});
        return results;
    }

    std::vector<size_t> idx(axes.size(), 0);
    while (true) {
        Config combo = cfg;
        std::ostringstream suffix;
        for (size_t a = 0; a < axes.size(); ++a) {
            const std::string& chosen = axes[a].values[idx[a]];
            combo.set(axes[a].key, chosen);
            if (a > 0) suffix << "_";
            suffix << axes[a].key << "_" << std::fixed << std::setprecision(4) << std::stod(chosen);
        }
        results.push_back({combo, suffix.str()});

        int carry = static_cast<int>(axes.size()) - 1;
        while (carry >= 0) {
            if (++idx[carry] < axes[carry].values.size()) break;
            idx[carry] = 0;
            --carry;
        }
        if (carry < 0) break;
    }
    return results;
}

// The interface a new dynamical system's Params struct must implement to
// plug into the generic runner (runner.cuh): a full specialization that
// builds a ParamsType from a resolved (single-value-per-key) Config, e.g.
//
//   template <> inline HortonSystemParams load_params_for<HortonSystemParams>(const Config& cfg) {
//       HortonSystemParams p;
//       p.A1 = cfg.get_double("A1", 1.0);
//       ...
//       return p;
//   }
//
// defined in the system's own header (maps/horton.h). Must be a template
// specialization (not a plain overload) so the generic runner's dependent
// call `load_params_for<ParamsType>(cfg)` resolves correctly regardless of
// header include order.
template <typename ParamsType>
ParamsType load_params_for(const Config& cfg);
