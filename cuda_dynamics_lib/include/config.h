#pragma once
/**
 * @file config.h
 * @brief Dependency-free "key = value" config file reader, parameter
 * sweeps, and the load_params_for<> extension point new systems implement.
 *
 * No external JSON/TOML library is required or used. File format:
 * @code
 *   # comments start with '#' and run to end of line
 *   key = value
 *   A2 = 0.0, 0.5, 1.0          # a comma-separated value makes this a
 *                                # sweep axis (see enumerate_sweeps())
 *   grid_dims = 1024, 1024      # EXCEPT grid_dims/grid_min/grid_max,
 *                                # which are always fixed-length
 *                                # per-dimension vectors, never swept
 * @endcode
 * Duplicate keys: the later occurrence wins, but the key keeps its
 * original position for ordering purposes (sweep-suffix order, entries()).
 */

#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iomanip>

/**
 * @brief An in-memory, ordered "key = value" store loaded from a config
 * file, with typed accessors (get_double(), get_int(), ...) and default
 * values. See config.h's file-level docs for the on-disk format.
 *
 * A Config is small and cheap to copy -- enumerate_sweeps() returns one
 * full copy per sweep combination, each with a single value substituted
 * for every swept key.
 */
class Config {
public:
    Config() = default;

    /**
     * @brief Parses a config file from disk.
     * @param filename Path to the config file.
     * @return The parsed key/value store.
     * @throws std::runtime_error if the file can't be opened, or a
     * non-blank/non-comment line isn't of the form `key = value`.
     */
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

    /** @brief Whether `key` is present. */
    bool has(const std::string& key) const {
        return find(key) != entries_.end();
    }

    /**
     * @brief Sets `key` to `value`, overwriting it in place if already
     * present (keeping its original file-order position), else appending.
     */
    void set(const std::string& key, const std::string& value) {
        auto it = find(key);
        if (it != entries_.end()) it->second = value;
        else entries_.emplace_back(key, value);
    }

    /** @brief Returns `key`'s raw string value. @throws std::runtime_error if missing. */
    std::string get_string(const std::string& key) const { return raw(key); }
    /** @brief Returns `key`'s raw string value, or `default_value` if `key` is absent. */
    std::string get_string(const std::string& key, const std::string& default_value) const {
        return has(key) ? raw(key) : default_value;
    }

    /** @brief Parses `key`'s value as a double. @throws std::runtime_error if missing or not a number. */
    double get_double(const std::string& key) const {
        return parse_double(key, raw(key));
    }
    /** @brief Parses `key`'s value as a double, or returns `default_value` if `key` is absent. */
    double get_double(const std::string& key, double default_value) const {
        return has(key) ? get_double(key) : default_value;
    }

    /** @brief Parses `key`'s value as an int. @throws std::runtime_error if missing or not an integer. */
    int get_int(const std::string& key) const {
        return parse_int(key, raw(key));
    }
    /** @brief Parses `key`'s value as an int, or returns `default_value` if `key` is absent. */
    int get_int(const std::string& key, int default_value) const {
        return has(key) ? get_int(key) : default_value;
    }

    /**
     * @brief Splits `key`'s value on commas and parses each token as a
     * double, e.g. for `grid_min`/`grid_max` (one entry per dimension).
     * @throws std::runtime_error if missing or any token isn't a number.
     */
    std::vector<double> get_double_list(const std::string& key) const {
        std::vector<double> out;
        for (const auto& tok : split_csv(raw(key))) out.push_back(parse_double(key, tok));
        return out;
    }
    /** @brief Same as get_double_list(), but parses each token as an int. */
    std::vector<int> get_int_list(const std::string& key) const {
        std::vector<int> out;
        for (const auto& tok : split_csv(raw(key))) out.push_back(parse_int(key, tok));
        return out;
    }

    /**
     * @brief All (key, raw_value) pairs, in file order (first-occurrence
     * position, last-occurrence value). Used by enumerate_sweeps() to find
     * sweep axes; rarely needed directly.
     */
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

/**
 * @brief One resolved combination from a parameter sweep (see
 * enumerate_sweeps()).
 */
struct SweepResult {
    /** A single value for every key -- ready for get_double()/get_string()/load_params_for(). */
    Config config;
    /** Names this combination, e.g. `"A2_0.5000_A3_0.0000"`, or `""` if nothing was swept. */
    std::string suffix;
};

/**
 * @brief Expands a config's comma-separated values into every combination
 * of a parameter sweep.
 *
 * Any key whose value contains a comma becomes a sweep axis (its values
 * tried one at a time), EXCEPT `grid_dims`/`grid_min`/`grid_max`, which are
 * always fixed-length per-dimension vectors rather than sweep lists.
 *
 * @param cfg The base config, as loaded from a file.
 * @return The cartesian product of all sweep axes, in the axes' file
 * order -- a single-element vector (with `suffix == ""`) if nothing in
 * `cfg` was swept.
 */
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

/**
 * @brief The parameter-loading half of the interface a new dynamical
 * system must implement to plug into the generic runner (see runner.cuh
 * and TUTORIAL.md).
 *
 * A new system's own header must provide a full template *specialization*
 * of this function for its `ParamsType`, building that struct from a
 * resolved (single-value-per-key) Config -- typically one `cfg.get_double(
 * "name", default_value)` call per field:
 * @code
 *   template <>
 *   inline HortonSystemParams load_params_for<HortonSystemParams>(const Config& cfg) {
 *       HortonSystemParams p;
 *       p.A1 = cfg.get_double("A1", 1.0);
 *       // ... one field at a time ...
 *       return p;
 *   }
 * @endcode
 * (defined in maps/horton.h; see also maps/standard_map.h and
 * maps/henon_heiles.h for two more worked examples). This lets
 * `run_ode_generic`/`run_map_generic` build any system's params struct
 * from a config file without knowing its field names.
 *
 * @note Must be a template *specialization*, not a plain overload of the
 * same name -- the generic runner calls `load_params_for<ParamsType>(cfg)`,
 * a dependent name resolved via template specialization lookup at the
 * point of instantiation (not ordinary/ADL lookup), so a same-named
 * overload in the wrong place would silently not be found.
 *
 * @tparam ParamsType The system's parameter struct type (e.g. `HortonSystemParams`).
 * @param cfg A resolved config (one value per key; see enumerate_sweeps()).
 * @return A fully-populated `ParamsType`.
 */
template <typename ParamsType>
ParamsType load_params_for(const Config& cfg);
