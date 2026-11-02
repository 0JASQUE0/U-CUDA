#include "support/artifacts.h"
#include "support/process.h"
#include "ucuda/cuda/runtime_settings.h"
#include "ucuda/platform/platform_paths.h"
#include <cuda.h>
#include <nvrtc.h>
#include <chrono>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <regex>
#ifdef _MSC_VER
#include <intrin.h>
#endif

using namespace regression;

namespace {
std::string identifier() {
    return std::to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}
Json environment(const fs::path& directory) {
    std::string cpu;
#ifdef _MSC_VER
    int words[4];
    __cpuid(words, (int)0x80000000);
    if ((unsigned)words[0] >= 0x80000004) {
        char brand[49]{};
        for (int i = 0; i < 3; ++i) { __cpuid(words, (int)0x80000002 + i); std::memcpy(brand + 16 * i, words, 16); }
        cpu = brand;
    }
#else
    std::ifstream in("/proc/cpuinfo");
    for (std::string line; std::getline(in, line);)
        if (line.find("model name") == 0) { cpu = line.substr(line.find(':') + 1); break; }
#endif
    int driver = 0, major = 0, minor = 0;
    if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS) throw std::runtime_error("nvrtcVersion failed");
    Json result = {{"cpu", cpu}, {"compiler", UCUDA_TEST_COMPILER}, {"configuration", UCUDA_TEST_CONFIGURATION},
        {"nvcc", UCUDA_TEST_CUDA_COMPILER},
        {"system", UCUDA_TEST_SYSTEM}, {"cpu_threads", std::thread::hardware_concurrency()},
        {"build_flags", UCUDA_TEST_BUILD_FLAGS}, {"numb_bytes", sizeof(numb)},
        {"fma", get_nvrtc_fmad()}, {"rdc", get_nvrtc_rdc()}, {"nvrtc", {major, minor}}, {"gpu", nullptr}};
    result["gpu_block_size"] = get_gpu_block_size();
    result["runtime_environment"] = Json::object();
    for (const char* key : {"CUDA_CACHE_DISABLE", "CUDA_MODULE_LOADING", "CUDA_VISIBLE_DEVICES",
                            "CUDA_LAUNCH_BLOCKING", "UCUDA_NVRTC_PRUNE"}) {
        const auto value = std::getenv(key);
        result["runtime_environment"][key] = value ? value : "";
    }
#ifdef _WIN32
    const auto programFiles = std::getenv("ProgramFiles(x86)");
    if (programFiles) {
        const auto vswhere = fs::path(programFiles) / "Microsoft Visual Studio/Installer/vswhere.exe";
        if (fs::exists(vswhere) && process({vswhere.string(), "-latest", "-products", "*", "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-format", "json", "-utf8"}, directory / "native-toolchain.json") == 0) {
            const auto installations = read_json(directory / "native-toolchain.json");
            if (!installations.empty()) result["native_toolchain"] = installations.at(0).at("installationVersion");
        }
    }
#endif
    if (cuInit(0) == CUDA_SUCCESS) {
        CUdevice device;
        if (cuDeviceGet(&device, 0) == CUDA_SUCCESS) {
            char name[256]{}; CUuuid uuid{};
            if (cuDeviceGetName(name, sizeof(name), device) != CUDA_SUCCESS ||
                cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS || cuDriverGetVersion(&driver) != CUDA_SUCCESS)
                throw std::runtime_error("Cannot identify CUDA device");
            std::ostringstream id;
            for (unsigned char c : uuid.bytes) id << std::hex << std::setw(2) << std::setfill('0') << (int)c;
            result["gpu"] = {{"name", name}, {"uuid", id.str()}, {"driver", driver}};
            result["gpu"]["driver_release"] = "unavailable";
            if (*UCUDA_TEST_NVIDIA_SMI) {
                const auto hex = id.str();
                const auto uuidText = "GPU-" + hex.substr(0, 8) + "-" + hex.substr(8, 4) + "-" + hex.substr(12, 4) + "-" + hex.substr(16, 4) + "-" + hex.substr(20);
                if (process({UCUDA_TEST_NVIDIA_SMI, "--id=" + uuidText, "--query-gpu=driver_version", "--format=csv,noheader"}, directory / "driver.txt") == 0)
                    result["gpu"]["driver_release"] = text_file(directory / "driver.txt");
            }
        }
    }
    return result;
}
std::map<std::string, std::string> options(int argc, char** argv, int from) {
    std::map<std::string, std::string> out;
    for (int i = from; i < argc; ++i) {
        const std::string key = argv[i];
        static const std::set<std::string> known = {"--record", "--config", "--baseline", "--run", "--reason",
            "--mode", "--backend", "--output", "--cases", "--cases-file", "--benchmarks"};
        if (!known.count(key)) throw std::runtime_error("Unknown option " + key);
        if (key == "--record") { out[key] = "true"; continue; }
        if (key.rfind("--", 0) != 0 || i + 1 == argc) throw std::runtime_error("Expected --option value");
        if (!out.emplace(key, argv[++i]).second) throw std::runtime_error("Duplicate option " + key);
    }
    return out;
}
std::set<std::string> selection(const std::string& value) {
    std::set<std::string> out;
    std::istringstream in(value);
    for (std::string s; std::getline(in, s, ',');) if (!s.empty()) out.insert(s);
    return out;
}
fs::path run_directory(const std::string& value, const std::map<std::string, std::string>& opt) {
    fs::path path = fs::u8path(value);
    if (fs::is_regular_file(path / "run.json")) return fs::absolute(path);
    if (!path.has_parent_path()) {
        const auto root = opt.count("--output") ? fs::path(opt.at("--output")) : fs::path(UCUDA_TEST_BUILD) / "test-results";
        path = root / path;
        if (fs::is_regular_file(path / "run.json")) return fs::absolute(path);
    }
    throw std::runtime_error("Unknown run: " + value);
}
Json compare(const Json& run, const Json& base, const Json& config) {
    if (run.at("schema") != 1 || base.at("schema") != 1) throw std::runtime_error("Unsupported artifact schema");
    Json comparisons = Json::array();
    if (!run.value("checks_passed", false))
        comparisons.push_back({{"id", "independent_checks"}, {"status", "failed"}});
    if (run.contains("profile")) {
        const auto profile = run.at("profile").get<std::string>();
        Json expected;
        if (base.contains("profiles") && base.at("profiles").contains(profile)) expected = base.at("profiles").at(profile);
        else if (run.contains("selection") && base.contains("profiles")) {
            const auto parent = run.at("mode").get<std::string>() + "|" + run.at("backend").get<std::string>() + "||";
            if (base.at("profiles").contains(parent)) {
                const auto& full = base.at("profiles").at(parent);
                expected = {{"tests", run.at("tests")}, {"cases", Json::array()}};
                for (const auto& test : run.at("tests"))
                    if (std::find(full.at("tests").begin(), full.at("tests").end(), test) == full.at("tests").end())
                        comparisons.push_back({{"id", test}, {"status", "new"}});
                const auto pattern = run.at("selection").value("benchmarks", "");
                const std::regex filter(pattern.empty() ? ".*" : pattern);
                for (const auto& idValue : full.at("cases")) {
                    const auto id = idValue.get<std::string>();
                    const auto& c = base.at("cases").at(id);
                    if (id.rfind("perf.", 0) == 0) {
                        if (std::regex_search(id.substr(5), filter)) expected["cases"].push_back(id);
                    } else for (const auto& test : c.value("tests", Json::array()))
                        if (std::find(run.at("tests").begin(), run.at("tests").end(), test) != run.at("tests").end()) {
                            expected["cases"].push_back(id); break;
                        }
                }
            }
        }
        if (expected.is_null()) comparisons.push_back({{"id", "profile"}, {"status", "new"}, {"reason", "No accepted coverage for this selection"}});
        else {
            for (const auto& id : expected.at("cases"))
                if (!run.at("cases").contains(id.get<std::string>()))
                    comparisons.push_back({{"id", id}, {"status", "missing"}});
            for (const auto& test : expected.at("tests"))
                if (std::find(run.at("tests").begin(), run.at("tests").end(), test) == run.at("tests").end())
                    comparisons.push_back({{"id", test}, {"status", "missing"}});
            comparisons.push_back({{"id", "coverage"}, {"status", "pass"}});
        }
    }
    for (auto it = run.at("cases").begin(); it != run.at("cases").end(); ++it) {
        if (!base.at("cases").contains(it.key())) comparisons.push_back({{"id", it.key()}, {"status", "new"}});
        else comparisons.push_back(compare_case(it.value(), base.at("cases").at(it.key()), config,
                                               run.at("environment") == base.at("environment")));
    }
    return comparisons;
}
bool passed(const Json& results) {
    return !results.empty() && std::all_of(results.begin(), results.end(), [](const Json& r) { return r.at("status") == "pass"; });
}
void provenance(const fs::path& dir, Json& run) {
    auto git = [&](std::vector<std::string> args, const char* file) {
        args.insert(args.begin(), {UCUDA_TEST_GIT, "-C", UCUDA_TEST_SOURCE});
        if (process(args, dir / file) != 0) throw std::runtime_error("Cannot capture Git provenance");
    };
    git({"rev-parse", "HEAD"}, "revision.txt");
    git({"status", "--porcelain=v1"}, "working-tree.txt");
    git({"diff", "HEAD", "--binary", "--no-ext-diff"}, "working-tree.patch");
    run["revision"] = text_file(dir / "revision.txt");
    run["working_tree"] = text_file(dir / "working-tree.txt");
    Json hashes = Json::object();
    // Record runtime source identity for provenance, not as a compatibility veto.
    const fs::path kernels = fs::path(platform::executable_dir()) / "kernels";
    for (const auto& entry : fs::directory_iterator(kernels)) {
        if (!entry.is_regular_file()) continue;
        uint64_t h = 14695981039346656037ull;
        for (unsigned char b : text_file(entry.path())) { h ^= b; h *= 1099511628211ull; }
        hashes[entry.path().filename().string()] = std::to_string(h);
    }
    run["runtime_fingerprints_fnv1a64"] = hashes;
}
}

int main(int argc, char** argv) {
    fs::path activeRun;
    try {
        if (argc < 2) throw std::runtime_error("Commands: run, compare, baseline accept");
        const std::string command = argv[1];
        const bool accept = command == "baseline" && argc > 2 && std::string(argv[2]) == "accept";
        auto opt = options(argc, argv, accept ? 3 : 2);
        const fs::path configPath = opt.count("--config") ? opt.at("--config") : UCUDA_TEST_SOURCE "/tests/test_config.json";
        const Json config = read_json(configPath); validate_config(config);
        const fs::path baselineRoot = fs::path(UCUDA_TEST_SOURCE) / "tests" / "baselines";
        const fs::path basePath = opt.count("--baseline") ? fs::path(opt.at("--baseline")) : baselineRoot / "active.json";
        if (command == "compare") {
            const auto runPath = run_directory(opt.at("--run"), opt);
            const Json result = compare(read_json(runPath / "run.json"), read_baseline(basePath), config);
            std::cout << std::setw(2) << result << '\n';
            return passed(result) ? 0 : 1;
        }
        if (accept) {
            const auto reason = opt.at("--reason");
            if (basePath.parent_path().filename().string().find("-history") != std::string::npos)
                throw std::runtime_error("Historical baseline files are immutable; select an active baseline file");
            if (reason.find_first_not_of(" \t\r\n") == std::string::npos) throw std::runtime_error("A reason is required");
            const auto source = run_directory(opt.at("--run"), opt);
            const Json candidate = read_json(source / "run.json");
            if (candidate.at("schema") != 1) throw std::runtime_error("Unsupported artifact schema");
            if (!candidate.at("checks_passed").get<bool>()) throw std::runtime_error("Independent checks did not pass");
            if (candidate.at("cases").empty()) throw std::runtime_error("No results to accept");
            Json base = fs::exists(basePath) ? read_baseline(basePath) : Json{{"schema", 1}, {"cases", Json::object()}};
            if (opt.count("--cases") && opt.count("--cases-file")) throw std::runtime_error("Choose --cases or --cases-file");
            const auto selected = opt.count("--cases-file") ? read_json(fs::u8path(opt.at("--cases-file"))).get<std::set<std::string>>()
                : selection(opt.count("--cases") ? opt.at("--cases") : "");
            if (opt.count("--cases-file") && selected.empty()) throw std::runtime_error("The selection file is empty");
            for (const auto& id : selected) if (!candidate.at("cases").contains(id)) throw std::runtime_error("Unknown case " + id);
            if (!base.at("cases").empty() && base.at("environment") != candidate.at("environment"))
                throw std::runtime_error("Use a separate --baseline file for a different environment");
            std::cout << std::setw(2) << compare(candidate, base, config) << '\n';
            for (auto it = candidate.at("cases").begin(); it != candidate.at("cases").end(); ++it) {
                if (!selected.empty() && !selected.count(it.key())) continue;
                if (!it.value().value("passed", false)) throw std::runtime_error("Case did not pass its independent checks");
                if (!it.value().at("version").is_number_integer() || it.value().at("version").get<int>() < 1)
                    throw std::runtime_error("Scenario version must be a positive integer");
                if (base.at("cases").contains(it.key())) {
                    const auto& previous = base.at("cases").at(it.key());
                    if (it.value().at("version") < previous.at("version") ||
                        (it.value().at("inputs") != previous.at("inputs") && it.value().at("version") == previous.at("version")))
                        throw std::runtime_error("Scenario inputs changed; increment the scenario version before accepting");
                }
                const auto values = it.value().at("values").get<std::vector<double>>();
                if (values.empty() && !it.value().contains("series")) throw std::runtime_error("Missing numerical result");
                for (double value : values) if (!std::isfinite(value)) throw std::runtime_error("Non-finite numerical result");
                if (it.value().contains("series")) {
                    const auto& series = it.value().at("series");
                    const auto& p = candidate.at("configuration").at("performance");
                    if ((int)series.size() != p.at("series").get<int>()) throw std::runtime_error("Incomplete timing series");
                    for (const auto& samples : series) {
                        if ((int)samples.size() != p.at("repetitions").get<int>()) throw std::runtime_error("Incomplete timing repetitions");
                        median(samples.get<std::vector<double>>());
                    }
                    if (compare_case(it.value(), it.value(), config, true).at("status") != "pass")
                        throw std::runtime_error("Unstable timing cannot be accepted; rerun the selected case");
                }
                base["cases"][it.key()] = it.value();
                base["cases"][it.key()]["source"] = {{"run", source.string()}, {"revision", candidate.at("revision")}};
            }
            auto covered = [&](const Json& ids) {
                for (const auto& id : ids) {
                    if (!base.at("cases").contains(id.get<std::string>())) return false;
                    const auto& accepted = base.at("cases").at(id.get<std::string>());
                    const auto& sourceCase = candidate.at("cases").at(id.get<std::string>());
                    // Explicitly accepted newer scenario versions retain their coverage.
                    if (accepted.at("version") < sourceCase.at("version") ||
                        (accepted.at("version") == sourceCase.at("version") && accepted.at("inputs") != sourceCase.at("inputs"))) return false;
                }
                return !ids.empty();
            };
            Json ids = Json::array();
            for (auto it = candidate.at("cases").begin(); it != candidate.at("cases").end(); ++it) ids.push_back(it.key());
            if (covered(ids))
                base["profiles"][candidate.at("profile").get<std::string>()] = {{"tests", candidate.at("tests")}, {"cases", ids}};
            // Each sub-mode becomes available only when all of its cases are accepted.
            if (candidate.contains("test_labels") && candidate.at("profile").get<std::string>() ==
                candidate.at("mode").get<std::string>() + "|" + candidate.at("backend").get<std::string>() + "||") {
                const auto sourceMode = candidate.at("mode").get<std::string>();
                const auto sourceBackend = candidate.at("backend").get<std::string>();
                for (const std::string mode : {"smoke", "regression", "performance", "all"}) {
                    if (sourceMode != "all" && mode != sourceMode && !(mode == "smoke" && sourceMode != "smoke")) continue;
                    for (const std::string backend : {"all", "cpu", "gpu"}) {
                        if (sourceBackend != "all" && backend != sourceBackend) continue;
                        Json tests = Json::array(), cases = Json::array();
                        const auto wanted = mode == "smoke" || mode == "performance" ? "smoke" : "regression";
                        for (auto test = candidate.at("test_labels").begin(); test != candidate.at("test_labels").end(); ++test) {
                            const auto& labels = test.value();
                            if (std::find(labels.begin(), labels.end(), wanted) == labels.end()) continue;
                            const auto exclude = backend == "cpu" ? "gpu" : backend == "gpu" ? "cpu" : "";
                            if (*exclude && std::find(labels.begin(), labels.end(), exclude) != labels.end()) continue;
                            tests.push_back(test.key());
                        }
                        for (auto c = candidate.at("cases").begin(); c != candidate.at("cases").end(); ++c) {
                            if (c.key().rfind("perf.", 0) == 0) {
                                if (mode != "all" && mode != "performance") continue;
                                if (backend != "all" && c.key().rfind("perf." + backend + "_", 0) != 0) continue;
                                cases.push_back(c.key()); continue;
                            }
                            for (const auto& test : c.value().value("tests", Json::array()))
                                if (std::find(tests.begin(), tests.end(), test) != tests.end()) { cases.push_back(c.key()); break; }
                        }
                        if (!tests.empty() && covered(cases)) base["profiles"][mode + "|" + backend + "||"] = {{"tests", tests}, {"cases", cases}};
                    }
                }
            }
            const std::string id = identifier();
            base["previous"] = base.value("id", ""); base["id"] = id;
            base["reason"] = reason; base["environment"] = candidate.at("environment");
            base["configuration"] = candidate.value("configuration", config);
            base["source_run"] = source.string();
            base["revision"] = candidate.at("revision");
            const auto history = basePath.parent_path() / (basePath.stem().string() + "-history");
            fs::create_directories(history);
            if (fs::exists(history / (id + ".json"))) throw std::runtime_error("Baseline version already exists");
            if (fs::exists(basePath) && !fs::exists(history / (base.value("previous", "initial") + ".json")))
                fs::copy_file(basePath, history / (base.value("previous", "initial") + ".json"));
            const auto manifest = baseline_manifest(base, basePath.parent_path() / (basePath.stem().string() + "-data"));
            write_json(history / (id + ".json"), manifest);
            const auto temp = fs::path(basePath.string() + ".tmp"); write_json(temp, manifest);
#ifdef _WIN32
            if (!MoveFileExW(temp.c_str(), basePath.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                throw std::runtime_error("Cannot replace baseline atomically");
#else
            fs::rename(temp, basePath);
#endif
            std::cout << "Accepted baseline " << id << " (" << (selected.empty() ? candidate.at("cases").size() : selected.size())
                << " cases) with reason: " << reason << '\n'; return 0;
        }
        if (command != "run") throw std::runtime_error("Unknown command");
        const auto mode = opt.count("--mode") ? opt.at("--mode") : "smoke";
        const auto backend = opt.count("--backend") ? opt.at("--backend") : "all";
        if (mode != "smoke" && mode != "regression" && mode != "performance" && mode != "all") throw std::runtime_error("Unknown mode");
        if (backend != "cpu" && backend != "gpu" && backend != "all") throw std::runtime_error("Unknown backend");
        const fs::path root = opt.count("--output") ? fs::path(opt.at("--output")) : fs::path(UCUDA_TEST_BUILD) / "test-results";
        fs::create_directories(root);
        const auto dir = fs::absolute(root / identifier());
        if (!fs::create_directory(dir)) throw std::runtime_error("Run history directory already exists");
        activeRun = dir;
        fs::create_directory(dir / "numeric");
        std::cout << "Results: " << dir << '\n';
        Json run = {{"schema", 1}, {"mode", mode}, {"backend", backend}, {"environment", environment(dir)},
            {"cases", Json::object()}, {"checks_passed", false}, {"configuration", config}, {"status", "incomplete"}};
        const auto selectedCases = opt.count("--cases") ? opt.at("--cases") : "";
        const auto selectedBenchmarks = opt.count("--benchmarks") ? opt.at("--benchmarks") : "";
        run["selection"] = {{"cases", selectedCases}, {"benchmarks", selectedBenchmarks}};
        run["profile"] = mode + "|" + backend + "|" + (selectedCases.empty() && selectedBenchmarks.empty() ? "|"
            : Json::array({selectedCases, selectedBenchmarks}).dump());
        provenance(dir, run); write_json(dir / "run.json", run);
        if (backend != "cpu" && run.at("environment").at("gpu").is_null()) throw std::runtime_error("Requested GPU is unavailable");
        const auto numeric = (dir / "numeric").string();
#ifdef _WIN32
        _putenv_s("UCUDA_TEST_RESULT_DIR", numeric.c_str());
#else
        setenv("UCUDA_TEST_RESULT_DIR", numeric.c_str(), 1);
#endif
        const std::string label = mode == "smoke" || mode == "performance" ? "smoke" : "regression";
        std::vector<std::string> args = {UCUDA_TEST_CTEST, "--test-dir", UCUDA_TEST_BUILD, "-C", UCUDA_TEST_CONFIGURATION,
            "-L", label, "--output-on-failure", "--no-tests=error", "--output-junit", (dir / "ctest.xml").string()};
        if (backend != "all") { args.push_back("-LE"); args.push_back(backend == "cpu" ? "gpu" : "cpu"); }
        if (opt.count("--cases")) { args.push_back("-R"); args.push_back(opt.at("--cases")); }
        auto listing = args; listing.push_back("--show-only=json-v1");
        if (process(listing, dir / "test-plan.json") != 0) throw std::runtime_error("Cannot enumerate selected tests");
        run["tests"] = Json::array();
        run["test_labels"] = Json::object();
        const auto plan = read_json(dir / "test-plan.json");
        for (const auto& test : plan.at("tests")) {
            const auto name = test.at("name").get<std::string>(); run["tests"].push_back(name);
            run["test_labels"][name] = Json::array();
            for (const auto& property : test.at("properties"))
                if (property.at("name") == "LABELS") run["test_labels"][name] = property.at("value");
        }
        if (run.at("tests").empty()) throw std::runtime_error("No selected tests");
        const int check = process(args, dir / "ctest.log");
        bool incomplete = !fs::exists(dir / "ctest.xml");
        run["optional_skipped"] = Json::array();
        if (!incomplete) {
            auto xml = text_file(dir / "ctest.xml");
            const auto name = xml.find("name=\"upstream_kicad\"");
            if (name != std::string::npos) {
                const auto begin = xml.rfind("<testcase", name), end = xml.find("</testcase>", name);
                if (begin != std::string::npos && end != std::string::npos && xml.substr(begin, end - begin).find("<skipped") != std::string::npos) {
                    run["optional_skipped"].push_back("upstream_kicad"); xml.erase(begin, end + 11 - begin);
                }
            }
            incomplete = xml.find("<skipped") != std::string::npos;
        }
        run["checks_passed"] = check == 0 && !incomplete;
        for (const auto& entry : fs::directory_iterator(dir / "numeric")) {
            if (entry.path().extension() != ".json") continue;
            auto c = read_json(entry.path());
            if (!run["cases"].emplace(c.at("id").get<std::string>(), c).second) throw std::runtime_error("Duplicate snapshot ID");
        }
        bool perfOk = true, unstableMeasurement = false;
        if ((mode == "performance" || mode == "all") && run.at("checks_passed").get<bool>()) {
            if (std::string(UCUDA_TEST_CONFIGURATION) != "Release") throw std::runtime_error("Performance requires Release");
            if (backend != "cpu" && run.at("environment").at("gpu").at("driver_release") == "unavailable")
                throw std::runtime_error("GPU timing requires driver identity from nvidia-smi");
#ifdef _WIN32
            _putenv_s("UCUDA_BENCHMARK_BACKEND", backend.c_str());
#else
            setenv("UCUDA_BENCHMARK_BACKEND", backend.c_str(), 1);
#endif
            const auto& p = config.at("performance");
            for (int i = 0; i < p.at("series").get<int>(); ++i) {
                const auto out = dir / ("benchmark-" + std::to_string(i) + ".json");
                std::vector<std::string> bargs = {(fs::path(platform::executable_dir()) / UCUDA_BENCHMARK_NAME).string(),
                    "--benchmark_out=" + out.string(), "--benchmark_out_format=json",
                    "--benchmark_min_warmup_time=" + std::to_string(p.at("warmup_seconds").get<double>()),
                    "--benchmark_min_time=" + std::to_string(p.at("minimum_seconds").get<double>()) + "s",
                    "--benchmark_repetitions=" + std::to_string(p.at("repetitions").get<int>())};
                if (backend != "all") bargs.push_back("--benchmark_filter=" + backend + "_");
                if (opt.count("--benchmarks") && !opt.at("--benchmarks").empty()) {
                    if (backend != "all") bargs.pop_back();
                    bargs.push_back("--benchmark_filter=" + opt.at("--benchmarks"));
                }
                if (process(bargs, dir / ("benchmark-" + std::to_string(i) + ".log")) != 0) { perfOk = false; break; }
                int count = 0;
                const auto report = read_json(out);
                for (const auto& b : report.at("benchmarks")) {
                    if (b.value("error_occurred", false)) { perfOk = false; continue; }
                    if (b.value("run_type", "") != "iteration") continue;
                    const auto id = "perf." + b.at("run_name").get<std::string>();
                    Json inputs = {{"benchmark", b.at("run_name")}, {"unit", b.at("time_unit")}};
                    for (auto field = b.begin(); field != b.end(); ++field)
                        if (field.key().rfind("input_", 0) == 0) inputs[field.key()] = field.value();
                    if (!run["cases"].contains(id)) run["cases"][id] = {{"id", id}, {"version", b.value("input_workload_version", 1)},
                        {"inputs", inputs},
                        {"values", Json::array()}, {"passed", true}, {"series", Json::array()}};
                    auto& series = run["cases"][id]["series"];
                    while ((int)series.size() <= i) series.push_back(Json::array());
                    series[i].push_back(b.at("real_time")); ++count;
                }
                if (!count) perfOk = false;
            }
            for (auto& c : run.at("cases")) {
                if (!c.contains("series")) continue;
                if ((int)c.at("series").size() != p.at("series").get<int>()) perfOk = false;
                std::vector<double> medians;
                for (const auto& samples : c.at("series")) {
                    if ((int)samples.size() != p.at("repetitions").get<int>()) perfOk = false;
                    medians.push_back(median(samples.get<std::vector<double>>()));
                }
                const double center = median(medians);
                const auto bounds = std::minmax_element(medians.begin(), medians.end());
                const double spread = (*bounds.second - *bounds.first) / center;
                c["statistics"] = {{"series_medians", medians}, {"median", center}, {"relative_spread", spread}};
                const bool noisy = spread > p.at("max_relative_spread").get<double>();
                c["measurement_status"] = noisy ? "inconclusive" : "stable";
                unstableMeasurement |= noisy;
            }
        }
        run["checks_passed"] = run.at("checks_passed").get<bool>() && perfOk;
        run["status"] = run.at("checks_passed").get<bool>() ? unstableMeasurement ? "inconclusive" : "recorded" : check || !perfOk ? "failed" : "incomplete";
        if (fs::exists(basePath) && run.at("checks_passed").get<bool>()) {
            run["comparison"] = compare(run, read_baseline(basePath), config);
            run["status"] = passed(run["comparison"]) ? "pass" : "regression_or_incomplete";
        } else if (!opt.count("--record") && run.at("checks_passed").get<bool>()) run["status"] = "baseline_missing";
        write_json(dir / "run.json", run);
        std::cout << "Status: " << run.at("status") << "; report: " << dir / "run.json" << '\n';
        if (!run.at("optional_skipped").empty()) std::cout << "Optional checks skipped: " << run.at("optional_skipped") << '\n';
        if (check) std::cout << text_file(dir / "ctest.log");
        return run.at("status") == "pass" || (opt.count("--record") && run.at("status") == "recorded") ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        if (!activeRun.empty()) try {
            const auto path = activeRun / "run.json";
            Json failed = fs::exists(path) ? read_json(path) : Json::object();
            failed["status"] = "incomplete"; failed["checks_passed"] = false; failed["error"] = e.what();
            write_json(path, failed);
        } catch (const std::exception& logError) { std::cerr << "Cannot save failure: " << logError.what() << '\n'; }
        return 1;
    }
}
