#include "platform_posix.h"
#include "platform_paths.h"
#include "krs_cpu.h"
#include "ocr_client_win.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

static std::string resource_dir;
std::string exe_dir() { return resource_dir; }
static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--fake-ocr") {
        std::cout << "READY\n" << std::flush;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            std::cout << (line == "PING" ? "OK pong\n" : "OK x^2\n") << std::flush;
        }
        return 0;
    }
    try {
        std::string pattern = (std::filesystem::temp_directory_path() / "ucuda-test-XXXXXX").string();
        require(mkdtemp(pattern.data()) != nullptr, "temporary directory");
        resource_dir = pattern;
        std::filesystem::create_directories(pattern + "/kernels");
        std::filesystem::copy_file(std::string(U_CUDA_TEST_SOURCE) + "/configCUDA.h", pattern + "/kernels/configCUDA.h");
        std::string output;
        require(platform::run({"printf", "%s", "space ' quote ; $(echo unsafe)"}, output) == 0,
                "process exit status");
        require(output == "space ' quote ; $(echo unsafe)", "arguments must not be shell-expanded");
        require(platform::run({"cat"}, output, "hello\n") == 0 && output == "hello\n", "process stdin");
        require(platform::run({"false"}, output) != 0, "failure exit status");
        bool missing = false;
        try { platform::Process child({"/no/such/ucuda-program"}); }
        catch (const std::exception&) { missing = true; }
        require(missing, "missing executable error");
        OcrClient client(std::filesystem::canonical("/proc/self/exe").string(), "--fake-ocr");
        require(client.ping(), "OCR handshake/ping");
        require(client.recognize_base64("eA==") == "x^2", "OCR response");
        require(b64encode({'f', 'o', 'o'}) == "Zm9v", "base64 protocol");
        KrsCpuStep step;
        std::vector<KrsCpuDiag> diags;
        require(step.compile("X[0] += h * a[0];", 1, 1, diags), "compile CPU scheme");
        numb x[] = {1}, a[] = {2};
        step.fn()(x, a, 0.25);
        require(std::abs(x[0] - 1.5) < 1e-12, "execute CPU scheme");
        require(step.compile("X[0] += h * a[0];", 1, 1, diags), "cached CPU scheme");
        diags.clear();
        require(!step.compile("X[1] = 0;", 1, 1, diags) && !diags.empty(), "reject out-of-bounds index");
        diags.clear();
        require(!step.compile("X[0] = missing_symbol;", 1, 1, diags), "compiler diagnostic");
        require(!diags.empty() && diags[0].line == 1, "diagnostic line number");
        std::filesystem::remove_all(pattern);
        std::cout << "Platform, OCR protocol and CPU compilation checks passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
