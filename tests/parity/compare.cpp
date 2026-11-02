#include "compare.h"
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 7) throw std::runtime_error("Usage: ucuda_parity_compare reference actual report.json [--tolerances absolute relative]");
        double absolute = 0, relative = 0;
        if (argc == 7) {
            if (std::string(argv[4]) != "--tolerances") throw std::runtime_error("Unknown option");
            size_t absolute_end = 0, relative_end = 0;
            absolute = std::stod(argv[5], &absolute_end); relative = std::stod(argv[6], &relative_end);
            if (absolute_end != std::string(argv[5]).size() || relative_end != std::string(argv[6]).size())
                throw std::runtime_error("Invalid tolerance");
            if (!std::isfinite(absolute) || !std::isfinite(relative) || absolute < 0 || relative < 0) throw std::runtime_error("Invalid tolerance");
        }
        const auto report = parity::compare(std::filesystem::u8path(argv[1]), std::filesystem::u8path(argv[2]), argc == 4, absolute, relative);
        const auto output = std::filesystem::u8path(argv[3]);
        if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
        std::ofstream out(output, std::ios::binary);
        if (!out || !(out << report.dump(2) << '\n')) throw std::runtime_error("Cannot write comparison report");
        std::cout << report.at("status") << ": " << report.at("files") << " files, " << report.at("numbers")
            << " numbers, " << report.at("bit_differences") << " bit differences, " << report.at("different_files") << " different files\n";
        return report.at("different_files") == 0 ? 0 : 1;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
