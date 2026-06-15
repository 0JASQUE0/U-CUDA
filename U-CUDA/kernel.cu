#include "system_library.h"
#include <iostream>

int main() {
    SystemLibrary lib("library");  // папка library рядом с exe

    // создаём систему
    SystemRecord r;
    r.name = "TestLorenz";
    r.note = "test note with \"quotes\" and \\backslash";
    r.latex_text = "\\dot{x} = \\sigma(y-x) \\\\\n \\dot{y} = -y";
    r.alphabet_text = "x,y,sigma";
    r.step_h = "0.01";
    r.init_conditions["x"] = "0.1";
    r.init_conditions["y"] = "";  // пустое

    std::string name = lib.save(r);
    std::cout << "Saved: " << name << "\n";

    // список
    std::cout << "Library contents:\n";
    for (auto& n : lib.list()) std::cout << "  - " << n << "\n";

    // загрузка
    SystemRecord loaded = lib.load("TestLorenz");
    std::cout << "Loaded latex: " << loaded.latex_text << "\n";
    std::cout << "Empty y preserved: [" << loaded.init_conditions["y"] << "]\n";

    // дубликат
    std::string copy = lib.duplicate("TestLorenz");
    std::cout << "Duplicated as: " << copy << "\n";

    return 0;
}