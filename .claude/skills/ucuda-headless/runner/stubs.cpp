// Заглушки символов, которые объектники U-CUDA ждут от GUI-части приложения.
#include <string>
#include <windows.h>
const char* basin_feature_name(int) { return "f"; }   // тянется из data_export
std::string exe_dir() {                               // ждёт krs_cpu (у движка своя копия)
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n);
    size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? std::string(".") : p.substr(0, k);
}
