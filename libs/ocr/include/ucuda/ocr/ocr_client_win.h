#pragma once
#include <string>
#include <vector>

// Запускает Python OCR-сервер через CreateProcess (Windows) или posix_spawn
// (Linux), общается с ним построчно. Имя файла сохранено для MSBuild-проекта.
class OcrClient {
public:
    // python_exe: "python" (как в вашем cmd). script: путь к ocr_server.py.
    OcrClient(const std::string& python_exe, const std::string& script);
    ~OcrClient();
    OcrClient(const OcrClient&) = delete;
    OcrClient& operator=(const OcrClient&) = delete;

    // base64 PNG -> LaTeX. Бросает std::runtime_error при ошибке OCR.
    std::string recognize_base64(const std::string& png_base64);
    bool ping();

private:
    void* hChildStdinWr_  = nullptr;  // HANDLE, пишем сюда (-> stdin питона)
    void* hChildStdoutRd_ = nullptr;  // HANDLE, читаем отсюда (<- stdout питона)
    void* hProcess_       = nullptr;  // HANDLE процесса
    std::string rbuf_;

    void write_line(const std::string& s);
    std::string read_line();
};

// base64-кодирование PNG-байтов (из буфера обмена)
std::string b64encode(const std::vector<unsigned char>& data);
