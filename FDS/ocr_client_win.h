#pragma once
#include <string>
#include <vector>

// Windows-версия OcrClient: запускает Python OCR-сервер через CreateProcess,
// общается с ним по анонимным пайпам построчно. Тот же интерфейс, что у
// POSIX-версии, поэтому main.cpp от платформы не зависит.
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
