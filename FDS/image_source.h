#pragma once
#include <string>
#include <vector>

// Абстракция источника изображения: возвращает PNG-байты.
// Реализации: файл (сейчас), буфер обмена (позже, Ctrl+V).
// OCR-путь не зависит от того, откуда взялась картинка.
struct ImageSource {
    virtual ~ImageSource() = default;
    // Возвращает PNG-байты или бросает std::runtime_error.
    virtual std::vector<unsigned char> get_png() = 0;
    virtual const char* name() const = 0;
};

// Источник из файла на диске (PNG/JPG — что прочитает OCR).
struct FileImageSource : ImageSource {
    std::string path;
    explicit FileImageSource(std::string p) : path(std::move(p)) {}
    std::vector<unsigned char> get_png() override;
    const char* name() const override { return "file"; }
};

// Заглушка для будущего Ctrl+V: захват из буфера обмена (платформенный код).
// Реализуется позже через Win32 OpenClipboard + конвертацию DIB->PNG.
struct ClipboardImageSource : ImageSource {
    std::vector<unsigned char> get_png() override; // пока бросает not-implemented
    const char* name() const override { return "clipboard"; }
};
