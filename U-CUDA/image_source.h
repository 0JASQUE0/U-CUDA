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

// Обратное направление: захватывает прямоугольник текущего default-
// framebuffer'а (физические пиксели, top-left origin — как у
// glfwGetFramebufferSize) и кладёт его в буфер обмена как CF_DIB. Звать
// строго после ImGui_ImplOpenGL3_RenderDrawData и до glfwSwapBuffers, пока
// ещё привязан default FBO (id 0). Только Windows; возвращает false в
// остальных случаях (в т.ч. если прямоугольник вырожден/вне framebuffer'а).
bool copy_framebuffer_rect_to_clipboard(int x0, int y0_top, int x1, int y1_top);
