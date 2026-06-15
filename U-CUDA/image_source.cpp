#include "image_source.h"
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>

// ---- Файловый источник (без изменений) ----
std::vector<unsigned char> FileImageSource::get_png() {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open image: " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),
        std::istreambuf_iterator<char>());
}

// ---- Источник из буфера обмена (Windows) ----
#ifdef _WIN32
#include <windows.h>

// stb_image_write для кодирования в PNG. STB_IMAGE_WRITE_IMPLEMENTATION
// должна быть определена РОВНО в одной .cpp — определяем здесь.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// колбэк stb: накапливает PNG-байты в вектор
static void stb_write_cb(void* ctx, void* data, int size) {
    auto* out = static_cast<std::vector<unsigned char>*>(ctx);
    auto* p = static_cast<unsigned char*>(data);
    out->insert(out->end(), p, p + size);
}

std::vector<unsigned char> ClipboardImageSource::get_png() {
    if (!IsClipboardFormatAvailable(CF_DIB))
        throw std::runtime_error("clipboard has no image (CF_DIB)");
    if (!OpenClipboard(nullptr))
        throw std::runtime_error("cannot open clipboard");

    std::vector<unsigned char> png;
    try {
        HANDLE h = GetClipboardData(CF_DIB);
        if (!h) throw std::runtime_error("GetClipboardData(CF_DIB) failed");

        void* dib = GlobalLock(h);
        if (!dib) throw std::runtime_error("GlobalLock failed");

        BITMAPINFOHEADER* bih = reinterpret_cast<BITMAPINFOHEADER*>(dib);
        int width = bih->biWidth;
        int height = bih->biHeight; // может быть отрицательной (top-down)
        int bpp = bih->biBitCount;

        if (bih->biSize != sizeof(BITMAPINFOHEADER)) {
            // встречаются BITMAPV4/V5HEADER — размер больше; учитываем это при поиске пикселей
        }
        if (bpp != 24 && bpp != 32) {
            GlobalUnlock(h);
            throw std::runtime_error("unsupported DIB bit depth (need 24/32): "
                + std::to_string(bpp));
        }

        bool top_down = height < 0;
        int abs_h = top_down ? -height : height;

        // указатель на пиксельные данные: после заголовка + палитры.
        // для 24/32 bpp без сжатия палитры нет (biClrUsed обычно 0).
        size_t header_size = bih->biSize;
        size_t palette_size = (size_t)bih->biClrUsed * sizeof(RGBQUAD);
        // для BI_BITFIELDS (часто у 32bpp) идут 3 маски по 4 байта
        if (bih->biCompression == BI_BITFIELDS) palette_size += 3 * sizeof(DWORD);

        unsigned char* bits = reinterpret_cast<unsigned char*>(dib) + header_size + palette_size;

        int channels = bpp / 8;                       // 3 или 4
        int src_stride = ((width * bpp + 31) / 32) * 4; // строки выровнены на 4 байта

        // конвертируем в RGBA (4 канала) с правильным порядком и переворотом строк
        std::vector<unsigned char> rgba((size_t)width * abs_h * 4);
        for (int y = 0; y < abs_h; ++y) {
            // в bottom-up DIB строки идут снизу вверх
            int src_y = top_down ? y : (abs_h - 1 - y);
            unsigned char* src_row = bits + (size_t)src_y * src_stride;
            unsigned char* dst_row = rgba.data() + (size_t)y * width * 4;
            for (int x = 0; x < width; ++x) {
                unsigned char* s = src_row + (size_t)x * channels;
                unsigned char* d = dst_row + (size_t)x * 4;
                // DIB хранит BGR(A); переставляем в RGBA
                d[0] = s[2]; // R
                d[1] = s[1]; // G
                d[2] = s[0]; // B
                d[3] = (channels == 4) ? s[3] : 255; // A
            }
        }

        GlobalUnlock(h);

        // кодируем RGBA -> PNG
        int rc = stbi_write_png_to_func(stb_write_cb, &png, width, abs_h, 4,
            rgba.data(), width * 4);
        if (rc == 0 || png.empty())
            throw std::runtime_error("PNG encode failed");
    }
    catch (...) {
        CloseClipboard();
        throw;
    }
    CloseClipboard();
    return png;
}

#else // не Windows — заглушка
std::vector<unsigned char> ClipboardImageSource::get_png() {
    throw std::runtime_error("clipboard image source: Windows only");
}
#endif