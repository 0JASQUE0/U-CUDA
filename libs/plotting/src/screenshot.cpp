#include "ucuda/plotting/screenshot.h"
#include "ucuda/platform/image_source.h"
#include <glad/glad.h>
#include <algorithm>
bool copy_framebuffer_rect_to_clipboard(int x0, int y0, int x1, int y1) {
    GLint vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
    x0 = std::clamp(x0, 0, vp[2]); x1 = std::clamp(x1, 0, vp[2]);
    y0 = std::clamp(y0, 0, vp[3]); y1 = std::clamp(y1, 0, vp[3]);
    const int w = x1 - x0, h = y1 - y0;
    if (w <= 0 || h <= 0) return false;
    std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 4);
    GLint alignment; glGetIntegerv(GL_PACK_ALIGNMENT, &alignment);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(x0, vp[3] - y1, w, h, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    glPixelStorei(GL_PACK_ALIGNMENT, alignment);
    return copy_rgba_to_clipboard(w, h, pixels);
}
