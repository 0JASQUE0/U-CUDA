// Compiled UI tests of the desktop app. Run: U-CUDA.exe --ui-test all (or a name filter).
// Add a test: IM_REGISTER_TEST(engine, "<category>", "<name>")->TestFunc = [](ImGuiTestContext* ctx) { ... };
// Item paths start at a window: "//MainHost/Phase analysis"; "**/Label" searches any depth.
#include "ui_test_host.h"

#include "ui_test_modes.h"

#include "imgui.h"
#include "imgui_te_engine.h"
#include "imgui_te_context.h"

#include "ucuda/application/app_model.h"

#include <cstring>

namespace ui_test {

void register_sweep_tests(ImGuiTestEngine* engine);   // ui_tests_sweep.cpp

void register_tests(ImGuiTestEngine* engine) {
    ImGuiTest* t = nullptr;

    t = IM_REGISTER_TEST(engine, "smoke", "startup");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        ctx->Yield(5);
        IM_CHECK(ctx->WindowInfo("//MainHost").Window != nullptr);
        IM_CHECK(ctx->ItemExists("//MainHost/Library"));
        IM_CHECK(ctx->ItemExists("//MainHost/Settings"));
    };

    // Every visible mode button switches AppModel::app_mode and the next frames draw without errors.
    t = IM_REGISTER_TEST(engine, "smoke", "mode_tabs");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* model = app_model();
        IM_CHECK_NO_RET(model != nullptr);
        if (!model) return;
        const AppModel::AppMode initial = model->app_mode;
        ctx->SetRef("//MainHost");
        for (const ModeButton& b : kModeButtons) {
            if (!ctx->ItemExists(b.label)) {
                ctx->LogInfo("mode button '%s' hidden, skipped", b.label);
                continue;
            }
            ctx->ItemClick(b.label);
            ctx->Yield(3);
            IM_CHECK_EQ((int)model->app_mode, (int)b.mode);
        }
        for (const ModeButton& b : kModeButtons)
            if (b.mode == initial && ctx->ItemExists(b.label)) ctx->ItemClick(b.label);
    };

    // Parametric/Bifurcation "Param lo" -> "Apply to..." -> All -> "Apply to N selected":
    // every bifurcation config (active or not) receives the value. Needs a system with
    // two bifurcation configs (Chen: run with a script that loads it).
    t = IM_REGISTER_TEST(engine, "parametric", "apply_param_lo_to_all");
    t->TestFunc = [](ImGuiTestContext* ctx) {
        AppModel* m = app_model();
        IM_CHECK_NO_RET(m != nullptr);
        if (!m) return;
        ctx->SetRef("//MainHost");
        ctx->ItemClick("Parametric");
        ctx->Yield(3);
        ctx->SetRef("//Parametric Controls");
        auto log_configs = [&](const char* when) {
            for (size_t i = 0; i < m->bifurcation_session.diagrams.size(); ++i) {
                const auto& d = m->bifurcation_session.diagrams[i];
                ctx->LogInfo("[%s] bif config %d \"%s\": lo=%s hi=%s 2d=%d", when, (int)i, d.label.c_str(),
                             d.param_lo_text.c_str(), d.param_hi_text.c_str(), (int)d.mode_2d);
            }
        };
        log_configs("before");
        ctx->ItemInputValue("**/Param lo", "31");
        ctx->ItemClick("**/Param lo", ImGuiMouseButton_Right);
        ctx->Yield(2);
        // The "active config of each tab" item is gone: targets are chosen only in "Apply to...".
        ImGuiTestItemList items;
        ctx->GatherItems(&items, "//##Popup_a4503e11", 1);
        for (const ImGuiTestItemInfo& it : items)
            IM_CHECK(std::strncmp(it.DebugLabel, "Apply to active", 15) != 0 &&
                     std::strncmp(it.DebugLabel, "Apply to all calc", 17) != 0);
        ctx->ItemClick("//##Popup_a4503e11/Apply to...");
        ctx->Yield(2);
        ctx->ItemClick("//###Menu_00/All");
        ctx->Yield(2);
        items.Clear();
        ctx->GatherItems(&items, "//###Menu_00", 1);
        ImGuiID apply_id = 0;
        for (const ImGuiTestItemInfo& it : items)
            if (std::strncmp(it.DebugLabel, "Apply to ", 9) == 0) { apply_id = it.ID; ctx->LogInfo("button: %s", it.DebugLabel); }
        IM_CHECK_NO_RET(apply_id != 0);
        if (apply_id) ctx->ItemClick(apply_id);
        ctx->Yield(3);
        log_configs("after");
        const auto& ds = m->bifurcation_session.diagrams;
        IM_CHECK_NO_RET(ds.size() >= 2);
        for (const auto& d : ds) IM_CHECK_STR_EQ(d.param_lo_text.c_str(), "31");
    };

    register_sweep_tests(engine);
}

} // namespace ui_test
