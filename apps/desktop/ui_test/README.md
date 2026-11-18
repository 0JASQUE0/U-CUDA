# UI test mode (Dear ImGui Test Engine)

The desktop app can be built with [Dear ImGui Test Engine](https://github.com/ocornut/imgui_test_engine)
compiled in. The engine then drives the real GUI by injecting mouse and keyboard input into ImGui and
finding widgets by their ImGui IDs. It needs no OS-level mouse control, and screenshots come from the
app's own framebuffer.

Only the CMake option `U_CUDA_UI_TESTS` turns this on (default `OFF`). The classic `U-CUDA.sln` build and
ordinary CMake test builds contain neither the engine nor these sources. In `app_main.cpp` every hook
sits under `#ifdef UCUDA_UI_TESTS`.

The engine is pinned to tag v1.92.8, which matches imgui `v1.92.8-docking` in `third_party/CMakeLists.txt`.
When imgui is updated, update the engine too. Licence: `imgui_test_engine/LICENSE.txt`. It is free for
individuals, educational use, open source and companies with turnover under USD 2M.

## Build

```powershell
$cm = "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
& $cm --preset windows -B $env:TEMP\ucb-ui -DU_CUDA_UI_TESTS=ON
& $cm --build $env:TEMP\ucb-ui --config Release --target U-CUDA
```

The exe and its runtime (kernels, library template, DLLs) end up in `$env:TEMP\ucb-ui\bin\Release`.

## Run

```
U-CUDA.exe --ui-script s.uit --ui-out <dir> --ui-fresh     # command script
U-CUDA.exe --ui-test all                                   # compiled tests (ui_tests.cpp)
U-CUDA.exe --ui-test smoke --ui-speed normal               # name filter, watchable speed
U-CUDA.exe --ui-list                                       # list compiled tests
U-CUDA.exe --ui-show                                       # interactive Test Engine window
```

| Option | Meaning |
|---|---|
| `--ui-test <filter>` | queue compiled tests; `all`, or the engine's filter: comma list of name substrings, `^name` prefix, `-name` exclude |
| `--ui-script <file>` | run a command script as test `script/<file name>` |
| `--ui-out <dir>` | screenshots (relative names) and `results.xml` (JUnit), default `ui_out` |
| `--ui-size WxH` | window size, default 1600x900 |
| `--ui-speed fast\|normal\|cinematic` | fast teleports the mouse and skips delays |
| `--ui-fresh` | delete `_app_config.json` and `imgui.ini` next to the exe and re-copy the isolated library (default layout and settings) |
| `--ui-keep` | do not exit after the queue is done |
| `--ui-show` | draw the Test Engine window |

**Library isolation.** The app saves tab sessions into the system library on every change. In
UI test mode `U_CUDA_LIBRARY` points to `<exe>/ui_library`, a copy of `U-CUDA/library` (the source is
only read; `--ui-fresh` re-copies it). If the caller has already set `U_CUDA_LIBRARY`, that value is used.
The log starts with `[ui] library (...)`.

The app exits by itself when the queue is empty. Exit code is 0 only if every queued test passed.
Output on stdout:
- `[ui] <line>: <command>` echoes each script command; `[ui]   ...` lines are command output (dump, read);
- the engine log uses `KO`/`OK` lines and shows the error and its location;
- `[ui-result] PASS|FAIL <test>` for each test, then `[ui-summary] tested N, passed M, not run K`.

## Item references

A reference is an ImGui ID path, the same one the code builds with `Begin`, `PushID` and labels:
- `//MainHost/Phase analysis`: absolute, starting at a window name;
- `Phase analysis`: relative to the current `ref` (scripts start with `ref //MainHost`);
- `**/Run`: wildcard, `Run` at any depth under the current ref;
- `0x1A2B3C4D`: an item ID taken from `dump` output;
- a label `Run##bif` must be written with its `##` suffix. Window names include their `##` part too.

Child windows have mangled names. Use `windows all` to see them, or use the ID from `dump`.
Docked windows keep their own name (`//Controls`). An open combo is a popup window `//##Combo_00`.
`dump` shows items that were clipped out of view (`hidden`) with an empty label. Scroll the window
(`scroll_y`) and dump again to read their labels. Many fields have no visible label, only an ID
(`##sh`); the text next to them is a separate `Text` item.

## Script commands (`*.uit`)

One command per line. `#` starts a comment line. Arguments containing spaces go in `"double quotes"`,
and `\"` escapes a quote inside them.

| Command | Action |
|---|---|
| `ref R` | set the base for relative references |
| `click R` / `rclick R` / `dclick R` / `hover R` | mouse on item |
| `check R` / `uncheck R` | set checkbox / checkable menu item |
| `open R` / `close R` | tree node, collapsing header, menu |
| `input R TEXT` | click a field, replace its text with TEXT, press Enter |
| `type TEXT` | type characters into the focused field |
| `key CHORD [N]` | e.g. `Enter`, `Escape`, `Ctrl+Z`, `Ctrl+Shift+T`, `F5` (ImGui key names) |
| `menu PATH` | `menu "File/Save"` relative to the current ref window |
| `combo PATH` | `combo "Method/RK4"`: open combo `Method`, pick `RK4` |
| `tab_close R`, `focus W`, `resize W w h`, `drag R dx dy` | tab / window helpers |
| `scroll_y W v`, `scroll_to R` | scroll a window / scroll an item into view |
| `mouse_void` | move the mouse to the bottom-right corner of the viewport (MainHost covers it all, so there is no empty space) |
| `popup_close` / `popup_close_all` | close the top popup / all popups (`key Escape` also closes the top non-modal popup, see `draw_gui`) |
| `find W PREFIX` | print the first item under W whose label starts with PREFIX |
| `click_prefix W PREFIX` / `rclick_prefix W PREFIX` | click it: for labels with counters, e.g. `"Reset to default (0)"` |
| `yield [N]` | let N frames pass |
| `sleep SEC` | wait real time (frames keep running) |
| `wait_exists R [SEC]` / `wait_missing R [SEC]` | poll until the item appears or disappears (default 10 s) |
| `wait_idle [SEC]` | wait until no analysis session is computing (default 120 s) |
| `expect_exists R` / `expect_missing R` | fail if not |
| `expect_checked R` / `expect_unchecked R`, `expect_open R` / `expect_closed R` | state checks |
| `read R` | print the value of an input/slider/drag (via select + copy) |
| `expect_value R TEXT` | fail unless that value equals TEXT exactly |
| `expect_mode NAME` | top tab: Library, Analysis, Parametric, Dft1D, Basins, FastSync, Custom, Settings, Order, Network, Neural, Tree |
| `rect R` | print one item: ID, label, rectangle, flags |
| `dump [R [DEPTH]]` | list items under R (default the current ref, depth 1); without a ref, list windows |
| `windows [all]` | visible windows (`all` includes child windows) |
| `screenshot FILE [W]` | PNG of the whole viewport or of window W; FILE relative to `--ui-out` |
| `log TEXT`, `fail TEXT` | print / fail deliberately |

The script stops at the first failing command. The engine log names that line.
With both `--ui-script` and `--ui-test`, the script runs first, so it can prepare the state (load a system).

Tab items (`BeginTabItem`) and fields under a `PushID` scope are not found by plain or `**/` label paths,
and their labels in `dump` are empty. Address fields by the ID from `dump` (`dump W 3` shows nested items).
For tabs whose label encodes state, write a compiled test.

Native file dialogs (Open/Save/image pickers) block the app. Scripts must not click buttons that open them.

## Compiled tests

`ui_tests.cpp` holds the regression tests. See the comment at its top for how to add one.
`ui_test::app_model()` gives a test direct access to `AppModel`, so it can check state rather than pixels.
