#!/usr/bin/env python3
import sys, os, base64, io, traceback

# UTF-8 на stdout, чтобы LaTeX не ломался на Windows-консоли
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ocr_error.log")
def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def load_model():
    from pix2text import Pix2Text
    return Pix2Text.from_config()

def recognize(model, png_bytes):
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes))
    latex = model.recognize_formula(img)
    return latex if isinstance(latex, str) else str(latex)

def main():
    log("=== server start, cwd=" + os.getcwd() + ", python=" + sys.executable + " ===")
    try:
        model = load_model()
    except Exception as e:
        log("LOAD FAILED:\n" + traceback.format_exc())
        sys.stdout.write("FATAL " + str(e).replace("\n"," ") + "\n")
        sys.stdout.flush()
        return
    sys.stdout.write("READY\n"); sys.stdout.flush()
    log("model loaded, READY sent")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if line == "QUIT": break
        if line == "PING":
            sys.stdout.write("OK pong\n"); sys.stdout.flush(); continue
        try:
            png = base64.b64decode(line, validate=True)
            latex = recognize(model, png).replace("\n"," ").strip()
            sys.stdout.write("OK " + latex + "\n")
        except Exception as e:
            log("RECOGNIZE FAILED:\n" + traceback.format_exc())
            sys.stdout.write("ERR " + str(e).replace("\n"," ") + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
