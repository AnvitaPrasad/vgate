"""The log view of VGate as a CPU-only Hugging Face Space (assembled by scripts/build_space.sh; not published yet)."""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location("vgate_demo", pathlib.Path(__file__).parent / "demo" / "app.py")
vgate_demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vgate_demo)
demo = vgate_demo.log_only_ui(note="This hosted page shows the log view only; the other views need the case study's per-query "
                                   "outputs or a GPU (see the repository).")

if __name__ == "__main__":
    demo.launch()
