"""Runs VGate's log view (demo/app.py: byo) in the browser on Pyodide, without Gradio: the same function the web
application calls, with the interface library replaced by a stub. Loaded by index.html."""
import base64
import importlib.util
import io
import json
import sys
import types


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return self

    def __getattr__(self, name):
        return _Dummy()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


sys.modules["gradio"] = _Stub("gradio")  # demo/app.py imports it for its views; the computation below does not use it
spec = importlib.util.spec_from_file_location("vgate_demo", "demo/app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def texts():
    return json.dumps(dict(help=app.BYO_HELP, audit_label=app.BYO_AUDIT_LABEL, footer="VGate, MIT licence. Sample logs: outcomes and signals only."))


def run(path):
    """The log view on one CSV file: the summary line, the frontier plot (PNG), the audit and planner tables (HTML) and the
    fitted gate (JSON text), as one JSON string."""
    import matplotlib.pyplot as plt
    fig, md, audit, plan, gate_path = app.byo(path)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    def html(d):
        if d is None:
            return None
        d = d.copy()
        if "acc." in d:  # accuracies with one decimal throughout (34.0, not 34)
            d["acc."] = d["acc."].map(lambda v: f"{v:.1f}")
        return d.to_html(index=False, border=0, na_rep="", float_format=lambda v: f"{v:g}")
    return json.dumps(dict(md=md, plot=base64.b64encode(buf.getvalue()).decode(), audit=html(audit), plan=html(plan),
                           gate=open(gate_path).read() if gate_path else None))
