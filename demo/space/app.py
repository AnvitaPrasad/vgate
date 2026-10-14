"""VGate on a Hugging Face ZeroGPU Space: the Live tab of demo/app.py behind `@spaces.GPU`.

STATUS: written from the ZeroGPU documentation (https://huggingface.co/docs/hub/spaces-zerogpu), NOT yet run on a
Space. Before relying on it, run the check in DEPLOY.md (gate scores on logged questions must match the T4 runs).

Layout expected in the Space repository:
  app.py, requirements.txt, README.md   this folder
  vgate/                                copy of src/vgate (with gates/)
  demo_app.py                           copy of demo/app.py (live_views, wiki_url)
  index/passage_emb.npy, passages.jsonl built by `python -m vgate.live_eval --build_index ...` (job J9)
"""
import gradio as gr
import spaces

from demo_app import FOOTER, live_views
from vgate.live import GATES_DIR, Gates, LivePipeline, Retriever
from vgate.models import VLM, prepare_image

MODEL = "Qwen3-VL-4B-Instruct"
# ZeroGPU: models are placed on `cuda` at module level (a CUDA emulation is active outside @spaces.GPU functions).
# Gates refit on logs of this retriever (PREREG v2g): gates fitted with the benchmarks' retrievers do not transfer.
PIPE = LivePipeline(VLM("Qwen/" + MODEL, dtype="float16", device="cuda"), Gates(GATES_DIR / f"{MODEL}__live.npz"),
                    Retriever("index", device="cpu"))


@spaces.GPU(duration=30)  # a visitor's daily quota is charged by this declared duration
def ask(image, question, run_all):
    res = PIPE.answer(prepare_image(image), question, run_all=bool(run_all))
    res.pop("hidden", None)  # return plain Python objects only (results cross a process boundary)
    return res


def answer(image, question, run_all):
    if image is None or not (question or "").strip():
        return "Upload an image and type a question.", None, None, ""
    return live_views(ask(image, question.strip(), run_all))


with gr.Blocks(title="VGate") as ui:
    gr.Markdown("## VGate — retrieve only when it helps\nUpload an image and ask a question. The model answers without "
                "evidence; VGate-pre decides whether to search; VGate-full decides whether the model reads what the "
                "retriever returned. Each pass is charged to the cost table.")
    with gr.Row():
        img = gr.Image(label="your image", type="pil", height=300)
        with gr.Column():
            q = gr.Textbox(label="question")
            run_all = gr.Checkbox(value=False, label="also run the passes VGate skips (to compare with always-retrieve)")
            go = gr.Button("Ask")
            dec = gr.Markdown()
    ans = gr.Dataframe(label="answers and cost")
    ev = gr.Dataframe(label="retrieved evidence (top-5)")
    t = gr.Markdown()
    gr.Markdown(f"<small>{FOOTER}</small>")
    go.click(answer, [img, q, run_all], [dec, ans, ev, t])

if __name__ == "__main__":
    ui.launch()
