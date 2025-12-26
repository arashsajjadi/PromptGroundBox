from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
from PIL import Image

from promptgroundboxbench.engines.grounding_dino import GroundingDINOEngine
from promptgroundboxbench.utils.draw import draw_boxes
from promptgroundboxbench.utils.prompt import parse_prompt
from promptgroundboxbench.utils.timing import timed_call


@dataclass
class DemoState:
    engine: GroundingDINOEngine
    sync_cuda: bool


def build_demo(engine: GroundingDINOEngine, sync_cuda: bool) -> gr.Blocks:
    state = DemoState(engine=engine, sync_cuda=sync_cuda)

    with gr.Blocks(title="PromptGroundBoxBench Demo") as demo:
        gr.Markdown("# PromptGroundBoxBench\nGrounding DINO prompt driven box detection")

        with gr.Row():
            in_img = gr.Image(type="pil", label="Input image")
            out_img = gr.Image(type="pil", label="Output with boxes")

        prompt = gr.Textbox(
            label="Prompt labels",
            value="a person. a chair. a laptop.",
            lines=2,
        )
        latency = gr.Number(label="Latency (ms)", precision=2)
        btn = gr.Button("Run Grounding DINO")

        def _run(img: Image.Image | None, prompt_text: str) -> tuple[Image.Image | None, float]:
            if img is None:
                return None, 0.0
            labels = parse_prompt(prompt_text)
            if not labels:
                raise gr.Error("Prompt is empty. Provide labels like 'a person. a chair.'")
            det, dt = timed_call(lambda: state.engine.predict(img, labels), sync_cuda=state.sync_cuda)
            vis = draw_boxes(img, det.boxes_xyxy, det.labels, det.scores.tolist())
            return vis, dt * 1000.0

        btn.click(_run, inputs=[in_img, prompt], outputs=[out_img, latency])

    return demo
