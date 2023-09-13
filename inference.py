"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse

from diffusers import DiffusionPipeline, DDIMScheduler
import torch


class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument(
            "--prompt", type=str, default="a photo of <asset0> at the beach"
        )
        parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
        )
        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        images = self.pipeline(prompts).images
        images[0].save(self.args.output_path)


if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save(
        prompts=[break_a_scene_inference.args.prompt]
    )
