"""SegFormer-B3 version of the city-disjoint Inria building segmentation pipeline.

This entry point reuses the identical data split, augmentations, loss, optimizer, checkpoints, and
evaluation flow in train_new.py. Only the segmentation architecture is replaced with SegFormer-B3
(HuggingFace `nvidia/mit-b3` encoder, fine-tuned with a randomly initialized 1-channel decode head).

Run:
    python train_segformer.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

import train_base as pipeline

SEGFORMER_CHECKPOINT = "nvidia/mit-b3"


class SegformerWrapper(nn.Module):
    """Adapts HuggingFace SegFormer to the {"out", "aux"} interface used by run_epoch."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, images):
        logits = self.hf_model(pixel_values=images).logits
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits, "aux": None}


def build_model(pretrained):
    if pretrained:
        # decode head is absent from the encoder-only checkpoint, so it is randomly initialized here.
        model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_CHECKPOINT, num_labels=1)
        print(f"Using ImageNet-pretrained SegFormer-B3 encoder ({SEGFORMER_CHECKPOINT}).")
    else:
        config = SegformerConfig.from_pretrained(SEGFORMER_CHECKPOINT, num_labels=1)
        model = SegformerForSemanticSegmentation(config)
        print("Training SegFormer-B3 from random initialization.")
    return SegformerWrapper(model)


def main():
    pipeline.build_model = build_model
    if "--out-dir" not in sys.argv:
        sys.argv.extend(("--out-dir", "outputs_city_split_segformer"))
    pipeline.main()


if __name__ == "__main__":
    main()


