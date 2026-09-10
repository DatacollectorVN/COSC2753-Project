"""Feature extractor using pretrained CLIP model from HuggingFace.

Wraps the CLIP vision encoder to produce normalized image embeddings.
The processor handles all image preprocessing (resize, normalize, etc.)
so no manual transform pipeline is needed.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPFeatureExtractor:
    """Extract image embeddings using a pretrained CLIP model.

    Attributes:
        model_name: HuggingFace model identifier.
        device: torch device for inference.
        embedding_dim: dimensionality of output vectors.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: torch.device | None = None):
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.projection_dim

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract a single image embedding.

        Args:
            image: PIL Image (RGB).

        Returns:
            L2-normalized embedding vector of shape (embedding_dim,).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model.get_image_features(pixel_values=pixel_values)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().squeeze(0)

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        """Extract embeddings for a batch of images.

        Args:
            images: list of PIL Images (RGB).

        Returns:
            L2-normalized embeddings of shape (N, embedding_dim).
        """
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        out = self.model.get_image_features(pixel_values=pixel_values)
        features = out.pooler_output if hasattr(out, "pooler_output") else out
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()
