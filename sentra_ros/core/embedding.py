"""
⚜️ Sentra ⚜️
------------
* SPDX-FileCopyrightText: 2023-2026 University of Luxembourg
* SPDX-License-Identifier: SDF26-0040
* © 2023-2026 University of Luxembourg
* Developed by: Ali Tourani at SnT/ARG
* Sentra is licensed under the GPL 3.0 License
* (Check LICENSE file for details)
"""

import torch
import numpy as np
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from sentra_ros.core.utils import vlm_models
from transformers import AutoProcessor, AutoModel


class MultimodalEncoder:
    def __init__(self, backbone="siglip", logger=None):
        """
        Initializes the multimodal RAG model and loads it onto GPU.

        Parameters
        ----------
        backbone (str):
            The name of the backbone model to load. Must be one of the following: {vlm_models}
        logger (rclpy.logging.Logger):
            Optional ROS logger for logging messages. If None, no logging will occur.
        """
        # Argument validation
        self.logger = logger
        if backbone not in vlm_models:
            if self.logger:
                self.logger.warning(
                    f"VLM model '{backbone}' is not supported. Switching to default ('siglip')..."
                )
            backbone = "siglip"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the embedding model
        if self.logger:
            self.logger.info(f"Initializing '{backbone}' VLM model...")

        if backbone == "siglip":
            try:
                # Using Google's SigLIP base model
                self.processor = AutoProcessor.from_pretrained(
                    "google/siglip-base-patch16-224"
                )
                # Load the model and transfer it to the GPU
                self.model = AutoModel.from_pretrained(
                    "google/siglip-base-patch16-224"
                ).to(self.device)
                # Set the model to evaluation mode (disables dropout, etc.)
                self.model.eval()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to load '{backbone}' model: {e}")
                raise e

        if self.logger:
            self.logger.info(
                f"Multimodal Encoder '{backbone}' initialized successfully!"
            )

    def get_text_embedding(self, text: str) -> np.ndarray:
        """
        Extracts embedding from a user text query and returns it as a normalized 1D numpy array.

        Parameters
        ----------
        text (str): The input text query to be embedded.
        """
        # Preprocess text and push tokens to GPU
        inputs = self.processor(
            text=[text],
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            # Extract the raw tensor data, convert to float, and detach safely
            text_features = self.model.get_text_features(**inputs)
            # Handle HuggingFace container wrapper vs raw tensor safely
            if (
                hasattr(text_features, "pooler_output")
                and text_features.pooler_output is not None
            ):
                text_features = text_features.pooler_output
            elif hasattr(text_features, "to_tuple"):
                text_features = text_features.to_tuple()[0]
            # If 3D tensor (1, seq_len, dim), mean-pool across sequence dimension
            if text_features.ndim == 3:
                text_features = text_features[:, 0, :]
            # L2-normalize to unit length (critical for exact Cosine Similarity matching later)
            text_features = text_features / torch.norm(
                text_features, dim=-1, keepdim=True
            )

        return text_features.cpu().numpy()[0]

    def get_image_embedding(self, image_msg: Image) -> np.ndarray:
        """
        Extracts embedding from an RGB image (PIL Image object).
        Returns a normalized 1D numpy array.
        """
        try:
            # Get the OpenCV image from the ROS Image message
            img_array = np.frombuffer(image_msg.data, dtype=np.uint8)
            img_matrix = img_array.reshape((image_msg.height, image_msg.width, 3))

            # Convert to RGB representation
            if "bgr" in image_msg.encoding.lower():
                img_matrix = img_matrix[:, :, ::-1]

            # Cast natively to PIL
            pil_img = PILImage.fromarray(img_matrix)

            # Preprocess text and push tokens to GPU
            inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)

            with torch.no_grad():
                img_features = self.model.get_image_features(**inputs)

                # Extract underlying tensor safely depending on Hugging Face version output wrap
                if (
                    hasattr(img_features, "pooler_output")
                    and img_features.pooler_output is not None
                ):
                    img_features = img_features.pooler_output
                elif hasattr(img_features, "to_tuple"):
                    img_features = img_features.to_tuple()[0]

                # If 3D tensor (1, seq_len, dim), mean-pool across sequence dimension
                if img_features.ndim == 3:
                    img_features = img_features.mean(dim=1)

                # L2-normalize to unit length
                img_features = img_features / torch.norm(
                    img_features, dim=-1, keepdim=True
                )

            # Extract embeddings
            return img_features.cpu().numpy()[0]

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to process callback image: {e}")
            return []
