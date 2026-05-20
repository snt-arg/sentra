import torch
from sentra_ros.core.utils import vlm_models
from transformers import AutoProcessor, AutoModel


class MultimodalEncoder:
    def __init__(self, backbone="siglip", logger=None):
        """
        Initializes the multimodal RAG model and loads it onto GPU.

        Parameters:
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
            self.logger.info(f"Multimodal Encoder {backbone} initialized successfully!")

    # def get_text_embedding(self, text: str) -> np.ndarray:
    #     """
    #     Extracts embedding from a user text query.
    #     Returns a normalized 1D numpy array.
    #     """
    #     # Preprocess text and push tokens to GPU
    #     inputs = self.processor(text=[text], padding="max_length", return_tensors="pt").to(self.device)

    #     with torch.no_grad():
    #         # Extract text features
    #         text_features = self.model.get_text_features(**inputs)
    #         # Normalize to unit length (critical for exact Cosine Similarity matching later)
    #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #     return text_features.cpu().numpy()[0]

    # def get_image_embedding(self, pil_image: Image.Image) -> np.ndarray:
    #     """
    #     Extracts embedding from an RGB image (PIL Image object).
    #     Returns a normalized 1D numpy array.
    #     """
    #     # Preprocess image (resizes to 224x224, normalizes channels) and push to GPU
    #     inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

    #     with torch.no_grad():
    #         # Extract image features
    #         image_features = self.model.get_image_features(**inputs)
    #         # Normalize to unit length
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    #     return image_features.cpu().numpy()[0]
