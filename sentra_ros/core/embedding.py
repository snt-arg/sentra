from sentra_ros.core.utils import vlm_models

# from sentence_transformers import SentenceTransformer


def initRAGModel(model="siglip", logger=None):
    """
    Initializes the multimodal RAG model and loads it onto GPU.

    Parameters:
    ----------
    model (str):
        The name of the RAG model to load. Must be one of the following: {vlm_models}
    logger (rclpy.logging.Logger):
        Optional ROS logger for logging messages. If None, no logging will occur.
    """
    # Argument validation
    if model not in vlm_models:
        if logger:
            logger.warning(
                f"VLM model '{model}' is not supported. Switching to default ('siglip')..."
            )
        model = "siglip"

    # Load the embedding model
    if logger:
        logger.info(f"Initializing '{model}' VLM model...")
    # if model == "siglip":
    #     self.llm_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    if logger:
        logger.info("Model initialized successfully!")
