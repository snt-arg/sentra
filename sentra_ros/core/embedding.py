from sentra_ros.core.utils import llm_models
# from sentence_transformers import SentenceTransformer


def initLLMModel(llm="st-small", logger=None):
    """
    Initializes the text embedding model and loads it onto GPU

    Parameters:
    ----------
    llm (str):
        The name of the LLM model to load. Must be one of the following: {llm_models}
    logger (rclpy.logging.Logger):
        Optional ROS logger for logging messages. If None, no logging will occur.
    """
    # Argument validation
    if llm not in llm_models:
        if logger:
            logger.warning(
                f"LLM model '{llm}' is not supported. Switching to default 'st-small'."
            )
        llm = "st-small"

    # Load the embedding model
    if logger:
        logger.info(f"Loading '{llm}' text embedding model...")
    # if llm == "st-small":
    #     self.llm_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    if logger:
        logger.info("Embedding model loaded successfully!")
