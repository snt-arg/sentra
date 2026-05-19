import gc
import torch
import torchvision

uiColors = {
    "pink": [245, 40, 130],
    "orange": [245, 135, 40],
}


def monitorParams(logger=None):
    """
    Prints a summary of various parameters
    """
    if logger:
        logger.info(f"* CUDA? {torch.cuda.is_available()}")
        logger.info(f"* PyTorch {torch.__version__}")
        logger.info(f"* Torchvision {torchvision.__version__}")


def cleanMemory(logger=None):
    """
    Cleans memory on GPU
    """
    torch.cuda.empty_cache()
    gc.collect()
    if logger:
        logger.info("* Memory cleaned! \n")