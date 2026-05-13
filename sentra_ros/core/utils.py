import gc
import torch
import torchvision


def monitorParams():
    """
    Prints a summary of various parameters
    """
    print("\nGeneral status check ...")
    print("* CUDA?", torch.cuda.is_available())
    print("* PyTorch", torch.__version__)
    print("* Torchvision", torchvision.__version__)
    print()


def cleanMemory():
    """
    Cleans memory on GPU
    """
    torch.cuda.empty_cache()
    gc.collect()
