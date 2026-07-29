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

import os
import gc
import torch
import shutil
import datetime
import torchvision

ui_colors = {
    "pink": [245, 40, 130],
    "orange": [245, 135, 40],
}

vlm_models = ["siglip"]


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


def timestamp_to_time(timestamp) -> str:
    """Converts a ROS float/string epoch timestamp into a highly readable time-only format

    (HH:MM:SS.mmm).
    """
    try:
        # Cover both string ("1784116342.747868895") and raw float inputs
        seconds_float = float(timestamp)
        # Convert to a local datetime object
        dt_object = datetime.datetime.fromtimestamp(seconds_float)
        # Format to time-only: Minutes:Seconds.Milliseconds (3 decimal places)
        return dt_object.strftime("%M:%S.%f")[:-3]
    except (ValueError, TypeError) as e:
        # Fallback to a string conversion if input is corrupted
        return f"Invalid Stamp ({timestamp})"


def clearKeyFramesDir(path: str, logger=None):
    """
    Clears the keyframes directory by deleting all files and subdirectories within it.

    Parameters
    ----------
    path (str):
        The path to the keyframes directory to be cleared.
    logger (rclpy.logging.Logger, optional):
        ROS logger for logging messages. If provided, logs the clearing process.
    """
    # Check if directory is empty
    if not os.listdir(path):
        if logger:
            logger.info(f"Keyframes directory '{path}' is already empty.\n")
        return

    # Otherwise, proceed to clear the directory
    if logger:
        logger.info(f"Clearing keyframes directory '{path}' ...")
    # Empty the directory by deleting all files and subdirectories
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            if logger:
                logger.error(f"* Failed to delete {file_path}. Reason: {e}")
    # Log the completion of the clearing process
    if logger:
        logger.info(f"Keyframes directory '{path}' cleared!\n")
