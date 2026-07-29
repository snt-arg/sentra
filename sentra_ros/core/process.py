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

import numpy as np
import pandas as pd


def searchKeyframes(
    query_embedding: np.ndarray,
    kf_visual_df: pd.DataFrame,
    logger,
    top_k: int = 5,
    min_similarity: float = 0.30,
) -> pd.DataFrame:
    """
    Computes cosine similarity between a query embedding and keyframe embeddings.

    Parameters
    ----------
    query_embedding (np.ndarray):
        The embedding vector of the query (1D array).
    kf_visual_df (pd.DataFrame):
        DataFrame containing stored keyframe embeddings with columns:
        ['kf_id', 'timestamp', 'embedding'].
    logger (rclpy.logging.Logger):
        ROS logger for logging messages.
    top_k (int):
        The number of top matches to return.
    min_similarity (float):
        The minimum cosine similarity threshold for a match to be considered relevant.

    Returns
    -------
    pd.DataFrame
        The top-k matching keyframes above the min_similarity threshold.
    """
    if kf_visual_df.empty:
        logger.warning("No keyframe visual embeddings stored yet!")
        return pd.DataFrame()

    # Convert the list of embeddings stored in DataFrame to a 2D NumPy array (N, D)
    visual_matrix = np.array(kf_visual_df["embedding"].tolist())

    # Cosine similarity via dot product (since vectors are L2-normalized)
    similarity_scores = np.dot(visual_matrix, query_embedding)

    # Copy DataFrame and assign similarity scores
    results_df = kf_visual_df.copy()
    results_df["similarity"] = similarity_scores

    # Sort and filter top matches
    results_df = results_df.sort_values(by="similarity", ascending=False)
    matched_results = results_df[results_df["similarity"] >= min_similarity].head(top_k)

    return matched_results
