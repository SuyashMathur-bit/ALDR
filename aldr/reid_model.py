"""
ALDR Vehicle Re-ID Model Module
Architecture, Embedding Extractor, Cosine Similarity (Stage 2)
"""

import os
from pathlib import Path
from typing import List, Tuple, Union, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from aldr.config import (
    REID_IMG_HEIGHT,
    REID_IMG_WIDTH,
    REID_EMBEDDING_DIM,
    REID_EMBEDDING_MODEL_PATH
)


def build_reid_models(
    num_classes: int = 460,
    freeze_backbone: bool = True
) -> Tuple[Model, Model]:
    """
    Constructs the Vehicle Re-ID Architecture:
    ResNet50 (ImageNet, frozen) -> GAP -> Dense(512) -> BatchNorm -> L2-Norm -> Softmax(num_classes)
    Returns:
        (training_model, embedding_model)
    """
    inputs = tf.keras.Input(
        shape=(REID_IMG_HEIGHT, REID_IMG_WIDTH, 3),
        name="vehicle_image"
    )

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(REID_IMG_HEIGHT, REID_IMG_WIDTH, 3)
    )
    base_model.trainable = not freeze_backbone

    features = base_model(inputs, training=False)
    features = layers.GlobalAveragePooling2D(name="global_average_pooling")(features)

    embedding = layers.Dense(
        REID_EMBEDDING_DIM,
        activation=None,
        name="embedding"
    )(features)
    embedding = layers.BatchNormalization(name="embedding_batch_norm")(embedding)
    normalized_embedding = layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name="normalized_embedding"
    )(embedding)

    classification_output = layers.Dense(
        num_classes,
        activation="softmax",
        name="vehicle_classifier"
    )(normalized_embedding)

    training_model = Model(
        inputs=inputs,
        outputs=classification_output,
        name="Vehicle_ReID_Classifier"
    )

    embedding_model = Model(
        inputs=inputs,
        outputs=normalized_embedding,
        name="Vehicle_ReID_Embedding_Model"
    )

    return training_model, embedding_model


def preprocess_crop(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Resizes a vehicle crop to 224x112 and applies ResNet50 preprocessing.
    """
    img = tf.image.resize(crop_rgb, [REID_IMG_HEIGHT, REID_IMG_WIDTH])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img.numpy()


class ReIDExtractor:
    """Extracts 512-dimensional L2-normalized embeddings for vehicle crops."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_training_classes: int = 460
    ):
        self.embedding_model = None

        if model_path and os.path.exists(str(model_path)):
            try:
                print(f"[ALDR Re-ID] Loading saved embedding model from {model_path}...")
                self.embedding_model = tf.keras.models.load_model(str(model_path), compile=False, safe_mode=False)
            except Exception as e:
                print(f"[ALDR Re-ID] Could not load model directly ({e}). Rebuilding architecture...")

        if self.embedding_model is None:
            print("[ALDR Re-ID] Initializing ImageNet ResNet50 Re-ID backbone...")
            _, self.embedding_model = build_reid_models(
                num_classes=num_training_classes,
                freeze_backbone=True
            )

    def extract_embedding(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Extracts a normalized 512-D embedding for a single crop."""
        preprocessed = preprocess_crop(crop_rgb)
        batch = np.expand_dims(preprocessed, axis=0)
        emb = self.embedding_model(batch, training=False).numpy()[0]
        return emb

    def extract_batch(self, crops_rgb: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """Extracts normalized embeddings for a batch of crops."""
        if not crops_rgb:
            return np.empty((0, REID_EMBEDDING_DIM), dtype=np.float32)

        preprocessed_list = [preprocess_crop(crop) for crop in crops_rgb]
        preprocessed_array = np.array(preprocessed_list, dtype=np.float32)

        embeddings = []
        for i in range(0, len(preprocessed_array), batch_size):
            chunk = preprocessed_array[i:i + batch_size]
            emb_chunk = self.embedding_model(chunk, training=False).numpy()
            embeddings.append(emb_chunk)

        return np.vstack(embeddings)

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two normalized vectors."""
        # For L2 normalized vectors: dot product == cosine similarity
        return float(np.dot(emb1, emb2))

    @staticmethod
    def compute_similarity_matrix(embs1: np.ndarray, embs2: np.ndarray) -> np.ndarray:
        """Pairwise cosine similarity matrix (N, M)."""
        if len(embs1) == 0 or len(embs2) == 0:
            return np.empty((len(embs1), len(embs2)), dtype=np.float32)
        return np.matmul(embs1, embs2.T)
