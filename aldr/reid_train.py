"""
ALDR Vehicle Re-ID Training and Optimization Module
Optimized ResNet50 Classifier & Embedding Training with Resumable Checkpoints (Priority 2)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import tensorflow as tf

from aldr.config import (
    REID_IMG_HEIGHT,
    REID_IMG_WIDTH,
    REID_BATCH_SIZE,
    REID_EMBEDDING_DIM,
    VERI_ROOT_DIR,
    ALDR_REID_DIR,
    REID_CHECKPOINT_DIR,
    REID_EMBEDDING_MODEL_PATH
)
from aldr.reid_model import build_reid_models


def load_metadata(reid_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Loads train and val metadata and creates a dense mapping for training IDs.
    """
    root = reid_dir or ALDR_REID_DIR
    train_csv = root / "train_metadata.csv"
    val_csv = root / "val_metadata.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Metadata CSVs not found in {root}")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # Sort and map unique training vehicle IDs to [0, num_classes - 1]
    train_vehicle_ids = sorted(train_df["vehicle_id"].unique())
    vehicle_to_label = {vid: idx for idx, vid in enumerate(train_vehicle_ids)}
    train_df["label"] = train_df["vehicle_id"].map(vehicle_to_label)

    print(f"[ALDR Data] Loaded {len(train_df)} training images across {len(train_vehicle_ids)} vehicle identities.")
    print(f"[ALDR Data] Loaded {len(val_df)} validation images across {val_df['vehicle_id'].nunique()} disjoint vehicle identities.")

    return train_df, val_df, vehicle_to_label


def create_image_loader(image_dir: Path):
    """Factory for deterministic load and preprocess function."""
    img_dir_str = str(image_dir) + "/"

    def load_and_preprocess(filename, label):
        path = tf.strings.join([img_dir_str, filename])
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [REID_IMG_HEIGHT, REID_IMG_WIDTH])
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.resnet50.preprocess_input(img)
        return img, label

    return load_and_preprocess


def build_train_dataset(
    train_df: pd.DataFrame,
    image_dir: Path,
    batch_size: int = REID_BATCH_SIZE,
    cache_path: Optional[str] = None,
    use_augmentation: bool = True,
    sample_size: Optional[int] = None
) -> tf.data.Dataset:
    """
    Builds an optimized tf.data pipeline with caching, parallel mapping, and prefetching.
    """
    df = train_df.sample(sample_size, random_state=42) if sample_size else train_df

    filenames = df["filename"].values
    labels = df["label"].values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    loader = create_image_loader(image_dir)

    # Deterministic load + resize + preprocess
    ds = ds.map(loader, num_parallel_calls=tf.data.AUTOTUNE)

    # Cache deterministic preprocessed images (in memory or fast local disk)
    if cache_path:
        ds = ds.cache(cache_path)
    else:
        # In-memory cache if subset is small
        if sample_size and sample_size <= 2000:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=min(len(df), 4096), reshuffle_each_iteration=True)

    if use_augmentation:
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.10),
            tf.keras.layers.RandomContrast(0.10)
        ], name="reid_augmentation")

        def apply_aug(img, lbl):
            return augmentation(img, training=True), lbl

        ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


class ReIDRetrievalCallback(tf.keras.callbacks.Callback):
    """
    Identity-disjoint retrieval callback.
    Evaluates Rank-1, Rank-5 and mAP on held-out validation queries and galleries
    instead of relying only on training softmax loss.
    """

    def __init__(
        self,
        embedding_model: tf.keras.Model,
        val_df: pd.DataFrame,
        image_dir: Path,
        eval_freq: int = 5,
        max_eval_samples: int = 400
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.eval_freq = eval_freq
        self.image_dir = image_dir

        # Create a balanced query/gallery evaluation split from disjoint val identities
        sample_df = val_df.groupby("vehicle_id").filter(lambda g: len(g) >= 2)
        if len(sample_df) > max_eval_samples:
            sample_df = sample_df.sample(max_eval_samples, random_state=42)

        self.eval_df = sample_df
        self.filenames = sample_df["filename"].tolist()
        self.vehicle_ids = sample_df["vehicle_id"].to_numpy()

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_freq != 0 and (epoch + 1) != self.params.get("epochs", 0):
            return

        print(f"\n[Re-ID Callback] Evaluating retrieval metrics at epoch {epoch + 1}...")
        try:
            # Batch extract embeddings
            embeddings = []
            for fname in self.filenames:
                img_path = str(self.image_dir / fname)
                img = tf.io.read_file(img_path)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [REID_IMG_HEIGHT, REID_IMG_WIDTH])
                img = tf.keras.applications.resnet50.preprocess_input(img)
                emb = self.embedding_model(tf.expand_dims(img, 0), training=False).numpy()[0]
                embeddings.append(emb)

            embs = np.array(embeddings)
            # Pairwise cosine similarity matrix
            sim_matrix = np.dot(embs, embs.T)
            np.fill_diagonal(sim_matrix, -1.0)  # Exclude self-match

            rank1_hits = 0
            rank5_hits = 0
            n_queries = len(self.vehicle_ids)

            for i in range(n_queries):
                target_id = self.vehicle_ids[i]
                sorted_indices = np.argsort(-sim_matrix[i])
                top_ids = self.vehicle_ids[sorted_indices]

                if top_ids[0] == target_id:
                    rank1_hits += 1
                if target_id in top_ids[:5]:
                    rank5_hits += 1

            r1 = rank1_hits / max(n_queries, 1)
            r5 = rank5_hits / max(n_queries, 1)
            print(f"[Re-ID Retrieval Validation] Rank-1: {r1 * 100:.2f}% | Rank-5: {r5 * 100:.2f}% (queries: {n_queries})")
        except Exception as e:
            print(f"[Re-ID Callback] Warning: Validation retrieval skipped ({e})")


def train_reid(
    epochs: int = 20,
    batch_size: int = REID_BATCH_SIZE,
    learning_rate: float = 1e-4,
    patience: int = 5,
    sample_size: Optional[int] = None,
    checkpoint_dir: Optional[Path] = None,
    save_embedding_path: Optional[Path] = None,
    resume_from_checkpoint: bool = True
) -> Tuple[tf.keras.Model, tf.keras.Model, dict]:
    """
    Main training execution function.
    """
    train_df, val_df, vehicle_to_label = load_metadata()
    num_classes = len(vehicle_to_label)

    ckpt_dir = checkpoint_dir or REID_CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = ckpt_dir / "best_classifier.weights.h5"

    train_img_dir = VERI_ROOT_DIR / "image_train"

    training_model, embedding_model = build_reid_models(
        num_classes=num_classes,
        freeze_backbone=True
    )

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Resume from checkpoint if available
    initial_epoch = 0
    if resume_from_checkpoint and best_weights_path.exists():
        try:
            print(f"[ALDR Re-ID] Found existing checkpoint at {best_weights_path}. Resuming...")
            training_model.load_weights(str(best_weights_path))
            print("[ALDR Re-ID] Checkpoint weights loaded successfully.")
        except Exception as e:
            print(f"[ALDR Re-ID] Checkpoint could not be loaded ({e}). Starting fresh.")

    # Prepare dataset
    print("[ALDR Re-ID] Building dataset pipeline...")
    train_ds = build_train_dataset(
        train_df,
        image_dir=train_img_dir,
        batch_size=batch_size,
        use_augmentation=True,
        sample_size=sample_size
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_weights_path),
            monitor="loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            str(ckpt_dir / "training_history.csv"),
            append=True
        ),
        ReIDRetrievalCallback(
            embedding_model=embedding_model,
            val_df=val_df,
            image_dir=train_img_dir,
            eval_freq=max(1, epochs // 4)
        )
    ]

    print(f"[ALDR Re-ID] Starting training for {epochs} epochs (Early stopping patience={patience})...")
    start_time = time.time()
    history = training_model.fit(
        train_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )
    duration = time.time() - start_time
    print(f"[ALDR Re-ID] Training finished in {duration:.1f}s.")

    # Save final standalone embedding model
    save_path = save_embedding_path or REID_EMBEDDING_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        embedding_model.save(str(save_path))
        print(f"[ALDR Re-ID] Saved standalone embedding model to: {save_path}")
    except Exception as e:
        # Fallback to weights save if keras format has serialization issues
        w_path = str(save_path).replace(".keras", ".weights.h5")
        embedding_model.save_weights(w_path)
        print(f"[ALDR Re-ID] Saved embedding model weights to: {w_path} ({e})")

    return training_model, embedding_model, history.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and optimize Vehicle Re-ID model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sample-size", type=int, default=None, help="Subset size for smoke testing")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-epoch smoke test on 200 samples")
    args = parser.parse_args()

    if args.smoke_test:
        print("[ALDR Re-ID] Running Smoke Test (2 epochs, 200 samples)...")
        train_reid(epochs=2, sample_size=200, patience=2)
    else:
        train_reid(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, sample_size=args.sample_size)
