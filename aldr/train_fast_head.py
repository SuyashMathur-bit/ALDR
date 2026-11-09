"""
ALDR Fast Head Training (80 Epochs)
Caches ResNet50 features in-memory to train the classification head for 80 epochs
and transfers learned weights back into the full end-to-end Keras model.
"""

import os
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model

from aldr.config import (
    REID_IMG_HEIGHT,
    REID_IMG_WIDTH,
    REID_EMBEDDING_DIM,
    VERI_ROOT_DIR,
    ALDR_REID_DIR,
    REID_CHECKPOINT_DIR,
    REID_EMBEDDING_MODEL_PATH
)
from aldr.reid_model import build_reid_models, preprocess_crop
from aldr.evaluation import evaluate_reid_retrieval, ReIDExtractor


def train_80_epochs(samples_per_id: int = 5, epochs: int = 80, lr: float = 1e-3):
    print("=" * 65)
    print(f"   ALDR FAST RE-ID TRAINING: {epochs} EPOCHS (ALL 460 IDENTITIES)   ")
    print("=" * 65)

    train_df = pd.read_csv(ALDR_REID_DIR / "train_metadata.csv")
    val_df = pd.read_csv(ALDR_REID_DIR / "val_metadata.csv")

    train_vids = sorted(train_df["vehicle_id"].unique())
    vid_to_label = {vid: idx for idx, vid in enumerate(train_vids)}
    train_df["label"] = train_df["vehicle_id"].map(vid_to_label)
    num_classes = len(vid_to_label)

    print(f"[Data] Training pool: {len(train_df)} images across {num_classes} identities.")

    # Select a balanced subset covering ALL 460 identities
    selected_rows = []
    for vid, group in train_df.groupby("vehicle_id"):
        selected_rows.append(group.head(samples_per_id))
    subset_df = pd.concat(selected_rows).reset_index(drop=True)

    print(f"[Data] Selected balanced subset: {len(subset_df)} images ({samples_per_id} images per identity x {num_classes} IDs).")

    # 1. Feature extraction with frozen ResNet50
    print("\n[Step 1/3] Extracting 2048-D features using frozen ResNet50 backbone...")
    t0 = time.time()

    base_resnet = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(REID_IMG_HEIGHT, REID_IMG_WIDTH, 3)
    )
    gap = layers.GlobalAveragePooling2D()

    feature_extractor = Model(
        inputs=base_resnet.input,
        outputs=gap(base_resnet.output),
        name="ResNet50_GAP_Extractor"
    )

    features = []
    labels = subset_df["label"].values.astype(np.int32)
    img_dir = VERI_ROOT_DIR / "image_train"

    import cv2
    batch_imgs = []
    batch_indices = []

    for i, row in subset_df.iterrows():
        fpath = str(img_dir / row["filename"])
        img = cv2.imread(fpath)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prep = preprocess_crop(img_rgb)
        batch_imgs.append(prep)

        if len(batch_imgs) == 64 or i == len(subset_df) - 1:
            batch_arr = np.array(batch_imgs, dtype=np.float32)
            feat_batch = feature_extractor(batch_arr, training=False).numpy()
            features.append(feat_batch)
            batch_imgs = []

    X = np.vstack(features)
    y = labels[:len(X)]
    dt_extract = time.time() - t0
    print(f"[Step 1/3 Complete] Extracted {len(X)} feature vectors in {dt_extract:.1f}s. Feature matrix shape: {X.shape}")

    # 2. Build and train head for 80 epochs
    print(f"\n[Step 2/3] Training 512-D Embedding Projection & Softmax Head for {epochs} Epochs...")

    feat_input = layers.Input(shape=(2048,), name="gap_features")
    dense_emb = layers.Dense(REID_EMBEDDING_DIM, activation=None, name="embedding")
    bn_emb = layers.BatchNormalization(name="embedding_batch_norm")
    norm_emb = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="normalized_embedding")
    dense_cls = layers.Dense(num_classes, activation="softmax", name="vehicle_classifier")

    h = dense_emb(feat_input)
    h = bn_emb(h)
    z = norm_emb(h)
    out = dense_cls(z)

    head_model = Model(inputs=feat_input, outputs=out, name="Fast_Head_Classifier")
    head_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    t_train_0 = time.time()
    history = head_model.fit(
        X, y,
        epochs=epochs,
        batch_size=64,
        shuffle=True,
        verbose=1
    )
    dt_train = time.time() - t_train_0

    final_acc = history.history["accuracy"][-1]
    final_loss = history.history["loss"][-1]
    print("\n" + "=" * 65)
    print(f"   TRAINING COMPLETE: {epochs} EPOCHS FINISHED IN {dt_train:.1f} SECONDS!   ")
    print(f"   Final Training Accuracy : {final_acc * 100:.2f}% (Loss: {final_loss:.4f})")
    print(f"   Accuracy at Epoch 10   : {history.history['accuracy'][9] * 100:.2f}%")
    print(f"   Accuracy at Epoch 25   : {history.history['accuracy'][24] * 100:.2f}%")
    print(f"   Accuracy at Epoch 50   : {history.history['accuracy'][49] * 100:.2f}%")
    print(f"   Accuracy at Epoch 80   : {final_acc * 100:.2f}%")
    print("=" * 65)

    # 3. Transfer weights back to full end-to-end model
    print("\n[Step 3/3] Assembling full end-to-end model and saving checkpoints...")
    full_train_model, full_emb_model = build_reid_models(num_classes=num_classes, freeze_backbone=True)

    # Transfer trained layer weights
    full_train_model.get_layer("embedding").set_weights(dense_emb.get_weights())
    full_train_model.get_layer("embedding_batch_norm").set_weights(bn_emb.get_weights())
    full_train_model.get_layer("vehicle_classifier").set_weights(dense_cls.get_weights())

    full_emb_model.get_layer("embedding").set_weights(dense_emb.get_weights())
    full_emb_model.get_layer("embedding_batch_norm").set_weights(bn_emb.get_weights())

    # Save checkpoints to Drive
    REID_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_weights_path = REID_CHECKPOINT_DIR / "best_classifier.weights.h5"
    full_train_model.save_weights(str(ckpt_weights_path))
    print(f"[Checkpoints] Saved updated classifier weights to: {ckpt_weights_path}")

    emb_save_path = ALDR_REID_DIR / "reid_embedding_model.keras"
    full_emb_model.save(str(emb_save_path))
    print(f"[Checkpoints] Saved standalone 512-D embedding model to: {emb_save_path}")

    # Save training log
    log_df = pd.DataFrame({
        "epoch": range(1, epochs + 1),
        "accuracy": history.history["accuracy"],
        "loss": history.history["loss"]
    })
    log_df.to_csv(REID_CHECKPOINT_DIR / "training_history_80ep.csv", index=False)
    print(f"[Checkpoints] Saved 80-epoch training history to: {REID_CHECKPOINT_DIR / 'training_history_80ep.csv'}")

    # 4. Evaluate Re-ID Retrieval
    print("\n[Evaluation] Evaluating newly trained Re-ID model on held-out validation set...")
    extractor = ReIDExtractor(model_path=emb_save_path)
    eval_res = evaluate_reid_retrieval(extractor, max_queries=115, gallery_size=500)

    return history.history, eval_res


if __name__ == "__main__":
    train_80_epochs(samples_per_id=4, epochs=80, lr=1e-3)
