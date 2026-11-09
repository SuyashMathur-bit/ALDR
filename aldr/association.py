"""
ALDR Cross-View Vehicle Association Module
Stage 3: Multi-View & Temporal Association
"""

from typing import List, Dict, Any, Optional
import numpy as np
from aldr.config import REID_SIMILARITY_THRESHOLD, TEMPORAL_WINDOW_SECONDS


class VehicleAssociator:
    """
    Associates vehicle observations across camera views and frames
    using Re-ID embeddings, cosine similarity, and spatio-temporal constraints.
    """

    def __init__(
        self,
        similarity_threshold: float = REID_SIMILARITY_THRESHOLD,
        temporal_window: float = TEMPORAL_WINDOW_SECONDS
    ):
        self.similarity_threshold = similarity_threshold
        self.temporal_window = temporal_window

    def associate(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Groups raw vehicle observations into unique vehicle profiles.
        Returns:
            List of vehicle profile dicts.
        """
        n = len(observations)
        if n == 0:
            return []

        # If only 1 observation, trivial profile
        if n == 1:
            return [{
                "vehicle_profile_id": "VEH_PROFILE_001",
                "observations": [observations[0]],
                "num_observations": 1,
                "cameras": [observations[0].get("camera_id", "cam_0")],
                "mean_similarity": 1.0,
                "pairwise_matches": []
            }]

        embeddings = np.array([obs["embedding"] for obs in observations], dtype=np.float32)
        # Pairwise cosine similarity matrix
        sim_matrix = np.matmul(embeddings, embeddings.T)

        # Adjacency matrix for clustering
        adj_matrix = np.zeros((n, n), dtype=bool)
        pairwise_matches = []

        for i in range(n):
            adj_matrix[i, i] = True
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])

                # Spatial / frame constraint: cannot be same vehicle if detected in the exact same image
                same_image = (
                    observations[i].get("source_file") == observations[j].get("source_file")
                    and observations[i].get("source_file") is not None
                )
                if same_image:
                    continue

                # Temporal constraint (if timestamps available)
                t_i = observations[i].get("timestamp")
                t_j = observations[j].get("timestamp")
                if t_i is not None and t_j is not None:
                    if abs(t_i - t_j) > self.temporal_window:
                        continue

                # Camera velocity constraint: cannot be in two different cameras instantaneously
                cam_i = observations[i].get("camera_id")
                cam_j = observations[j].get("camera_id")
                if cam_i != cam_j and t_i is not None and t_j is not None:
                    if abs(t_i - t_j) < 1.0:  # < 1 second apart on different cameras
                        continue

                if sim >= self.similarity_threshold:
                    adj_matrix[i, j] = True
                    adj_matrix[j, i] = True
                    pairwise_matches.append({
                        "obs_a": observations[i].get("detection_id"),
                        "obs_b": observations[j].get("detection_id"),
                        "camera_a": cam_i,
                        "camera_b": cam_j,
                        "similarity": round(sim, 4),
                        "decision": "ASSOCIATED"
                    })

        # Connected component clustering
        visited = [False] * n
        clusters = []

        for i in range(n):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    for neighbor in range(n):
                        if adj_matrix[curr, neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                clusters.append(cluster)

        # Build structured vehicle profiles
        profiles = []
        for p_idx, cluster in enumerate(clusters, 1):
            cluster_obs = [observations[idx] for idx in cluster]
            cluster_cams = list({obs.get("camera_id", "cam_0") for obs in cluster_obs})

            # Calculate internal cluster similarity
            if len(cluster) > 1:
                sub_sims = [sim_matrix[a, b] for a in cluster for b in cluster if a < b]
                mean_sim = float(np.mean(sub_sims)) if sub_sims else 1.0
            else:
                mean_sim = 1.0

            cluster_matches = [
                m for m in pairwise_matches
                if any(m["obs_a"] == obs.get("detection_id") for obs in cluster_obs)
            ]

            profiles.append({
                "vehicle_profile_id": f"VEH_PROFILE_{p_idx:03d}",
                "observations": cluster_obs,
                "num_observations": len(cluster_obs),
                "cameras": cluster_cams,
                "mean_similarity": round(mean_sim, 4),
                "pairwise_matches": cluster_matches
            })

        return profiles
