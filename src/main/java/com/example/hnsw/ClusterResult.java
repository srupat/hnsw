package com.example.hnsw;

import java.util.Collections;
import java.util.Map;

/** Result of clustering / partitioning. */
public class ClusterResult {
    /** Label per input point (length = n). Noise is usually -1 for HDBSCAN. */
    public final int[] labels;

    /** Representative centers (KMeans centroids or Voronoi sites). May be null. */
    public final double[][] representatives;

    /** Optional extra info (e.g., algorithm name, params). */
    public final Map<String, Object> meta;

    ClusterResult(int[] labels, double[][] reps, Map<String, Object> meta) {
        this.labels = labels;
        this.representatives = reps;
        this.meta = meta == null ? Collections.emptyMap() : meta;
    }
}