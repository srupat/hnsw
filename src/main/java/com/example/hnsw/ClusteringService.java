package com.example.hnsw;

import java.util.List;


/** Interface you asked for: give points + dim + algorithm (+params) → clustering. */
public interface ClusteringService {
    ClusterResult cluster(List<double[]> points, int dim,
                          ClusteringAlgorithm algo, ClusteringParams params);
}