package com.example.hnsw;

import java.util.List;

public interface ClusteringService {
    ClusterResult cluster(List<double[]> points, int dim,
                          ClusteringAlgorithm algo, ClusteringParams params);
}