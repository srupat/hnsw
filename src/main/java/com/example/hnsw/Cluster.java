package com.example.hnsw;
import com.example.hnsw.ClusteringParams;
import com.example.hnsw.ClusteringService;
import com.example.hnsw.SmileClusteringService;

import java.util.*;

class Cluster {
    public static void main(String[] args) {
        List<double[]> points = Arrays.asList(
                new double[]{1.0, 2.0}, new double[]{1.5, 2.2}, new double[]{0.8, 1.8},
                new double[]{5.0, 8.0}, new double[]{5.5, 8.2}, new double[]{6.0, 7.5},
                new double[]{9.0, 0.5}, new double[]{8.5, 1.0}, new double[]{9.5, 1.2}
        );
        int dim = 2;

        ClusteringService svc = new SmileClusteringService();

        // KMeans++
        ClusteringParams p1 = new ClusteringParams();
        p1.k = 3;
        ClusterResult r1 = svc.cluster(points, dim, ClusteringAlgorithm.KMEANS_PLUS_PLUS, p1);
        System.out.println("KMeans++ labels: " + Arrays.toString(r1.labels));

        // HDBSCAN
        ClusteringParams p2 = new ClusteringParams();
        p2.minPts = 4;
        ClusterResult r2 = svc.cluster(points, dim, ClusteringAlgorithm.HDBSCAN, p2);
        System.out.println("HDBSCAN labels: " + Arrays.toString(r2.labels));

        // KDTree + Voronoi with explicit sites (optional)
        ClusteringParams p3 = new ClusteringParams();
        p3.voronoiSites = new double[][]{{1.0, 2.0}, {5.0, 8.0}, {9.0, 1.0}};
        ClusterResult r3 = svc.cluster(points, dim, ClusteringAlgorithm.KDTREE_VORONOI, p3);
        System.out.println("KDTree+Voronoi labels: " + Arrays.toString(r3.labels));
    }
}
