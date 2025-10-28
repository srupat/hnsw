// ClusteringParams.java
package com.example.hnsw;

class ClusteringParams {
    // KMeans/KMeans++
    public Integer k;
    public Integer maxIters = 300;
    public Long randomSeed = 42L;

    // DBSCAN (Smile)
    public Double dbscanEps = 1.0;   // set this based on your scale
    public Integer dbscanMinPts = 5;

    // KDTree + Voronoi
    public double[][] voronoiSites = null;
}
