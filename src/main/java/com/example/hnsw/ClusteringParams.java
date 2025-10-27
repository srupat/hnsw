/** Optional knobs per algorithm. Provide what you need. */
package com.example.hnsw;

import smile.math.distance.Distance;
import smile.math.distance.EuclideanDistance;

public class ClusteringParams {
    // Shared
    public Distance<double[]> distance = new EuclideanDistance();

    // KMeans/KMeans++
    public Integer k;                 // required for KMeans / KMeans++
    public Integer maxIters = 300;
    public Long randomSeed = 42L;

    // HDBSCAN
    public Integer minPts = 5;        // typical default: 5–15

    // KDTree+Voronoi
    // If you provide 'sites', we'll build Voronoi cells around these.
    // Otherwise we’ll pick 'k' sites via KMeans++ seeding.
    public double[][] voronoiSites = null;  // shape: [k][dim]
}
