package com.example.hnsw;

import smile.clustering.HDBSCAN;
import smile.clustering.KMeans;
import smile.neighbor.KDTree;
import smile.neighbor.Neighbor;

import java.util.*;

class SmileClusteringService implements ClusteringService {

    @Override
    public ClusterResult cluster(List<double[]> points, int dim,
                                 ClusteringAlgorithm algo, ClusteringParams params) {
        Objects.requireNonNull(points, "points");
        if (points.isEmpty()) throw new IllegalArgumentException("points is empty");
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        double[][] X = toArray(points, dim);

        switch (algo) {
            case KMEANS:            return runKMeans(X, params, false);
            case KMEANS_PLUS_PLUS:  return runKMeans(X, params, true);
            case HDBSCAN:           return runHDBSCAN(X, params);
            case KDTREE_VORONOI:    return runKDTreeVoronoi(X, dim, params);
            default:
                throw new UnsupportedOperationException("Unknown algorithm: " + algo);
        }
    }

    // ---------- KMEANS / KMEANS++ (Euclidean) ----------
    private ClusterResult runKMeans(double[][] X, ClusteringParams p, boolean plusPlus) {
        if (p.k == null || p.k <= 0) throw new IllegalArgumentException("k must be provided > 0");

        KMeans.Initialization init = plusPlus
                ? KMeans.Initialization.KMEANS_PLUS_PLUS
                : KMeans.Initialization.RANDOM;

        Random rng = new Random(p.randomSeed == null ? 42L : p.randomSeed);
        KMeans model = KMeans.fit(X, p.k, p.maxIters, init, rng);

        int[] labels = model.y;
        double[][] centroids = model.centroids();

        Map<String, Object> meta = new HashMap<>();
        meta.put("algo", plusPlus ? "kmeans++" : "kmeans");
        meta.put("inertia", model.error());
        meta.put("iterations", model.iterations());
        return new ClusterResult(labels, centroids, meta);
    }

    // ---------- HDBSCAN (Euclidean) ----------
    private ClusterResult runHDBSCAN(double[][] X, ClusteringParams p) {
        int minPts = (p.minPts == null) ? 5 : p.minPts;
        HDBSCAN h = HDBSCAN.fit(X, minPts);
        int[] labels = h.y;
        Map<String, Object> meta = new HashMap<>();
        meta.put("algo", "hdbscan");
        meta.put("numClusters", Arrays.stream(labels).filter(c -> c >= 0).distinct().count());
        meta.put("hasNoise", Arrays.stream(labels).anyMatch(c -> c < 0));
        return new ClusterResult(labels, null, meta);
    }

    // ---------- KD-TREE + VORONOI (Euclidean) ----------
    private ClusterResult runKDTreeVoronoi(double[][] X, int dim, ClusteringParams p) {
        double[][] sites;
        if (p.voronoiSites != null) {
            sites = validateSites(p.voronoiSites, dim);
        } else {
            if (p.k == null || p.k <= 0)
                throw new IllegalArgumentException("Provide params.k or params.voronoiSites");
            sites = kmeansPlusPlusSeeds(X, p.k, p.randomSeed == null ? 42L : p.randomSeed);
        }

        Integer[] siteLabels = new Integer[sites.length];
        for (int i = 0; i < sites.length; i++) siteLabels[i] = i;

        KDTree<Integer> tree = new KDTree<>(sites, siteLabels);

        int n = X.length;
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            Neighbor<double[], Integer> nn = tree.nearest(X[i]);
            labels[i] = nn.value;
        }

        Map<String, Object> meta = new HashMap<>();
        meta.put("algo", "kdtree_voronoi");
        meta.put("numSites", sites.length);
        return new ClusterResult(labels, sites, meta);
    }

    // ---------- Utilities ----------
    private static double[][] toArray(List<double[]> points, int dim) {
        double[][] X = new double[points.size()][dim];
        for (int i = 0; i < points.size(); i++) {
            double[] p = points.get(i);
            if (p.length != dim)
                throw new IllegalArgumentException("Point " + i + " has dim=" + p.length + " but expected " + dim);
            X[i] = Arrays.copyOf(p, dim);
        }
        return X;
    }

    private static double[][] validateSites(double[][] sites, int dim) {
        if (sites.length == 0) throw new IllegalArgumentException("voronoiSites is empty");
        for (int i = 0; i < sites.length; i++) {
            if (sites[i] == null || sites[i].length != dim)
                throw new IllegalArgumentException("voronoiSites[" + i + "] has wrong dimensionality");
        }
        return sites;
    }

    /** Simple KMeans++ seeding (Euclidean). */
    private static double[][] kmeansPlusPlusSeeds(double[][] X, int k, long seed) {
        Random rng = new Random(seed);
        int n = X.length, d = X[0].length;

        List<double[]> centers = new ArrayList<>(k);
        centers.add(Arrays.copyOf(X[rng.nextInt(n)], d));

        double[] d2 = new double[n];
        for (int i = 0; i < n; i++) d2[i] = sqDist(X[i], centers.get(0));

        while (centers.size() < k) {
            double sum = 0.0;
            for (double v : d2) sum += v;
            double r = rng.nextDouble() * sum;
            double cum = 0.0;
            int next = 0;
            for (; next < n - 1; next++) { cum += d2[next]; if (cum >= r) break; }
            centers.add(Arrays.copyOf(X[next], d));
            for (int i = 0; i < n; i++) {
                double nd = sqDist(X[i], centers.get(centers.size() - 1));
                if (nd < d2[i]) d2[i] = nd;
            }
        }

        double[][] out = new double[k][d];
        for (int i = 0; i < k; i++) out[i] = Arrays.copyOf(centers.get(i), d);
        return out;
    }

    private static double sqDist(double[] a, double[] b) {
        double s = 0.0;
        for (int i = 0; i < a.length; i++) { double d = a[i] - b[i]; s += d * d; }
        return s;
    }
}
