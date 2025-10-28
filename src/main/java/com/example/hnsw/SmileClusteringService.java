package com.example.hnsw;

//import smile.clustering.HDBSCAN;

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

        return switch (algo) {
            case KMEANS -> runKMeans(X, params, false);
            case KMEANS_PLUS_PLUS -> runKMeans(X, params, true);
//            case HDBSCAN:           return runHDBSCAN(X, params);
            case KDTREE_VORONOI -> runKDTreeVoronoi(X, dim, params);
            default -> throw new UnsupportedOperationException("Unknown algorithm: " + algo);
        };
    }


    // --- Replace your runKMeans with this ---
    private ClusterResult runKMeans(double[][] X, ClusteringParams p, boolean plusPlus) {
        if (p == null) throw new IllegalArgumentException("params cannot be null");
        if (p.k == null || p.k <= 0) throw new IllegalArgumentException("k must be > 0");
        final int k = p.k;
        final int maxIters = (p.maxIters == null ? 100 : p.maxIters);
        final long seed = (p.randomSeed == null ? 42L : p.randomSeed);

        // Choose init: K-Means++ or plain random
        double[][] init = plusPlus
                ? kmeansPlusPlusSeeds(X, k, seed)
                : randomSeeds(X, k, seed);

        // Lloyd iterations with the chosen init
        LloydResult lr = lloyd(X, init, maxIters);

        Map<String, Object> meta = new HashMap<>();
        meta.put("algo", plusPlus ? "kmeans++" : "kmeans");
        meta.put("k", k);
        meta.put("maxIters", maxIters);
        meta.put("itersRun", lr.iters);
        meta.put("changedLastIter", lr.changed);
        return new ClusterResult(lr.labels, lr.centroids, meta);
    }

// --- Helpers: random init, lloyd updates, etc. ---

    /** Randomly pick k distinct rows as initial centers. */
    private static double[][] randomSeeds(double[][] X, int k, long seed) {
        Random rng = new Random(seed);
        int n = X.length, d = X[0].length;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        // Fisher–Yates shuffle first k
        for (int i = 0; i < k; i++) {
            int j = i + rng.nextInt(n - i);
            int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }
        double[][] C = new double[k][d];
        for (int c = 0; c < k; c++) C[c] = Arrays.copyOf(X[idx[c]], d);
        return C;
    }

    /** One Lloyd run from given init centers. */
    private static LloydResult lloyd(double[][] X, double[][] C0, int maxIters) {
        int n = X.length, d = X[0].length, k = C0.length;
        double[][] C = new double[k][d];
        for (int c = 0; c < k; c++) C[c] = Arrays.copyOf(C0[c], d);

        int[] labels = new int[n];
        Arrays.fill(labels, -1);

        boolean changed = true;
        int it = 0;
        while (it < maxIters && changed) {
            changed = false;

            // Assign step
            for (int i = 0; i < n; i++) {
                int best = 0; double bestDist = Double.POSITIVE_INFINITY;
                for (int c = 0; c < k; c++) {
                    double s = 0.0;
                    for (int j = 0; j < d; j++) {
                        double diff = X[i][j] - C[c][j];
                        s += diff * diff;
                    }
                    if (s < bestDist) { bestDist = s; best = c; }
                }
                if (labels[i] != best) { labels[i] = best; changed = true; }
            }

            if (!changed) break;

            // Update step
            double[][] nextC = new double[k][d];
            int[] counts = new int[k];
            for (int i = 0; i < n; i++) {
                int c = labels[i];
                counts[c]++;
                for (int j = 0; j < d; j++) nextC[c][j] += X[i][j];
            }
            for (int c = 0; c < k; c++) {
                if (counts[c] == 0) {
                    // Empty cluster: keep old center (or re-seed randomly if you prefer)
                    continue;
                }
                for (int j = 0; j < d; j++) nextC[c][j] /= counts[c];
            }
            C = nextC;
            it++;
        }
        return new LloydResult(C, labels, it, changed);
    }

    private static class LloydResult {
        final double[][] centroids;
        final int[] labels;
        final int iters;
        final boolean changed;
        LloydResult(double[][] C, int[] y, int iters, boolean changed) {
            this.centroids = C; this.labels = y; this.iters = iters; this.changed = changed;
        }
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
