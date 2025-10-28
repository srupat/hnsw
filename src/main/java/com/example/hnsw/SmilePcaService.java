package com.example.hnsw;

import smile.projection.PCA;
import java.util.*;

public class SmilePcaService implements PcaService {
    @Override
    public double[][] reduce(List<double[]> points, int dim, int targetDim) {
        if (points == null || points.isEmpty()) throw new IllegalArgumentException("points empty");
        if (dim <= 0 || targetDim <= 0 || targetDim > dim) throw new IllegalArgumentException("bad dims");
        double[][] X = toArray(points, dim);

        PCA pca = PCA.fit(X);          // covariance-based, mean-centered
        pca.setProjection(targetDim);  // choose number of components
        return pca.project(X);         // reduced representation
    }

    private static double[][] toArray(List<double[]> pts, int dim) {
        double[][] X = new double[pts.size()][dim];
        for (int i = 0; i < pts.size(); i++) {
            double[] row = pts.get(i);
            if (row.length != dim) throw new IllegalArgumentException("row " + i + " has dim=" + row.length);
            X[i] = Arrays.copyOf(row, dim);
        }
        return X;
    }
}
