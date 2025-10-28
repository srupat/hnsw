package com.example.hnsw;

import java.util.*;

public class PCA {
    public static void main(String[] args) {
        List<double[]> points = Arrays.asList(
                new double[]{1.0, 2.0}, new double[]{1.5, 2.2}, new double[]{0.8, 1.8},
                new double[]{5.0, 8.0}, new double[]{5.5, 8.2}, new double[]{6.0, 7.5},
                new double[]{9.0, 0.5}, new double[]{8.5, 1.0}, new double[]{9.5, 1.2}
        );

        PcaService pcaSvc = new SmilePcaService();
        double[][] Z = pcaSvc.reduce(points, 2, 1);

        System.out.println(java.util.Arrays.deepToString(Z));

    }
}
