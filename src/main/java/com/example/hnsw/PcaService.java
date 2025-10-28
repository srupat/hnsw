package com.example.hnsw;

import java.util.List;

public interface PcaService {
    double[][] reduce(List<double[]> points, int dim, int targetDim);
}
