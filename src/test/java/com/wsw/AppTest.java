package com.wsw;


import com.wsw.knn.ExampleKNN;
import com.wsw.knn.core.KNN;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * @author Wangshuwei
 * @since 2017-5-7
 */
public class AppTest {

    /**
     * Test for K-近邻算法的classify0()
     */
    @Test
    public void testKNN() {
        INDArray inXs = Nd4j.create(new float[]{20000, 200});

        INDArray dataSet = Nd4j.create(new float[]{3, 104, 2, 100, 1, 81, 101, 10, 99, 5, 98, 2}, new int[]{6, 2});
        String[] labels = {"love", "love", "love", "action", "action", "action"};
        System.out.println(KNN.classify0(inXs, dataSet, labels, 4));
    }

    @Test
    public void testFileToMatrix() {
        Map<String, Object> map = ExampleKNN.fileToMatrix("/data/datingTestSet2.txt");
        System.out.println(map.toString());
    }

    @Test
    public void testAutoNorm() {
        ExampleKNN.aotuNorm((INDArray) ExampleKNN.fileToMatrix("/data/datingTestSet2.txt").get("dataSet"));
    }

    @Test
    public void testDatingClassifier() {
        ExampleKNN.datingClassesTest();
    }

}
