package com.wsw;


import com.alibaba.fastjson.JSON;
import com.wsw.decisiontree.core.Trees;
import com.wsw.knn.ExampleKNN;
import com.wsw.knn.core.KNN;
import com.wsw.utils.Util;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * @author Wangshuwei
 * @since 2017-5-7
 */
public class AppTest {

    @Test
    public void testNd4j() {
        INDArray array = Nd4j.create(new double[]{1, 2, 3, 1, 2, 3}, new int[]{3, 2});
        INDArray a1 = Nd4j.create(new double[]{0, 0, 0});
        //System.out.println(Nd4j.repeat(array, 1000));
        a1 = Nd4j.tile(a1, 2);
        System.out.println(array.distance2(a1));
    }

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

    /**
     * age(0,1,2:youth,middle,senior)
     * income(0,1,2:low,middle,high)
     * student(0,1:yes,no)
     * credit_rating(0,1:fair,excellent)
     * result(yes,no)
     */
    @Test
    public void testCalcShnnoEnt() {
        List<List<Object>> dataSets = new ArrayList<>();

        dataSets.add(Util.convertList(0, 2, 0, 0, "no"));
        dataSets.add(Util.convertList(0, 2, 0, 1, "no"));
        dataSets.add(Util.convertList(1, 2, 0, 0, "yes"));
        dataSets.add(Util.convertList(2, 1, 0, 0, "yes"));
        dataSets.add(Util.convertList(2, 0, 1, 0, "yes"));
        dataSets.add(Util.convertList(2, 0, 1, 1, "no"));
        dataSets.add(Util.convertList(1, 0, 1, 1, "yes"));
        dataSets.add(Util.convertList(0, 1, 0, 0, "no"));
        dataSets.add(Util.convertList(0, 0, 0, 0, "yes"));
        dataSets.add(Util.convertList(2, 1, 0, 0, "yes"));
        dataSets.add(Util.convertList(0, 1, 1, 1, "yes"));
        dataSets.add(Util.convertList(1, 1, 0, 1, "yes"));
        dataSets.add(Util.convertList(1, 2, 1, 0, "yes"));
        dataSets.add(Util.convertList(2, 1, 0, 1, "no"));

        List<String> labels = new ArrayList<>();
        labels.add("age");
        labels.add("income");
        labels.add("student");
        labels.add("credit_rating");

        Object o=Trees.createTree(dataSets, labels);
        System.out.println(JSON.toJSONString(o));
        //System.out.println(Trees.calcShnnoEnt(dataSets));
        //System.out.println(Trees.splitDataSet(dataSets, 0, 1));
        //System.out.println(dataSets);
        // System.out.println(Trees.chooseBestFeatureToSplit(dataSets));

    }

}
