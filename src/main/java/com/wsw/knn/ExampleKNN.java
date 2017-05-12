package com.wsw.knn;

import com.google.common.io.Files;
import com.wsw.knn.core.KNN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.*;

/**
 * 《机器学习实战》书中使用K-近邻算法改进约会网站的配对实例java的实现
 *
 * @author Wangshuwei
 * @since 2017-5-8
 */
public class ExampleKNN {

    private static final Logger LOGGER = LoggerFactory.getLogger(ExampleKNN.class);

    /**
     * 讲文件中的数据集合转换成样本数据集合和样本数据标签
     *
     * @param fileName 数据集合文件
     * @return 包含样本数据集合和样本数据标签的map
     */
    public static Map<String, Object> fileToMatrix(String fileName) {
        // 从类路径里获得测试文件
        File file = new File(ExampleKNN.class.getClass().getResource(fileName).getFile());
        List<String> lines = new ArrayList<>();
        try {
            lines = Files.readLines(file, Charset.defaultCharset());  //读取所有的数据
        } catch (IOException e) {
            LOGGER.error(e.getMessage());
        }

        double[] dataSet = new double[3 * lines.size()]; //创建样本数据数组
        int datasetIndex = 0;

        String[] lebels = new String[lines.size()]; //创建样本数据标签数组
        int lebelIndex = 0;

        //将样本数据和样本数据标签分开
        for (String line : lines) {
            String[] strings = line.split("\\t");

            String[] dataSetStr = Arrays.copyOf(strings, 3);
            lebels[lebelIndex] = Arrays.copyOfRange(strings, 3, 4)[0];
            lebelIndex++;
            //LOGGER.info(Arrays.toString(dataSetStr));

            for (String aDataSetStr : dataSetStr) {
                dataSet[datasetIndex] = Double.parseDouble(aDataSetStr);
                datasetIndex++;
            }

        }

        INDArray dataMat = Nd4j.create(dataSet, new int[]{1000, 3});
        Map<String, Object> map = new HashMap<>();
        map.put("dataSet", dataMat);
        map.put("label", lebels);
        return map;

    }

    /**
     * 归一化特征值
     * newValue = (oldValue-min)/ (max-min)
     *
     * @param dataSet 样本数据集合
     * @return 归一化后的样本数据集合
     */
    public static INDArray aotuNorm(INDArray dataSet) {

        //计算最小值和最大值
        INDArray minVals = dataSet.min(0);
        INDArray maxVals = dataSet.max(0);

        //归一化特征值
        int rows = dataSet.rows();
        INDArray ranges = maxVals.sub(minVals);
        INDArray normDataSet = dataSet.sub(Nd4j.tile(minVals, rows));
        normDataSet = normDataSet.div(Nd4j.tile(ranges, rows));

        /*LOGGER.info("dataSizeRows: " + dataSet.rows());
        LOGGER.info("minVals: " + minVals);
        LOGGER.info("maxVals: " + maxVals);
        LOGGER.info("ranges: " + ranges);
        LOGGER.info("normDataSet: " + normDataSet);*/

        return normDataSet;
    }

    /**
     * 测试分类器
     */
    public static void datingClassesTest() {
        //设定测试数据的比率
        double hoRatios = 0.5;

        //获取数据和数据标签
        Map<String, Object> map = fileToMatrix("/data/datingTestSet2.txt");
        INDArray dataSet = (INDArray) map.get("dataSet");
        String[] labels = (String[]) map.get("label");
        //归一化特征值
        INDArray normDataSet = aotuNorm(dataSet);

        int size = dataSet.rows();  //数据大小
        int numOfTest = (int) (size * hoRatios);    //测试数据集合的大小
        int errorCount = 0; //错误个数
        //从数据中选取后50%的数据作为样本数据集合
        INDArray datingDataSet = normDataSet.get(NDArrayIndex.interval(numOfTest, size), NDArrayIndex.all());
        String[] datingLabels = Arrays.copyOfRange(labels, numOfTest, size);
        //System.out.println(datingDataSet + " : " + Arrays.toString(datingLabels));

        //开始测试
        for (int i = 0; i < numOfTest; i++) {
            String testLabel = KNN.classify0(normDataSet.get(NDArrayIndex.point(i), NDArrayIndex.all()),
                    datingDataSet, datingLabels, 3);

            if (!testLabel.equals(labels[i])) {
                LOGGER.error("the classfier came back with {},the real answer is {}", testLabel, labels[i]);
                errorCount++;
            } else {
                LOGGER.info("the classfier came back with {},the real answer is {}", testLabel, labels[i]);
            }
        }

        LOGGER.info("the total error size is : {}", (errorCount / (double) numOfTest));

    }
}
