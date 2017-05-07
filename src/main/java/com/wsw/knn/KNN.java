package com.wsw.knn;

import com.wsw.utils.Sort;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.*;

/**
 * @author Wangshuwei
 * @since 2017-5-7
 * K-近邻算法的实现
 */
public class KNN {

    /**
     * @param inXs     测试数据集合
     * @param dataSets 样本数据集合
     * @param labels   样本数据标签
     * @param k        k值
     * @return 测试数据集合的类别（标签）
     */
    public static String classify0(INDArray inXs, INDArray dataSets, String[] labels, int k) {
        //获取样本数据集合的个数
        int dataSetSize = dataSets.rows();
        List<Double> distances = new ArrayList<>();

        //讲测试数据集合扩展成跟样本数据集合大小相同的集合
        inXs = Nd4j.repeat(inXs, dataSetSize);

        //计算测试数据集合与每个样本数据集合的距离
        for (int i = 0; i < dataSetSize; i++) {
            INDArray inX = inXs.get(NDArrayIndex.point(i), NDArrayIndex.all());
            INDArray dataSet = dataSets.get(NDArrayIndex.point(i), NDArrayIndex.all());
            double distance = dataSet.distance2(inX);
            distances.add(distance);
        }
        System.out.println(distances.toString());

        //距离排序
        Sort<Double> sort = new Sort<>();
        Double[] data = new Double[distances.size()];
        sort.setArray(distances.toArray(data));
        Integer[] indexs = sort.createIndexArray();
        Arrays.sort(indexs, sort);
        System.out.println(Arrays.toString(indexs));

        //统计距离最近的k个已知数据的类别，以多数投票的形式确定未知数据的类别。
        Map<String, Integer> vote = new HashMap<>();
        for (int i = 0; i < k; i++) {
            int index = indexs[i];
            //list.add(label[index]);
            String label = labels[index];
            Integer count = vote.get(label);
            if (count == null) {
                vote.put(label, 1);
            } else {
                vote.put(label, count + 1);
            }
        }
        System.out.println(vote.toString());
        int count = 0;
        String label = null;
        for (Map.Entry<String, Integer> entry : vote.entrySet()) {
            if (entry.getValue() > count) {
                label = entry.getKey();
                count = entry.getValue();
            }
        }

        return label;
    }

}
