package com.wsw.utils;

import java.util.Comparator;

/**
 * @author Wangshuwei
 * @since 2017-5-7
 */
public class Sort<T extends Comparable<T>> implements Comparator<Integer> {

    private  T[] array;

    public void setArray(T[] array) {
        this.array = array;
    }

    public Integer[] createIndexArray()
    {
        Integer[] indexes = new Integer[array.length];
        for (int i = 0; i < array.length; i++)
        {
            indexes[i] = i;
        }
        return indexes;
    }

    @Override
    public int compare(Integer index1, Integer index2)
    {
        return array[index1].compareTo(array[index2]);
    }
}
