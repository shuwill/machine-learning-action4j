package com.wsw.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author Wangshuwei
 * @since 2017-5-12
 */
public class Util {

    public static List<Object> convertList(Object... objects) {
        return new ArrayList<>(Arrays.asList(objects));
    }
}
