package com.wsw.logregres.core;

import java.io.File;

/**
 * @author Wangshuwei
 * @since 2017-5-17
 */
public class Logregres {

    public static void loadDataSet(String classPathFileName) {
        File file = new File(Logregres.class.getResource(classPathFileName).getFile());


    }

}
