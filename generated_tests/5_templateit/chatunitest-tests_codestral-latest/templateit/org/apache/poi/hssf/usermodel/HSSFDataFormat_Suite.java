package org.apache.poi.hssf.usermodel;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { HSSFDataFormat_getBuiltinFormat_2_0_Test.class, HSSFDataFormat_getFormat_3_4_Test.class, HSSFDataFormat_getFormat_4_0_Test.class, HSSFDataFormat_getBuiltinFormat_5_1_Test.class })
public class HSSFDataFormat_Suite {
}
