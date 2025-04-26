package org.templateit;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { DynamicTemplate_getRowHeight_3_2_Test.class, DynamicTemplate_getCell_4_1_Test.class })
public class DynamicTemplate_Suite {
}
