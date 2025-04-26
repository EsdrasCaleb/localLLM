package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Util_StringIsEmpty_0_2_Test.class, Util_NormalizeString_1_1_Test.class, Util_VectorEqualsUnordered_4_0_Test.class, Util_IntMaxString_5_0_Test.class, Util_DoubleMaxString_6_0_Test.class })
public class Util_Suite {
}
