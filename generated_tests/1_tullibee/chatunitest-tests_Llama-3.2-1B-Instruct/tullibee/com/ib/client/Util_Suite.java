package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Util_StringIsEmpty_0_1_Test.class, Util_NormalizeString_1_3_Test.class, Util_StringCompare_2_0_Test.class, Util_StringCompareIgnCase_3_0_Test.class, Util_Suite.class })
public class Util_Suite {
}
