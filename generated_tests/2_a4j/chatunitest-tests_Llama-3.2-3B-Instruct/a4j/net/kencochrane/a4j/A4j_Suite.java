package net.kencochrane.a4j;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { A4j_getFullProductFromASIN_0_2_Test.class, A4j_BlendedSearch_1_0_Test.class, A4j_Suite.class })
public class A4j_Suite {
}
