package net.kencochrane.a4j;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { A4j_BlendedSearch_1_1_Test.class, A4j_ActorSearch_3_4_Test.class, A4j_UpcSearch_8_2_Test.class, A4j_ListmaniaSearch_9_0_Test.class, A4j_AddtoCart_12_4_Test.class, A4j_getFullProductFromASIN_0_2_Test.class, A4j_Suite.class })
public class A4j_Suite {
}
