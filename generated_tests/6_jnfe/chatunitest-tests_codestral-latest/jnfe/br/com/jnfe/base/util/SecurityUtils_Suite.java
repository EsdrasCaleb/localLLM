package br.com.jnfe.base.util;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { SecurityUtils_openStore_1_2_Test.class, SecurityUtils_openStore_2_0_Test.class, SecurityUtils_openStore_3_0_Test.class, SecurityUtils_openTrustStore_4_1_Test.class, SecurityUtils_openTrustStore_5_0_Test.class, SecurityUtils_main_8_3_Test.class })
public class SecurityUtils_Suite {
}
