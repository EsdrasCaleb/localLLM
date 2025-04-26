package br.com.jnfe.base.util;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { SecurityUtils_openStore_0_3_Test.class, SecurityUtils_Suite.class })
public class SecurityUtils_Suite {
}
