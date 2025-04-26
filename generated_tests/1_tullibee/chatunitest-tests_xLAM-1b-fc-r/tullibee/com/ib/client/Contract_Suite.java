package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Contract_equals_1_0_Test.class, Contract_Suite.class, Contract_equals_1_1_Test.class })
public class Contract_Suite {
}
