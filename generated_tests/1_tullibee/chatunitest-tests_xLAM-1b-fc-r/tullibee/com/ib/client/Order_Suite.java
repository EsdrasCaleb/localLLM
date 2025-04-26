package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Order_equals_0_0_Test.class, Order_Suite.class, Order_equals_0_1_Test.class })
public class Order_Suite {
}
