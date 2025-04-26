package com.ib.client;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { OrderState_equals_0_0_Test.class, OrderState_Suite.class, OrderState_equals_0_4_Test.class })
public class OrderState_Suite {
}
