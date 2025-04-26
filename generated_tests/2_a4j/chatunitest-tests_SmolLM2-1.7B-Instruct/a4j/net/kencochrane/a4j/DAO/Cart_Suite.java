package net.kencochrane.a4j.DAO;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Cart_AddtoCart_0_1_Test.class, Cart_addToExistingCart_1_0_Test.class, Cart_clearCart_2_2_Test.class })
public class Cart_Suite {
}
