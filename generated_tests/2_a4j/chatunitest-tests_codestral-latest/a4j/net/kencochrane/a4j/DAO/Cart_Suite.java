package net.kencochrane.a4j.DAO;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Cart_addToExistingCart_1_0_Test.class, Cart_modifyCart_3_4_Test.class, Cart_GetItemsFromCart_4_2_Test.class })
public class Cart_Suite {
}
