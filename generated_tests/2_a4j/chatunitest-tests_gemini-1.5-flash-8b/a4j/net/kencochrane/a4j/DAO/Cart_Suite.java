package net.kencochrane.a4j.DAO;

import org.junit.runner.RunWith;
import org.junit.platform.runner.JUnitPlatform;
import org.junit.platform.suite.api.SelectClasses;

@RunWith(value = JUnitPlatform.class)
@SelectClasses(value = { Cart_AddtoCart_0_1_Test.class, Cart_addToExistingCart_1_1_Test.class, Cart_modifyCart_3_2_Test.class, Cart_GetItemsFromCart_4_0_Test.class, Cart_RemoveFromCart_5_0_Test.class })
public class Cart_Suite {
}
