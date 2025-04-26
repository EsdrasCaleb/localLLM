package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;

public class ShoppingCart_toString_8_0_Test {

    @Test
    public void testToString() {
        ShoppingCart cart = new ShoppingCart();
        cart.setCartId("testCartId");
        cart.setHMAC("testHMAC");
        cart.setPurchaseURL("testPurchaseURL");
        // Assume we have an Items object with some items
        cart.setItems(new Items());
        // Call the toString method
        String result = cart.toString();
        // Assert that the result is as expected
        assertEquals("HMAC = testHMAC\nPurchase URL = testPurchaseURL\nCartId = testCartId\nitems = null\n", result);
    }
}
