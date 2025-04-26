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
    void testToString() {
        ShoppingCart shoppingCart = new ShoppingCart();
        shoppingCart.setCartId("12345");
        shoppingCart.setHMAC("abc123");
        shoppingCart.setPurchaseURL("https://example.com/purchase");
        shoppingCart.setItems(new Items());
        String expectedString = "HMAC = abc123\nPurchase URL = https://example.com/purchase\nCartId = 12345\nitems = null";
        String actualString = shoppingCart.toString();
        assertEquals(expectedString, actualString);
    }
}
