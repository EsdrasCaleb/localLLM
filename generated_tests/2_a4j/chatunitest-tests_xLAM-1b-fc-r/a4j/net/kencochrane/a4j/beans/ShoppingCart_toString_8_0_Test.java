package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
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
    public void testToString() throws Exception {
        ShoppingCart cart = new ShoppingCart();
        cart.setCartId("12345");
        cart.setHMAC("hmacValue");
        cart.setPurchaseURL("purchaseURLValue");
        Field field = ShoppingCart.class.getDeclaredField("items");
        field.setAccessible(true);
        cart.setItems(new Items());
        String expected = "HMAC = hmacValue\nPurchase URL = purchaseURLValue\nCartId = 12345\nitems = ShoppingCart.Items@12345";
        String actual = cart.toString();
        assertEquals(expected, actual);
    }
}
