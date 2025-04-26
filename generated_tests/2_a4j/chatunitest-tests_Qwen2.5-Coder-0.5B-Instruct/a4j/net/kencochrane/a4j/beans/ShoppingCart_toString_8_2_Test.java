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

class ShoppingCart_toString_8_2_Test {

    @Test
    public void testToString() {
        ShoppingCart cart = new ShoppingCart();
        cart.setHMAC("abc123");
        cart.setPurchaseURL("https://example.com");
        cart.setCartId("12345");
        String expectedOutput = "HMAC = abc123\nPurchase URL = https://example.com\nCartId = 12345\nitems = ";
        String actualOutput = cart.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
