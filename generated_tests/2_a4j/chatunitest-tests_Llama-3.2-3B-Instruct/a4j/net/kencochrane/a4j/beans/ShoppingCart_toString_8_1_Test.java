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

public class ShoppingCart_toString_8_1_Test {

    @Test
    public void testToString() {
        ShoppingCart cart = new ShoppingCart();
        cart.setHMAC("1234567890");
        cart.setPurchaseURL("https://example.com");
        cart.setCartId("ABC123");
        cart.setItems(new Items());
        String expected = "HMAC = 1234567890\n" + "Purchase URL = https://example.com\n" + "CartId = ABC123\n" + "items = Items@...";
        String actual = cart.toString();
        assertEquals(expected, actual);
    }
}
