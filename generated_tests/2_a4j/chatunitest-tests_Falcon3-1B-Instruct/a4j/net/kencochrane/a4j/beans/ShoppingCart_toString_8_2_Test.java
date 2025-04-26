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

public class ShoppingCart_toString_8_2_Test {

    @Test
    public void testToString() {
        // Create an instance of ShoppingCart
        ShoppingCart cart = new ShoppingCart();
        // Verify that the 'toString()' method returns the expected string
        String expectedString = "HMAC = null, Purchase URL = null, CartId = null, items = null";
        String actualString = cart.toString();
        // Assertions to verify the correctness
        Assertions.assertEquals(expectedString, actualString);
    }
}
