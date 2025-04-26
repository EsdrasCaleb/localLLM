package net.kencochrane.a4j.beans;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
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

    @Mock
    private ShoppingCart shoppingCart;

    @InjectMocks
    private ShoppingCart shoppingCartUnderTest;

    @Test
    public void testToString() {
        // Arrange
        shoppingCartUnderTest.setCartId("cartId");
        shoppingCartUnderTest.setHMAC("HMAC");
        shoppingCartUnderTest.setPurchaseURL("purchaseURL");
        shoppingCartUnderTest.setItems(new Items());
        // Act
        String result = shoppingCartUnderTest.toString();
        // Assert
        assertEquals("HMAC = cartId\nPurchase URL = purchaseURL\nCartId = cartId\nitems = items\n", result);
    }
}
