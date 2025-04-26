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

class ShoppingCart_toString_8_0_Test {

    @Mock
    private ShoppingCart shoppingCart;

    @Test
    public void testToString() {
        // Arrange
        ShoppingCart shoppingCart = mock(ShoppingCart.class);
        when(shoppingCart.getHMAC()).thenReturn("abc123");
        when(shoppingCart.getPurchaseURL()).thenReturn("https://example.com/purchase");
        when(shoppingCart.getCartId()).thenReturn("12345");
        // Act
        String result = shoppingCart.toString();
        // Assert
        assertEquals("HMAC = abc123\n" + "Purchase URL = https://example.com/purchase\n" + "CartId = 12345\n" + "items = [Item{ourPrice=10.99, quantity=2}]", result);
    }
}
