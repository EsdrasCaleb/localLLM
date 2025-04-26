package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

public class A4j_GetItemsFromCart_16_0_Test {

    @Test
    void testGetItemsFromCart() {
        // Arrange
        A4j a4j = mock(A4j.class);
        when(a4j.GetItemsFromCart("hmac", "cartId")).thenReturn(new ShoppingCart());
        // Act
        ShoppingCart cart = a4j.GetItemsFromCart("hmac", "cartId");
        // Assert
        verify(a4j, times(1)).GetItemsFromCart("hmac", "cartId");
        // Assert that the returned object is not null
        assertNotNull(cart);
    }
}
