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

public class A4j_RemoveFromCart_17_2_Test {

    @Test
    public void testRemoveFromCart() {
        // Arrange
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        A4j a4j = mock(A4j.class);
        ShoppingCart shoppingCart = new ShoppingCart();
        when(a4j.RemoveFromCart(hmac, cartId, itemId)).thenReturn(shoppingCart);
        // Act
        ShoppingCart result = a4j.RemoveFromCart(hmac, cartId, itemId);
        // Assert
        assertEquals(shoppingCart, result);
    }
}
