package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_RemoveFromCart_17_3_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testRemoveFromCart() {
        // Given
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(cart.RemoveFromCart(hmac, cartId, itemId)).thenReturn(expectedShoppingCart);
        // When
        ShoppingCart actualShoppingCart = a4j.RemoveFromCart(hmac, cartId, itemId);
        // Then
        assertEquals(expectedShoppingCart, actualShoppingCart);
    }
}
