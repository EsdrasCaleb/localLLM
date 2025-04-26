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

public class A4j_GetItemsFromCart_16_2_Test {

    @Mock
    private Cart mockCart;

    @Test
    public void testGetItemsFromCart() {
        MockitoAnnotations.initMocks(this);
        String hmac = "hmac";
        String cartId = "cartId";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(mockCart.GetItemsFromCart(hmac, cartId)).thenReturn(expectedShoppingCart);
        A4j a4j = new A4j();
        ShoppingCart actualShoppingCart = a4j.GetItemsFromCart(hmac, cartId);
        assertEquals(expectedShoppingCart, actualShoppingCart);
    }
}
