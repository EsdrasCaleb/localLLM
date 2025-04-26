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
public class A4j_GetItemsFromCart_16_0_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testGetItemsFromCart() {
        String hmac = "dummyHmac";
        String cartId = "dummyCartId";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.GetItemsFromCart(hmac, cartId)).thenReturn(expectedCart);
        ShoppingCart actualCart = a4j.GetItemsFromCart(hmac, cartId);
        assertEquals(expectedCart, actualCart);
    }
}
