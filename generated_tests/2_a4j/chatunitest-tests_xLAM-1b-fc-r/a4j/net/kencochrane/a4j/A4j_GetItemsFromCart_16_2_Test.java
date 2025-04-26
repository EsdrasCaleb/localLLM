package net.kencochrane.a4j;

import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
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

@MockitoSettings(strictness = Strictness.LENIENT)
public class A4j_GetItemsFromCart_16_2_Test {

    @InjectMocks
    private A4j a4j;

    @Mock
    private Cart cart;

    @Test
    public void testGetItemsFromCart() {
        String hmac = "hmac";
        String cartId = "cartId";
        when(cart.GetItemsFromCart(hmac, cartId)).thenReturn(new ShoppingCart());
        ShoppingCart result = a4j.GetItemsFromCart(hmac, cartId);
        assertEquals(new ShoppingCart(), result);
        verify(cart, times(1)).GetItemsFromCart(hmac, cartId);
    }
}
