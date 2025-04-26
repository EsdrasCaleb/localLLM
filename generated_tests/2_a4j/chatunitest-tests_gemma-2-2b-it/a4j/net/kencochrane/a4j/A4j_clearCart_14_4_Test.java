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
class A4j_clearCart_14_4_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @Test
    void clearCart() {
        String hmac = "hmac";
        String cartId = "cartId";
        when(cart.clearCart(hmac, cartId)).thenReturn(new ShoppingCart());
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertEquals(result, new ShoppingCart());
    }
}
