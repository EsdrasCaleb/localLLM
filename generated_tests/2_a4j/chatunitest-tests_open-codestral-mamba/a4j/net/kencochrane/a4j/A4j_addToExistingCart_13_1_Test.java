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
public class A4j_addToExistingCart_13_1_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testAddToExistingCart() {
        String cartId = "123";
        String hmac = "abc";
        String asin = "xyz";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertEquals(expectedCart, result);
    }
}
