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
public class A4j_addToExistingCart_13_2_Test {

    @Mock
    private Cart mockCart;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testAddToExistingCart() {
        String cartId = "cartId";
        String hmac = "hmac";
        String asin = "asin";
        String quantity = "quantity";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(mockCart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedShoppingCart);
        ShoppingCart actualShoppingCart = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertEquals(expectedShoppingCart, actualShoppingCart);
    }
}
