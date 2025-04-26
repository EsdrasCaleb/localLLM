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
public class A4j_modifyCart_15_0_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @Test
    public void modifyCartTest() {
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        String quantity = "quantity";
        ShoppingCart expected = new ShoppingCart();
        when(cart.modifyCart(hmac, cartId, itemId, quantity)).thenReturn(expected);
        ShoppingCart result = a4j.modifyCart(hmac, cartId, itemId, quantity);
        assertEquals(expected, result);
    }
}
