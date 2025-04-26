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
public class A4j_addToExistingCart_13_0_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    private String cartId;

    private String hmac;

    private String asin;

    private String quantity;

    private ShoppingCart shoppingCart;

    @BeforeEach
    public void setUp() {
        cartId = "cart123";
        hmac = "hmac123";
        asin = "asin123";
        quantity = "2";
        shoppingCart = new ShoppingCart();
    }

    @Test
    public void testAddToExistingCart() {
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(shoppingCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(shoppingCart, result);
        verify(cart, times(1)).addToExistingCart(cartId, hmac, asin, quantity);
    }
}
