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

public class A4j_addToExistingCart_13_0_Test {

    @InjectMocks
    private A4j a4j;

    @Mock
    private Cart cart;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAddToExistingCart_Success() {
        String cartId = "cart123";
        String hmac = "validHmac";
        String asin = "item123";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(expectedCart, result);
        verify(cart).addToExistingCart(cartId, hmac, asin, quantity);
    }

    @Test
    public void testAddToExistingCart_NullCartId() {
        String cartId = null;
        String hmac = "validHmac";
        String asin = "item123";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(expectedCart, result);
        verify(cart).addToExistingCart(cartId, hmac, asin, quantity);
    }

    @Test
    public void testAddToExistingCart_EmptyHmac() {
        String cartId = "cart123";
        String hmac = "";
        String asin = "item123";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(expectedCart, result);
        verify(cart).addToExistingCart(cartId, hmac, asin, quantity);
    }

    @Test
    public void testAddToExistingCart_InvalidAsin() {
        String cartId = "cart123";
        String hmac = "validHmac";
        String asin = "invalidAsin";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(expectedCart, result);
        verify(cart).addToExistingCart(cartId, hmac, asin, quantity);
    }

    @Test
    public void testAddToExistingCart_NegativeQuantity() {
        String cartId = "cart123";
        String hmac = "validHmac";
        String asin = "item123";
        String quantity = "-1";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedCart);
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(expectedCart, result);
        verify(cart).addToExistingCart(cartId, hmac, asin, quantity);
    }
}
