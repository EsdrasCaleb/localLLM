package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
class A4j_clearCart_14_0_Test {

    private A4j a4j;

    @Mock
    private Cart mockCart;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        try {
            Field cartField = A4j.class.getDeclaredField("cart");
            cartField.setAccessible(true);
            cartField.set(a4j, mockCart);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set mock Cart instance: " + e.getMessage());
        }
    }

    @Test
    void testClearCart() {
        String hmac = "validHmac";
        String cartId = "validCartId";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.clearCart(hmac, cartId)).thenReturn(expectedCart);
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertNotNull(result);
        assertSame(expectedCart, result);
        verify(mockCart).clearCart(hmac, cartId);
    }

    @Test
    void testClearCartWithNullHmac() {
        String hmac = null;
        String cartId = "validCartId";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.clearCart(hmac, cartId)).thenReturn(expectedCart);
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertNotNull(result);
        assertSame(expectedCart, result);
        verify(mockCart).clearCart(hmac, cartId);
    }

    @Test
    void testClearCartWithNullCartId() {
        String hmac = "validHmac";
        String cartId = null;
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.clearCart(hmac, cartId)).thenReturn(expectedCart);
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertNotNull(result);
        assertSame(expectedCart, result);
        verify(mockCart).clearCart(hmac, cartId);
    }

    @Test
    void testClearCartWithEmptyCartId() {
        String hmac = "validHmac";
        String cartId = "";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.clearCart(hmac, cartId)).thenReturn(expectedCart);
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertNotNull(result);
        assertSame(expectedCart, result);
        verify(mockCart).clearCart(hmac, cartId);
    }
}
