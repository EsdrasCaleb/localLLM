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

class A4j_clearCart_14_4_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testClearCart() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        ShoppingCart mockShoppingCart = mock(ShoppingCart.class);
        when(cart.clearCart(hmac, cartId)).thenReturn(mockShoppingCart);
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertNotNull(result);
        assertEquals(mockShoppingCart, result);
        verify(cart, times(1)).clearCart(hmac, cartId);
    }
}
