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
public class A4j_clearCart_14_0_Test {

    @Mock
    private Cart cartMock;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testClearCart() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        assertNotNull(result);
        verify(cartMock).clearCart(hmac, cartId);
    }
}
