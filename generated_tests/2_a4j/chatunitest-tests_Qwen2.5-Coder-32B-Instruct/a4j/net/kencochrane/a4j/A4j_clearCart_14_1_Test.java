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

public class A4j_clearCart_14_1_Test {

    @Mock
    private Cart mockCart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testClearCart_Success() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(mockCart.clearCart(hmac, cartId)).thenReturn(expectedShoppingCart);
        // Act
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        // Assert
        assertEquals(expectedShoppingCart, result);
        verify(mockCart, times(1)).clearCart(hmac, cartId);
    }

    @Test
    void testClearCart_Exception() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        RuntimeException expectedException = new RuntimeException("Error clearing cart");
        when(mockCart.clearCart(hmac, cartId)).thenThrow(expectedException);
        // Act & Assert
        Exception exception = assertThrows(RuntimeException.class, () -> {
            a4j.clearCart(hmac, cartId);
        });
        assertEquals(expectedException.getMessage(), exception.getMessage());
        verify(mockCart, times(1)).clearCart(hmac, cartId);
    }
}
