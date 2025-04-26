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

public class A4j_modifyCart_15_0_Test {

    @Mock
    private Cart cartMock;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testModifyCart_Success() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "testQuantity";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(cartMock.modifyCart(hmac, cartId, itemId, quantity)).thenReturn(expectedShoppingCart);
        // Act
        ShoppingCart result = a4j.modifyCart(hmac, cartId, itemId, quantity);
        // Assert
        assertEquals(expectedShoppingCart, result);
        verify(cartMock, times(1)).modifyCart(hmac, cartId, itemId, quantity);
    }

    @Test
    public void testModifyCart_ExceptionHandling() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "testQuantity";
        when(cartMock.modifyCart(hmac, cartId, itemId, quantity)).thenThrow(new RuntimeException("Test Exception"));
        // Act & Assert
        assertThrows(RuntimeException.class, () -> {
            a4j.modifyCart(hmac, cartId, itemId, quantity);
        });
        verify(cartMock, times(1)).modifyCart(hmac, cartId, itemId, quantity);
    }
}
