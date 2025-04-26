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

class A4j_RemoveFromCart_17_1_Test {

    @Mock
    private Cart mockCart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testRemoveFromCart() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(mockCart.RemoveFromCart(hmac, cartId, itemId)).thenReturn(expectedShoppingCart);
        // Act
        ShoppingCart result = a4j.RemoveFromCart(hmac, cartId, itemId);
        // Assert
        assertNotNull(result, "The result should not be null");
        assertEquals(expectedShoppingCart, result, "The result should match the expected shopping cart");
        verify(mockCart, times(1)).RemoveFromCart(hmac, cartId, itemId);
    }
}
