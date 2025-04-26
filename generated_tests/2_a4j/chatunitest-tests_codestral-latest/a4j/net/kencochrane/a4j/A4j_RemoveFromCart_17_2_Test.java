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

public class A4j_RemoveFromCart_17_2_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testRemoveFromCart() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.RemoveFromCart(hmac, cartId, itemId)).thenReturn(expectedCart);
        // Act
        ShoppingCart result = a4j.RemoveFromCart(hmac, cartId, itemId);
        // Assert
        assertEquals(expectedCart, result);
        verify(cart, times(1)).RemoveFromCart(hmac, cartId, itemId);
    }
}
