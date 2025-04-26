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

public class A4j_GetItemsFromCart_16_2_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetItemsFromCart() {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(cart.GetItemsFromCart(hmac, cartId)).thenReturn(expectedShoppingCart);
        // Act
        ShoppingCart result = a4j.GetItemsFromCart(hmac, cartId);
        // Assert
        assertNotNull(result);
        assertEquals(expectedShoppingCart, result);
        verify(cart, times(1)).GetItemsFromCart(hmac, cartId);
    }
}
