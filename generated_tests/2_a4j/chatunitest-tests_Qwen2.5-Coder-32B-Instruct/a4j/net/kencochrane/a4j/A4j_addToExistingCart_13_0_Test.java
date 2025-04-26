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

    @Mock
    private Cart mockCart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAddToExistingCart() {
        // Arrange
        String cartId = "12345";
        String hmac = "secureHmac";
        String asin = "B08N5WRWNW";
        String quantity = "2";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(mockCart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(expectedShoppingCart);
        // Act
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        // Assert
        assertEquals(expectedShoppingCart, result);
        verify(mockCart, times(1)).addToExistingCart(cartId, hmac, asin, quantity);
    }
}
