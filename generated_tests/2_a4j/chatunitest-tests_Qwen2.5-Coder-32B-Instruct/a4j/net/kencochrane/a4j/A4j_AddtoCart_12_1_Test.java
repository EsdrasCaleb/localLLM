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

public class A4j_AddtoCart_12_1_Test {

    @Mock
    private Cart mockCart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAddtoCart() {
        // Arrange
        String asin = "B08N5WRWNW";
        String quantity = "2";
        // Assuming ShoppingCart has a default constructor
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        // Mock the behavior of the Cart's AddtoCart method
        when(mockCart.AddtoCart(asin, quantity)).thenReturn(expectedShoppingCart);
        // Act
        ShoppingCart result = a4j.AddtoCart(asin, quantity);
        // Assert
        assertEquals(expectedShoppingCart, result);
        verify(mockCart, times(1)).AddtoCart(asin, quantity);
    }
}
