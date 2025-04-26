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

class A4j_AddtoCart_12_0_Test {

    private A4j a4j;

    private Cart mockCart;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        mockCart = mock(Cart.class);
    }

    @Test
    void testAddtoCart_ValidInputs() {
        // Arrange
        String asin = "B000123456";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.AddtoCart(asin, quantity)).thenReturn(expectedCart);
        // Act
        ShoppingCart result = a4j.AddtoCart(asin, quantity);
        // Assert
        assertEquals(expectedCart, result);
        verify(mockCart).AddtoCart(asin, quantity);
    }

    @Test
    void testAddtoCart_EmptyASIN() {
        // Arrange
        String asin = "";
        String quantity = "1";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.AddtoCart(asin, quantity)).thenReturn(expectedCart);
        // Act
        ShoppingCart result = a4j.AddtoCart(asin, quantity);
        // Assert
        assertEquals(expectedCart, result);
        verify(mockCart).AddtoCart(asin, quantity);
    }

    @Test
    void testAddtoCart_NullASIN() {
        // Arrange
        String asin = null;
        String quantity = "1";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.AddtoCart(asin, quantity)).thenReturn(expectedCart);
        // Act
        ShoppingCart result = a4j.AddtoCart(asin, quantity);
        // Assert
        assertEquals(expectedCart, result);
        verify(mockCart).AddtoCart(asin, quantity);
    }

    @Test
    void testAddtoCart_NegativeQuantity() {
        // Arrange
        String asin = "B000123456";
        String quantity = "-1";
        ShoppingCart expectedCart = new ShoppingCart();
        when(mockCart.AddtoCart(asin, quantity)).thenReturn(expectedCart);
        // Act
        ShoppingCart result = a4j.AddtoCart(asin, quantity);
        // Assert
        assertEquals(expectedCart, result);
        verify(mockCart).AddtoCart(asin, quantity);
    }
}
