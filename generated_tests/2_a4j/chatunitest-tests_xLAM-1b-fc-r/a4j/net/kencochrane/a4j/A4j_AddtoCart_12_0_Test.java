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

public class A4j_AddtoCart_12_0_Test {

    @Test
    public void testAddtoCart() {
        // Arrange
        A4j a4j = new A4j();
        String asin = "1234567890";
        String quantity = "2";
        ShoppingCart expectedCart = new ShoppingCart();
        // Act
        ShoppingCart actualCart = a4j.AddtoCart(asin, quantity);
        // Assert
        assertEquals(expectedCart, actualCart);
    }
}
