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

public class A4j_clearCart_14_0_Test {

    @Test
    public void clearCartTest() {
        // Arrange
        A4j a4j = new A4j();
        String hmac = "hmac";
        String cartId = "cartId";
        ShoppingCart expectedCart = new ShoppingCart();
        // Act
        ShoppingCart actualCart = a4j.clearCart(hmac, cartId);
        // Assert
        assertEquals(expectedCart, actualCart);
    }
}
