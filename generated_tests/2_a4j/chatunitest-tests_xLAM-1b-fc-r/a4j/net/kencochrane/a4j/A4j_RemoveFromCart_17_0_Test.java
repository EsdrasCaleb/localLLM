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

class A4j_RemoveFromCart_17_0_Test {

    @Test
    void removeFromCart() {
        // Arrange
        A4j a4j = new A4j();
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        ShoppingCart expected = new ShoppingCart();
        // Act
        ShoppingCart result = a4j.RemoveFromCart(hmac, cartId, itemId);
        // Assert
        assertEquals(expected, result);
    }
}
