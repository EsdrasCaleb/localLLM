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

    @Test
    public void addToExistingCartTest() {
        // Arrange
        A4j a4j = new A4j();
        String cartId = "cartId";
        String hmac = "hmac";
        String asin = "asin";
        String quantity = "quantity";
        // Mock the Cart object
        Cart mockCart = Mockito.mock(Cart.class);
        when(mockCart.addToExistingCart(cartId, hmac, asin, quantity)).thenReturn(new ShoppingCart());
        // Act
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        // Assert
        assertEquals(mockCart, result);
    }
}
