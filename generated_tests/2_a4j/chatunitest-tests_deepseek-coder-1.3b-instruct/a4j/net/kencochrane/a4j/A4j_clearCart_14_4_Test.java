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

public class A4j_clearCart_14_4_Test {

    @Test
    public void testClearCart() {
        // Arrange
        String hmac = "hmac";
        String cartId = "cartId";
        ShoppingCart mockedShoppingCart = Mockito.mock(ShoppingCart.class);
        Cart mockedCart = Mockito.mock(Cart.class);
        // Mock the behavior of clearCart method
        Mockito.when(mockedCart.clearCart(hmac, cartId)).thenReturn(mockedShoppingCart);
        Mockito.when(mockedShoppingCart.getCartId()).thenReturn(cartId);
        // Act
        A4j a4j = new A4j();
        ShoppingCart result = a4j.clearCart(hmac, cartId);
        // Assert
        assertEquals(mockedShoppingCart, result);
    }
}
