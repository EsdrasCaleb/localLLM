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

public class A4j_modifyCart_15_0_Test {

    @Test
    public void testModifyCart() {
        // Given
        A4j a4j = new A4j();
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        String quantity = "quantity";
        // Create a mock of Cart
        Cart mockCart = mock(Cart.class);
        when(mockCart.modifyCart(hmac, cartId, itemId, quantity)).thenReturn(new ShoppingCart());
        // When
        ShoppingCart result = a4j.modifyCart(hmac, cartId, itemId, quantity);
        // Then
        assertEquals(mockCart.modifyCart(hmac, cartId, itemId, quantity), result);
    }
}
