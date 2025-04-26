package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_AddtoCart_12_2_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testAddtoCart() {
        // Given
        String asin = "123";
        String quantity = "2";
        when(cart.AddtoCart(asin, quantity)).thenReturn(new ShoppingCart());
        // When
        ShoppingCart shoppingCart = a4j.AddtoCart(asin, quantity);
        // Then
        assertEquals(new ShoppingCart(), shoppingCart);
    }
}
