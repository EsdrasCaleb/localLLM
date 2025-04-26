package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.A4j;
import net.kencochrane.a4j.DAO.Cart;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_addToExistingCart_13_3_Test {

    private A4j a4j;

    private Cart mockCart;

    @BeforeEach
    public void setUp() {
        a4j = new A4j();
        mockCart = mock(Cart.class);
    }

    @Test
    public void testAddToExistingCart() throws Exception {
        // Given
        String cartId = "123";
        String hmac = "abc";
        String asin = "0-619-70810-X";
        String quantity = "2";
        // When
        ShoppingCart result = a4j.addToExistingCart(cartId, hmac, asin, quantity);
        // Then
        assertEquals(result, mockCart);
    }
}
