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
public class A4j_AddtoCart_12_0_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    private ShoppingCart shoppingCart;

    @BeforeEach
    public void setUp() {
        shoppingCart = new ShoppingCart();
    }

    @Test
    public void testAddtoCart() {
        String asin = "12345";
        String quantity = "2";
        when(cart.AddtoCart(asin, quantity)).thenReturn(shoppingCart);
        ShoppingCart result = a4j.AddtoCart(asin, quantity);
        assertNotNull(result);
        assertEquals(shoppingCart, result);
        verify(cart, times(1)).AddtoCart(asin, quantity);
    }
}
