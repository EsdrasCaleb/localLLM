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

public class A4j_modifyCart_15_4_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testModifyCart() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "2";
        ShoppingCart mockShoppingCart = mock(ShoppingCart.class);
        when(cart.modifyCart(hmac, cartId, itemId, quantity)).thenReturn(mockShoppingCart);
        ShoppingCart result = a4j.modifyCart(hmac, cartId, itemId, quantity);
        assertNotNull(result);
        assertEquals(mockShoppingCart, result);
        verify(cart, times(1)).modifyCart(hmac, cartId, itemId, quantity);
    }
}
