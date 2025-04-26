package net.kencochrane.a4j.DAO;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.ShoppingCart;
import net.kencochrane.a4j.beans.ShoppingCartResponse;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.file.FileUtil;

class Cart_GetItemsFromCart_4_2_Test {

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetItemsFromCart() throws FileNotFoundException, IOException {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String queryString = "testQueryString";
        File file = mock(File.class);
        ShoppingCartResponse cartBean = mock(ShoppingCartResponse.class);
        ShoppingCart shoppingCart = mock(ShoppingCart.class);
        when(query.GetItemsFromCart(cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenReturn(mock(FileInputStream.class));
        when(cartBean.getShoppingCart()).thenReturn(shoppingCart);
        ShoppingCart result = cart.GetItemsFromCart(hmac, cartId);
        assertNotNull(result);
        assertEquals(shoppingCart, result);
    }

    @Test
    void testGetItemsFromCart_FileNull() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String queryString = "testQueryString";
        when(query.GetItemsFromCart(cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(null);
        ShoppingCart result = cart.GetItemsFromCart(hmac, cartId);
        assertNull(result);
    }

    @Test
    void testGetItemsFromCart_CartBeanNull() throws FileNotFoundException, IOException {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String queryString = "testQueryString";
        File file = mock(File.class);
        when(query.GetItemsFromCart(cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenReturn(mock(FileInputStream.class));
        ShoppingCart result = cart.GetItemsFromCart(hmac, cartId);
        assertNull(result);
    }

    @Test
    void testGetItemsFromCart_Exception() throws FileNotFoundException, IOException {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String queryString = "testQueryString";
        File file = mock(File.class);
        when(query.GetItemsFromCart(cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenThrow(new FileNotFoundException());
        ShoppingCart result = cart.GetItemsFromCart(hmac, cartId);
        assertNull(result);
    }
}
