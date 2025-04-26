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

class Cart_addToExistingCart_1_0_Test {

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
    void testAddToExistingCart() throws FileNotFoundException, IOException {
        String cartId = "123";
        String hmac = "abc";
        String asin = "456";
        String quantity = "2";
        String queryString = "queryString";
        File file = mock(File.class);
        FileInputStream fin = mock(FileInputStream.class);
        JOXBeanInputStream joxIn = mock(JOXBeanInputStream.class);
        ShoppingCartResponse cartBean = mock(ShoppingCartResponse.class);
        ShoppingCart shoppingCart = mock(ShoppingCart.class);
        when(query.AddToExistingCart(asin, quantity, cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenReturn(fin);
        when(new JOXBeanInputStream(fin)).thenReturn(joxIn);
        when(joxIn.readObject(ShoppingCartResponse.class)).thenReturn(cartBean);
        when(cartBean.getShoppingCart()).thenReturn(shoppingCart);
        ShoppingCart result = cart.addToExistingCart(cartId, hmac, asin, quantity);
        assertNotNull(result);
        assertEquals(shoppingCart, result);
        verify(query).AddToExistingCart(asin, quantity, cartId, hmac);
        verify(fileUtil).downloadCart(queryString);
        verify(joxIn).close();
        verify(fin).close();
    }

    @Test
    void testAddToExistingCartFileNull() {
        String cartId = "123";
        String hmac = "abc";
        String asin = "456";
        String quantity = "2";
        String queryString = "queryString";
        when(query.AddToExistingCart(asin, quantity, cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(null);
        ShoppingCart result = cart.addToExistingCart(cartId, hmac, asin, quantity);
        assertNull(result);
        verify(query).AddToExistingCart(asin, quantity, cartId, hmac);
        verify(fileUtil).downloadCart(queryString);
    }

    @Test
    void testAddToExistingCartCartBeanNull() throws FileNotFoundException, IOException {
        String cartId = "123";
        String hmac = "abc";
        String asin = "456";
        String quantity = "2";
        String queryString = "queryString";
        File file = mock(File.class);
        FileInputStream fin = mock(FileInputStream.class);
        JOXBeanInputStream joxIn = mock(JOXBeanInputStream.class);
        when(query.AddToExistingCart(asin, quantity, cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenReturn(fin);
        when(new JOXBeanInputStream(fin)).thenReturn(joxIn);
        when(joxIn.readObject(ShoppingCartResponse.class)).thenReturn(null);
        ShoppingCart result = cart.addToExistingCart(cartId, hmac, asin, quantity);
        assertNull(result);
        verify(query).AddToExistingCart(asin, quantity, cartId, hmac);
        verify(fileUtil).downloadCart(queryString);
        verify(joxIn).close();
        verify(fin).close();
    }

    @Test
    void testAddToExistingCartShoppingCartNull() throws FileNotFoundException, IOException {
        String cartId = "123";
        String hmac = "abc";
        String asin = "456";
        String quantity = "2";
        String queryString = "queryString";
        File file = mock(File.class);
        FileInputStream fin = mock(FileInputStream.class);
        JOXBeanInputStream joxIn = mock(JOXBeanInputStream.class);
        ShoppingCartResponse cartBean = mock(ShoppingCartResponse.class);
    }
}
