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

public class Cart_modifyCart_3_4_Test {

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testModifyCart_QuantityZero() throws Exception {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "0";
        ShoppingCart expectedCart = new ShoppingCart();
        when(cart.RemoveFromCart(hmac, cartId, itemId)).thenReturn(expectedCart);
        ShoppingCart result = cart.modifyCart(hmac, cartId, itemId, quantity);
        assertEquals(expectedCart, result);
        verify(cart, times(1)).RemoveFromCart(hmac, cartId, itemId);
    }

    @Test
    public void testModifyCart_QuantityNotZero() throws Exception {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "2";
        String queryString = "testQueryString";
        File file = mock(File.class);
        FileInputStream fin = mock(FileInputStream.class);
        JOXBeanInputStream joxIn = mock(JOXBeanInputStream.class);
        ShoppingCartResponse cartBean = mock(ShoppingCartResponse.class);
        ShoppingCart expectedCart = new ShoppingCart();
        when(query.ModifyCart(itemId, quantity, cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenReturn(fin);
        when(new JOXBeanInputStream(fin)).thenReturn(joxIn);
        when(joxIn.readObject(ShoppingCartResponse.class)).thenReturn(cartBean);
        when(cartBean.getShoppingCart()).thenReturn(expectedCart);
        ShoppingCart result = cart.modifyCart(hmac, cartId, itemId, quantity);
        assertEquals(expectedCart, result);
        verify(query, times(1)).ModifyCart(itemId, quantity, cartId, hmac);
        verify(fileUtil, times(1)).downloadCart(queryString);
        verify(joxIn, times(1)).readObject(ShoppingCartResponse.class);
        verify(joxIn, times(1)).close();
        verify(fin, times(1)).close();
    }

    @Test
    public void testModifyCart_FileNull() throws Exception {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "2";
        String queryString = "testQueryString";
        when(query.ModifyCart(itemId, quantity, cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(null);
        ShoppingCart result = cart.modifyCart(hmac, cartId, itemId, quantity);
        assertNull(result);
        verify(query, times(1)).ModifyCart(itemId, quantity, cartId, hmac);
        verify(fileUtil, times(1)).downloadCart(queryString);
    }

    @Test
    public void testModifyCart_FileNotFoundException() throws Exception {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "2";
        String queryString = "testQueryString";
        File file = mock(File.class);
        FileInputStream fin = mock(FileInputStream.class);
        when(query.ModifyCart(itemId, quantity, cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        when(new FileInputStream(file)).thenThrow(FileNotFoundException.class);
        ShoppingCart result = cart.modifyCart(hmac, cartId, itemId, quantity);
        assertNull(result);
        verify(query, times(1)).ModifyCart(itemId, quantity, cartId, hmac);
        verify(fileUtil, times(1)).downloadCart(queryString);
    }

    @Test
    public void testModifyCart_IOException() throws Exception {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "2";
    }
}
