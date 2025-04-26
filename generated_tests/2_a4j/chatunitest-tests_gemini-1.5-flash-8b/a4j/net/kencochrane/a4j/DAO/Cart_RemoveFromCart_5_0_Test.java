package net.kencochrane.a4j.DAO;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.File;
import java.util.Optional;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.ShoppingCart;
import net.kencochrane.a4j.beans.ShoppingCartResponse;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.file.FileUtil;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class Cart_RemoveFromCart_5_0_Test {

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @Mock
    private FileInputStream fin;

    @Mock
    private JOXBeanInputStream joxIn;

    @BeforeEach
    void setUp() {
        cart = new Cart();
    }

    @Test
    void removeFromCart_success() throws IOException, ClassNotFoundException {
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        String queryString = "queryString";
        ShoppingCartResponse cartBean = new ShoppingCartResponse();
        ShoppingCart shoppingCart = new ShoppingCart();
        cartBean.setShoppingCart(shoppingCart);
        Mockito.when(query.RemoveFromCart(itemId, cartId, hmac)).thenReturn(queryString);
        Mockito.when(fileUtil.downloadCart(queryString)).thenReturn(new File("testFile"));
        Mockito.when(joxIn.readObject(ShoppingCartResponse.class)).thenReturn(cartBean);
        // Correctly mock the close methods
        Mockito.doNothing().when(fin).close();
        Mockito.doNothing().when(joxIn).close();
        ShoppingCart result = cart.RemoveFromCart(hmac, cartId, itemId);
        assertNotNull(result);
        assertEquals(shoppingCart, result);
    }

    @Test
    void removeFromCart_fileNotFound() {
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        String queryString = "queryString";
        Mockito.when(query.RemoveFromCart(itemId, cartId, hmac)).thenReturn(queryString);
        Mockito.when(fileUtil.downloadCart(queryString)).thenReturn(null);
        ShoppingCart result = cart.RemoveFromCart(hmac, cartId, itemId);
        assertNull(result);
    }

    @Test
    void removeFromCart_cartBeanNull() throws IOException, ClassNotFoundException {
        String hmac = "hmac";
        String cartId = "cartId";
        String itemId = "itemId";
        String queryString = "queryString";
        // Important:  Not null but without shopping cart
        ShoppingCartResponse cartBean = new ShoppingCartResponse();
        Mockito.when(query.RemoveFromCart(itemId, cartId, hmac)).thenReturn(queryString);
        Mockito.when(fileUtil.downloadCart(queryString)).thenReturn(new File("testFile"));
        Mockito.when(joxIn.readObject(ShoppingCartResponse.class)).thenReturn(cartBean);
        Mockito.doNothing().when(fin).close();
        Mockito.doNothing().when(joxIn).close();
        ShoppingCart result = cart.RemoveFromCart(hmac, cartId, itemId);
        // Correct assertion for null shopping cart
        assertNull(result);
    }
}
