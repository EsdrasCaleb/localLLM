package net.kencochrane.a4j.DAO;

import java.io.File;
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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class Cart_GetItemsFromCart_4_1_Test {

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetItemsFromCart() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String queryString = "testQueryString";
        File file = new File("testFilePath");
        when(query.GetItemsFromCart(cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        ShoppingCart shoppingCart = cart.GetItemsFromCart(hmac, cartId);
        assertNotNull(shoppingCart);
    }
}
