package net.kencochrane.a4j.DAO;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
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
class Cart_clearCart_2_3_Test {

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    void testClearCart() throws FileNotFoundException {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String queryString = "testQueryString";
        File file = new File("testFilePath");
        ShoppingCartResponse cartBean = new ShoppingCartResponse();
        cartBean.setShoppingCart(new ShoppingCart());
        when(query.ClearCart(cartId, hmac)).thenReturn(queryString);
        when(fileUtil.downloadCart(queryString)).thenReturn(file);
        try {
            try (JOXBeanInputStream joxBeanInputStream = new JOXBeanInputStream(new FileInputStream(file))) {
                when(joxBeanInputStream.readObject(ShoppingCartResponse.class)).thenReturn(cartBean);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        ShoppingCart result = cart.clearCart(hmac, cartId);
        assertNotNull(result);
    }
}
