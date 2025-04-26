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

class Cart_modifyCart_3_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    void testModifyCart() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "0";
        ShoppingCart expectedCart = null;
        when(fileUtil.downloadCart(Mockito.anyString())).thenReturn(null);
        ShoppingCart actualCart = cart.modifyCart(hmac, cartId, itemId, quantity);
        assertEquals(expectedCart, actualCart);
    }

    @Test
    void testModifyCartIOException() {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "5";
        ShoppingCart expectedCart = null;
        File file = new File("testFile");
        when(fileUtil.downloadCart(Mockito.anyString())).thenReturn(file);
        try (FileInputStream fin = new FileInputStream(file)) {
            when(fin.read()).thenThrow(new IOException());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        ShoppingCart actualCart = cart.modifyCart(hmac, cartId, itemId, quantity);
        assertEquals(expectedCart, actualCart);
    }
}
