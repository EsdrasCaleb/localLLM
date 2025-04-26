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
