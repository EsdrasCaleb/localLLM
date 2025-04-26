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
    public void testModifyCart_IOException() throws Exception {
        String hmac = "testHmac";
        String cartId = "testCartId";
        String itemId = "testItemId";
        String quantity = "2";
    }
}
