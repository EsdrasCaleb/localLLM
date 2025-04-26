package net.kencochrane.a4j.DAO;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.ShoppingCart;
import net.kencochrane.a4j.beans.ShoppingCartResponse;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.file.FileUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

@ExtendWith(MockitoExtension.class)
public class Cart_clearCart_2_0_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Cart cart;

    @Test
    public void testClearCart() throws Exception {
        // Arrange
        String hmac = "testHmac";
        String cartId = "testCartId";
        when(fileUtil.downloadCart(anyString())).thenReturn(new File("path/to/cart"));
        // Act
        ShoppingCart result = cart.clearCart(hmac, cartId);
        // Assert
        assertNotNull(result);
        verify(fileUtil).downloadCart(anyString());
    }
}
