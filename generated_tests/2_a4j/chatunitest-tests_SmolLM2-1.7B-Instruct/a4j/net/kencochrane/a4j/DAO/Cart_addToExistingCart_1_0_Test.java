package net.kencochrane.a4j.DAO;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import static org.junit.Assert.assertNotNull;
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
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

@RunWith(MockitoJUnitRunner.class)
public class Cart_addToExistingCart_1_0_Test {

    @Mock
    private Cart cart;

    @InjectMocks
    private Cart cartUnderTest;

    @Test
    public void testAddToExistingCart() throws IOException {
        // Arrange
        String cartId = "existingCartId";
        String hmac = "existingHmac";
        String asin = "productAsin";
        String quantity = "1";
        // Act
        ShoppingCart shoppingCart = cartUnderTest.addToExistingCart(cartId, hmac, asin, quantity);
        // Assert
        assertNotNull(shoppingCart);
    }
}
