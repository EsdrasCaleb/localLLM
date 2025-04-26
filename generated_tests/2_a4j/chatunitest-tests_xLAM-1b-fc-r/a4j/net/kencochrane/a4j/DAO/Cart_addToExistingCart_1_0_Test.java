package net.kencochrane.a4j.DAO;

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
import java.io.IOException;

public class Cart_addToExistingCart_1_0_Test {

    @Test
    public void addToExistingCartTest() {
        // Given
        String cartId = "12345";
        String hmac = "hmac";
        String asin = "asin";
        String quantity = "1";
        Query query = new Query();
        FileUtil fileUtil = new FileUtil();
        JOXBeanInputStream joxIn = null;
        ShoppingCartResponse cartBean = new ShoppingCartResponse();
        File file = new File("file");
        // When
        ShoppingCart shoppingCart = new Cart().addToExistingCart(cartId, hmac, asin, quantity);
        // Then
        assertEquals(cartBean, shoppingCart);
    }
}
