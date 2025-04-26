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
public class Cart_addToExistingCart_1_0_Test {

    @InjectMocks
    private Cart cart;

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @Test
    void addToExistingCart() {
        String cartId = "123";
        String hmac = "abc";
        String asin = "123456";
        String quantity = "1";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        when(query.AddToExistingCart(asin, quantity, cartId, hmac)).thenReturn("queryString");
        when(fileUtil.downloadCart("queryString")).thenReturn(new File("path"));
        ShoppingCart actualShoppingCart = cart.addToExistingCart(cartId, hmac, asin, quantity);
        assertEquals(expectedShoppingCart, actualShoppingCart);
    }
}
