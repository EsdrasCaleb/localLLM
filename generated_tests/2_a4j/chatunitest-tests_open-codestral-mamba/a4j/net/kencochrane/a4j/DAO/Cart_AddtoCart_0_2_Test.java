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
public class Cart_AddtoCart_0_2_Test {

    @Mock
    private Query query;

    @Mock
    private FileUtil fileUtil;

    @Mock
    private JOXBeanInputStream joxIn;

    @InjectMocks
    private Cart cart;

    @Test
    public void testAddtoCart() {
        String asin = "B00005NQWW";
        String quantity = "2";
        when(query.AddtoCart(asin, quantity)).thenReturn("queryString");
        when(fileUtil.downloadCart("queryString")).thenReturn(new File("cart.txt"));
        try {
            when(joxIn.readObject(ShoppingCartResponse.class)).thenReturn(new ShoppingCartResponse());
        } catch (IOException e) {
            fail("IOException should not be thrown");
        }
        ShoppingCart shoppingCart = cart.AddtoCart(asin, quantity);
        assertNotNull(shoppingCart);
    }
}
