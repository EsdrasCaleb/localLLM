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

public class Cart_GetItemsFromCart_4_0_Test {

    @Test
    void testGetItemsFromCart() {
        Cart cart = mock(Cart.class);
        ShoppingCart cartResponse = mock(ShoppingCart.class);
        when(cart.GetItemsFromCart("abc", "123")).thenReturn(cartResponse);
        ShoppingCart result = cart.GetItemsFromCart("abc", "123");
        assertEquals(cartResponse, result);
    }
}
