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

public class Cart_clearCart_2_1_Test {

    @Test
    public void clearCartTest() {
        // Given
        String hmac = "hmac";
        String cartId = "cartId";
        Cart cart = new Cart();
        Query mockQuery = mock(Query.class);
        FileUtil mockFileUtil = mock(FileUtil.class);
        ShoppingCartResponse mockShoppingCartResponse = mock(ShoppingCartResponse.class);
        ShoppingCart mockShoppingCart = mock(ShoppingCart.class);
        when(mockQuery.ClearCart(cartId, hmac)).thenReturn("queryString");
        when(mockFileUtil.downloadCart("queryString")).thenReturn(new File("file"));
        when(mockFileUtil.downloadCart("queryString")).thenReturn(null);
        when(mockShoppingCartResponse.getShoppingCart()).thenReturn(mockShoppingCart);
        // When
        ShoppingCart result = cart.clearCart(hmac, cartId);
        // Then
        assertEquals(mockShoppingCart, result);
    }
}
