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

class Cart_addToExistingCart_1_3_Test {

    @Test
    void addToExistingCart() {
        // Given
        String cartId = "cartId";
        String hmac = "hmac";
        String asin = "asin";
        String quantity = "quantity";
        // Mock the dependencies
        Cart cart = new Cart();
        Query query = Mockito.mock(Query.class);
        FileUtil fileUtil = Mockito.mock(FileUtil.class);
        // Mock the query method
        Mockito.when(query.AddToExistingCart(asin, quantity, cartId, hmac)).thenReturn("queryString");
        // Mock the fileUtil method
        Mockito.when(fileUtil.downloadCart("queryString")).thenReturn(new File("file"));
        // When
        ShoppingCart shoppingCart = cart.addToExistingCart(cartId, hmac, asin, quantity);
        // Then
        assertNotNull(shoppingCart);
    }
}
