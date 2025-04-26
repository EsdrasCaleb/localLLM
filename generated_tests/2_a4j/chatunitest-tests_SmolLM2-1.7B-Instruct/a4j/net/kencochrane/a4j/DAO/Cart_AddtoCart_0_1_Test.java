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

public class Cart_AddtoCart_0_1_Test {

    @Test
    public void testAddtoCart() {
        // Arrange
        Cart cart = new Cart();
        String asin = "B01M8L5LQS";
        String quantity = "2";
        // Act
        ShoppingCart shoppingCart = cart.AddtoCart(asin, quantity);
        // Assert
        assertNotNull(shoppingCart);
        assertTrue(shoppingCart instanceof ShoppingCart);
    }
}
