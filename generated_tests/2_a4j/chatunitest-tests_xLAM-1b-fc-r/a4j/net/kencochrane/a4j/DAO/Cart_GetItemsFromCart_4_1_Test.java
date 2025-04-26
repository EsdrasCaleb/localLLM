package net.kencochrane.a4j.DAO;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.beans.ShoppingCart;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.ShoppingCartResponse;
import net.kencochrane.a4j.data.Query;
import net.kencochrane.a4j.file.FileUtil;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

@ExtendWith(MockitoExtension.class)
public class Cart_GetItemsFromCart_4_1_Test {

    @Test
    public void testGetItemsFromCart() {
        // Arrange
        String hmac = "hmac";
        String cartId = "cartId";
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        // Act
        ShoppingCart actualShoppingCart = new Cart().GetItemsFromCart(hmac, cartId);
        // Assert
        assertEquals(expectedShoppingCart, actualShoppingCart);
    }
}
