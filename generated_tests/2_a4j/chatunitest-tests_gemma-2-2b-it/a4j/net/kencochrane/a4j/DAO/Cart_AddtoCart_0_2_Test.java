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

public class Cart_AddtoCart_0_2_Test {

    @Test
    void AddtoCart_ValidInput_ReturnsShoppingCart() {
        Cart cart = new Cart();
        ShoppingCart expectedShoppingCart = new ShoppingCart();
        ShoppingCart actualShoppingCart = cart.AddtoCart("1234567890", "2");
        assertEquals(expectedShoppingCart, actualShoppingCart);
    }

    @Test
    void AddtoCart_InvalidInput_ThrowsException() {
        Cart cart = new Cart();
        assertThrows(IllegalArgumentException.class, () -> cart.AddtoCart("1234567890", "abc"));
    }
}
