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

public class Cart_clearCart_2_2_Test {

    @Test
    public void testClearCart() {
        ShoppingCart shoppingCart = new ShoppingCart();
        String cartId = "cartId";
        String hmac = "hmac";
        ShoppingCartResponse shoppingCartResponse = new ShoppingCartResponse();
        shoppingCartResponse.setShoppingCart(shoppingCart);
        Cart cart = new Cart();
        cart.clearCart(hmac, cartId);
        assertNotNull(shoppingCartResponse.getShoppingCart());
    }
}
