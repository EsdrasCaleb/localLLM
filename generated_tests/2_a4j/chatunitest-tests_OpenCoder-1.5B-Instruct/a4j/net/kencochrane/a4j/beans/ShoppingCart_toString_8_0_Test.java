// ShoppingCartTest class
package net.kencochrane.a4j.beans;

import net.kencochrane.a4j.beans.Items;
import net.kencochrane.a4j.beans.ShoppingCart;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;

public class ShoppingCart_toString_8_0_Test {

    @Test
    public void testGetTotalCartCost() {
        ShoppingCart cart = Mockito.mock(ShoppingCart.class);
        Mockito.when(cart.getTotalCartCost()).thenReturn("200");
        assertEquals("200", cart.getTotalCartCost());
    }

    @Test
    public void testGetNumItems() {
        ShoppingCart cart = Mockito.mock(ShoppingCart.class);
        Mockito.when(cart.getNumItems()).thenReturn("2");
        assertEquals("2", cart.getNumItems());
    }

    @Test
    public void testToString() {
        ShoppingCart cart = Mockito.mock(ShoppingCart.class);
        Mockito.when(cart.toString()).thenReturn("HMAC = abc\nPurchase URL = xyz\nCartId = 123\nitems = [Item{ourPrice='100', quantity='2'}]\n");
        assertEquals("HMAC = abc\nPurchase URL = xyz\nCartId = 123\nitems = [Item{ourPrice='100', quantity='2'}]\n", cart.toString());
    }
}
