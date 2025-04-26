package net.kencochrane.a4j.beans;

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

public class ShoppingCart_getItem_10_0_Test {

    private ShoppingCart cart;

    @BeforeEach
    public void setUp() {
        cart = new ShoppingCart();
        cart.setItems(new Items());
        cart.setHMAC("HMAC123");
        cart.setPurchaseURL("https://example.com/checkout");
    }

    @Test
    public void getItem_WithCorrectItemId_ReturnsItem() {
        String itemId = "123";
        Item item = cart.getItem(itemId);
        assertNotNull(item);
        assertEquals(itemId, item.getItemId());
    }

    @Test
    public void getItem_WithoutCorrectItemId_ReturnsNull() {
        String itemId = "456";
        Item item = cart.getItem(itemId);
        assertNull(item);
    }

    @Test
    public void getItem_EmptyCart_ReturnsNull() {
        cart.setItems(null);
        Item item = cart.getItem("456");
        assertNull(item);
    }

    @Test
    public void getItem_InvalidItemId_ReturnsNull() {
        String itemId = "789";
        Item item = cart.getItem(itemId);
        assertNull(item);
    }
}
