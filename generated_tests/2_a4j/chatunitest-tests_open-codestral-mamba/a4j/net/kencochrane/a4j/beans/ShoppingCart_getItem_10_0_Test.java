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

    private ShoppingCart shoppingCart;

    private Items items;

    private Item item;

    @BeforeEach
    public void setup() {
        shoppingCart = new ShoppingCart();
        items = Mockito.mock(Items.class);
        shoppingCart.setItems(items);
        item = new Item();
        item.setItemId("123");
        item.setOurPrice("10.00");
        item.setQuantity("2");
        ArrayList<Item> itemsList = new ArrayList<>();
        itemsList.add(item);
        Mockito.when(items.getItemsArrayList()).thenReturn(itemsList);
    }

    @Test
    public void testGetItemFound() {
        Item result = shoppingCart.getItem("123");
        assertEquals("123", result.getItemId());
        assertEquals("10.00", result.getOurPrice());
        assertEquals("2", result.getQuantity());
    }

    @Test
    public void testGetItemNotFound() {
        Item result = shoppingCart.getItem("456");
        assertNull(result);
    }
}
