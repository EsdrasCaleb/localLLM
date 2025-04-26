package net.kencochrane.a4j.beans;

import java.util.ArrayList;
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

class ShoppingCart_getItem_10_0_Test {

    private ShoppingCart shoppingCart;

    private Items items;

    @BeforeEach
    void setUp() {
        shoppingCart = new ShoppingCart();
        items = Mockito.mock(Items.class);
        shoppingCart.setItems(items);
    }

    @Test
    void testGetItem_ItemExists() {
        // Arrange
        String itemId = "item1";
        Item item = Mockito.mock(Item.class);
        Mockito.when(item.getItemId()).thenReturn(itemId);
        ArrayList<Item> itemList = new ArrayList<>();
        itemList.add(item);
        Mockito.when(items.getItemsArrayList()).thenReturn(itemList);
        shoppingCart.setItems(items);
        // Act
        Item result = shoppingCart.getItem(itemId);
        // Assert
        assertSame(item, result);
    }

    @Test
    void testGetItem_EmptyCart() {
        // Arrange
        Mockito.when(items.getItemsArrayList()).thenReturn(new ArrayList<>());
        shoppingCart.setItems(items);
        // Act
        Item result = shoppingCart.getItem("item1");
        // Assert
        assertNull(result);
    }
}
