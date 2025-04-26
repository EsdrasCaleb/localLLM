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

    @InjectMocks
    private ShoppingCart shoppingCart;

    @Mock
    private Items items;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGetItem_ItemFound() {
        // Arrange
        Item item1 = new Item();
        item1.setItemId("123");
        Item item2 = new Item();
        item2.setItemId("456");
        ArrayList<Item> itemList = new ArrayList<>();
        itemList.add(item1);
        itemList.add(item2);
        when(items.getItemsArrayList()).thenReturn(itemList);
        shoppingCart.setItems(items);
        // Act
        Item result = shoppingCart.getItem("123");
        // Assert
        assertNotNull(result);
        assertEquals("123", result.getItemId());
    }

    @Test
    void testGetItem_ItemNotFound() {
        // Arrange
        Item item1 = new Item();
        item1.setItemId("123");
        Item item2 = new Item();
        item2.setItemId("456");
        ArrayList<Item> itemList = new ArrayList<>();
        itemList.add(item1);
        itemList.add(item2);
        when(items.getItemsArrayList()).thenReturn(itemList);
        shoppingCart.setItems(items);
        // Act
        Item result = shoppingCart.getItem("789");
        // Assert
        assertNull(result);
    }

    @Test
    void testGetItem_NoItems() {
        // Arrange
        when(items.getItemsArrayList()).thenReturn(null);
        shoppingCart.setItems(items);
        // Act
        Item result = shoppingCart.getItem("123");
        // Assert
        assertNull(result);
    }

    @Test
    void testGetItem_EmptyItemList() {
        // Arrange
        ArrayList<Item> itemList = new ArrayList<>();
        when(items.getItemsArrayList()).thenReturn(itemList);
        shoppingCart.setItems(items);
        // Act
        Item result = shoppingCart.getItem("123");
        // Assert
        assertNull(result);
    }
}
