package net.kencochrane.a4j.beans;

import java.math.BigDecimal;
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

class // Add more tests for various edge cases
ShoppingCart_getItem_10_0_Test {

    @Test
    void getItem_itemExists() {
        // Arrange
        Items items = Mockito.mock(Items.class);
        ArrayList<Item> itemsArrayList = new ArrayList<>();
        Item item = new Item();
        item.setItemId("123");
        itemsArrayList.add(item);
        Mockito.when(items.getItemsArrayList()).thenReturn(itemsArrayList);
        ShoppingCart cart = new ShoppingCart();
        cart.setItems(items);
        // Act
        Item foundItem = cart.getItem("123");
        // Assert
        assertNotNull(foundItem);
        assertEquals("123", foundItem.getItemId());
    }

    @Test
    void getItem_itemDoesNotExist() {
        // Arrange
        Items items = Mockito.mock(Items.class);
        ArrayList<Item> itemsArrayList = new ArrayList<>();
        Mockito.when(items.getItemsArrayList()).thenReturn(itemsArrayList);
        ShoppingCart cart = new ShoppingCart();
        cart.setItems(items);
        // Act
        Item foundItem = cart.getItem("456");
        // Assert
        assertNull(foundItem);
    }

    @Test
    void getItem_emptyCart() {
        // Arrange
        Items items = Mockito.mock(Items.class);
        ArrayList<Item> itemsArrayList = new ArrayList<>();
        Mockito.when(items.getItemsArrayList()).thenReturn(itemsArrayList);
        ShoppingCart cart = new ShoppingCart();
        cart.setItems(items);
        // Act
        Item foundItem = cart.getItem("789");
        // Assert
        assertNull(foundItem);
    }

    @Test
    void getItem_nullItems() {
        // Arrange
        ShoppingCart cart = new ShoppingCart();
        cart.setItems(null);
        // Act
        Item foundItem = cart.getItem("abc");
        // Assert
        assertNull(foundItem);
    }
}
