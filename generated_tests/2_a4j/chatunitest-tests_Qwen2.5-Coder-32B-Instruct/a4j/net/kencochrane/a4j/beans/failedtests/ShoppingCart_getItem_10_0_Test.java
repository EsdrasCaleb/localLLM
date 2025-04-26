package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
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

public class ShoppingCart_getItem_10_0_Test {

    @Mock
    private Items items;

    @InjectMocks
    private ShoppingCart shoppingCart;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetItem_ItemFound() throws Exception {
        // Arrange
        Item mockItem = new Item();
        mockItem.setItemId("item123");
        mockItem.setQuantity("1");
        mockItem.setOurPrice("10.00");
        ArrayList<Item> itemsList = new ArrayList<>();
        itemsList.add(mockItem);
        when(items.getItemsArrayList()).thenReturn(itemsList);
        // Set the 'items' field in ShoppingCart using reflection
        Field itemsField = ShoppingCart.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(shoppingCart, items);
        // Act
        Item result = shoppingCart.getItem("item123");
        // Assert
        assertNotNull(result);
        assertEquals("item123", result.getItemId());
    }

    @Test
    public void testGetItem_ItemNotFound() throws Exception {
        // Arrange
        ArrayList<Item> itemsList = new ArrayList<>();
        when(items.getItemsArrayList()).thenReturn(itemsList);
        // Set the 'items' field in ShoppingCart using reflection
        Field itemsField = ShoppingCart.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(shoppingCart, items);
        // Act
        Item result = shoppingCart.getItem("nonexistentItem");
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetItem_ItemsListNull() throws Exception {
        // Arrange
        when(items.getItemsArrayList()).thenReturn(null);
        // Set the 'items' field in ShoppingCart using reflection
        Field itemsField = ShoppingCart.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(shoppingCart, items);
        // Act
        Item result = shoppingCart.getItem("item123");
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetItem_ItemsListEmpty() throws Exception {
        // Arrange
        ArrayList<Item> itemsList = new ArrayList<>();
        when(items.getItemsArrayList()).thenReturn(itemsList);
        // Set the 'items' field in ShoppingCart using reflection
        Field itemsField = ShoppingCart.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(shoppingCart, items);
        // Act
        Item result = shoppingCart.getItem("item123");
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetItem_ItemsListWithNullItem() throws Exception {
        // Arrange
        Item mockItem = new Item();
        mockItem.setItemId("item123");
        mockItem.setQuantity("1");
        mockItem.setOurPrice("10.00");
        ArrayList<Item> itemsList = new ArrayList<>();
        itemsList.add(null);
        itemsList.add(mockItem);
        when(items.getItemsArrayList()).thenReturn(itemsList);
        // Set the 'items' field in ShoppingCart using reflection
        Field itemsField = ShoppingCart.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(shoppingCart, items);
        // Act
        Item result = shoppingCart.getItem("item123");
        // Assert
        assertNotNull(result);
        assertEquals("item123", result.getItemId());
    }

    @Test
    public void testGetItem_ItemsListWithWhitespaceAndCaseInsensitive() throws Exception {
        // Arrange
        Item mockItem = new Item();
        mockItem.setItemId("  Item123  ");
        mockItem.setQuantity("1");
        mockItem.setOurPrice("10.00");
        ArrayList<Item> itemsList = new ArrayList<>();
        itemsList.add(mockItem);
        when(items.getItemsArrayList()).thenReturn(itemsList);
        // Set the 'items' field in ShoppingCart using reflection
        Field itemsField = ShoppingCart.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(shoppingCart, items);
        // Act
        Item result = shoppingCart.getItem("item123");
        // Assert
        assertNotNull(result);
        assertEquals("  Item123  ", result.getItemId());
    }
}
