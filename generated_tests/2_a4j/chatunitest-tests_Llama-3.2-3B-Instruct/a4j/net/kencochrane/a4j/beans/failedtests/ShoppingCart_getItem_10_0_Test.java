package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;
import java.math.BigDecimal;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class ShoppingCart_getItem_10_0_Test {

    @Mock
    private Items items;

    @InjectMocks
    private ShoppingCart shoppingCart;

    @Test
    public void testGetItem_WhenItemExists_ReturnsItem() {
        // Arrange
        String itemId = "12345";
        Item item = new Item();
        item.setItemId(itemId);
        when(items.getItemsArrayList().get(0)).thenReturn(item);
        // Act
        Item result = shoppingCart.getItem(itemId);
        // Assert
        assertEquals(item, result);
    }

    @Test
    public void testGetItem_WhenItemDoesNotExist_ReturnsNull() {
        // Arrange
        String itemId = "12345";
        when(items.getItemsArrayList().get(0)).thenReturn(null);
        // Act
        Item result = shoppingCart.getItem(itemId);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetItem_WhenItemsListIsEmpty_ReturnsNull() {
        // Arrange
        String itemId = "12345";
        when(items.getItemsArrayList()).thenReturn(new ArrayList<>());
        // Act
        Item result = shoppingCart.getItem(itemId);
        // Assert
        assertNull(result);
    }
}
