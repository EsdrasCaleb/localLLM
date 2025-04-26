package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;

public class Items_toString_3_1_Test {

    @Test
    void testToString() {
        // Setup
        Items items = new Items();
        List<Item> itemsList = new ArrayList<>();
        itemsList.add(new Item());
        itemsList.add(new Item());
        items.setItem(itemsList.toArray(new Item[0]));
        // Act
        String result = items.toString();
        // Assert
        assertEquals("Item1\nItem2", result);
        assertTrue(items.getItemsArrayList().isEmpty());
    }

    @Test
    void testToStringEmptyItems() {
        // Setup
        Items items = new Items();
        items.setItem(new Item[0]);
        // Act
        String result = items.toString();
        // Assert
        assertEquals("", result);
        assertTrue(items.getItemsArrayList().isEmpty());
    }

    @Test
    void testToStringNullItems() {
        // Setup
        Items items = new Items();
        items.setItem(new Item[0]);
        // Act
        String result = items.toString();
        // Assert
        assertEquals("", result);
        assertTrue(items.getItemsArrayList().isEmpty());
    }

    @Test
    void testToStringWithNullItems() {
        // Setup
        Items items = new Items();
        items.setItem(new Item[0]);
        // Act
        String result = items.toString();
        // Assert
        assertEquals("", result);
        assertTrue(items.getItemsArrayList().isEmpty());
    }
}
