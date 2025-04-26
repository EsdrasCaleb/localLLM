package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;
import java.util.ArrayList;

class Items_toString_3_0_Test {

    private Items items;

    @BeforeEach
    void setUp() {
        items = new Items();
    }

    @Test
    void testToStringWithEmptyItems() {
        // Given
        items.setItem(new Item[0]);
        // When
        String result = items.toString();
        // Then
        assertEquals("", result);
    }

    @Test
    void testToStringWithNullItems() {
        // Given
        items.setItem(new Item[] { null });
        // When
        String result = items.toString();
        // Then
        assertEquals("", result);
    }

    @Test
    void testToStringWithNonNullItems() {
        // Given
        Item item1 = new Item("Item1");
        Item item2 = new Item("Item2");
        items.setItem(new Item[] { item1, item2 });
        // When
        String result = items.toString();
        // Then
        assertEquals("Item1Item2", result);
    }

    @Test
    void testToStringWithMixedItems() {
        // Given
        Item item1 = new Item("Item1");
        Item item2 = null;
        Item item3 = new Item("Item3");
        items.setItem(new Item[] { item1, item2, item3 });
        // When
        String result = items.toString();
        // Then
        assertEquals("Item1Item3", result);
    }
}

class Item {

    private String name;

    public Item(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name != null ? name : "";
    }
}

class Items {

    private Item[] items;

    public void setItem(Item[] items) {
        this.items = items;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        if (items != null) {
            for (Item item : items) {
                if (item != null) {
                    result.append(item.toString());
                }
            }
        }
        return result.toString();
    }
}
