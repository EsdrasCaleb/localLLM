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

public class Items_toString_3_1_Test {

    @Test
    public void testToString_EmptyList() {
        Items items = new Items();
        String result = items.toString();
        assertEquals("", result);
    }

    @Test
    public void testToString_NonEmptyList() {
        Items items = new Items();
        Item item1 = new Item("Item1");
        Item item2 = new Item("Item2");
        Item item3 = new Item("Item3");
        items.setItem(item1, item2, item3);
        String result = items.toString();
        assertEquals("Item1\nItem2\nItem3", result);
    }

    @Test
    public void testToString_NullItem() {
        Items items = new Items();
        Item item1 = new Item("Item1");
        items.setItem(item1, null, new Item("Item3"));
        String result = items.toString();
        assertEquals("Item1\n", result);
    }

    @Test
    public void testToString_NullItemsArray() {
        Items items = new Items();
        Item[] itemsArray = null;
        assertThrows(NullPointerException.class, () -> items.setItem(itemsArray));
    }

    @Test
    public void testToString_InvalidItem() {
        Items items = new Items();
        Item item = null;
        assertThrows(NullPointerException.class, () -> items.setItem(item));
    }
}

class Item implements java.io.Serializable {

    private String name;

    public Item(String name) {
        this.name = name;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}

class Items implements java.io.Serializable {

    private Item[] items;

    public Items() {
        this.items = new Item[0];
    }

    public void setItem(Item... items) {
        this.items = items;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (Item item : items) {
            sb.append(item.getName()).append("\n");
        }
        return sb.toString();
    }
}
