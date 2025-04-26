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
    public void testToString_NullItemsArray() {
        Items items = new Items();
        Item[] itemsArray = null;
        assertThrows(NullPointerException.class, () -> items.setItem(itemsArray));
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
