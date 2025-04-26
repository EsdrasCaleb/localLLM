package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;

class Items_toString_3_1_Test {

    private static Items items;

    @BeforeAll
    static void setUp() {
        items = new Items();
    }

    @Test
    void testToString_EmptyList() {
        assertEquals("", items.toString());
    }

    @Test
    void testToString_NullList() {
        try {
            Field itemsField = Items.class.getDeclaredField("items");
            itemsField.setAccessible(true);
            itemsField.set(items, null);
            assertEquals("", items.toString());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access items field: " + e.getMessage());
        }
    }
}

class Items implements Serializable {

    private Item[] items;

    public void setItem(Item[] items) {
        this.items = items;
    }

    @Override
    public String toString() {
        if (items == null || items.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (Item item : items) {
            if (item != null) {
                sb.append(item.toString());
            }
        }
        return sb.toString();
    }
}

class Item implements Serializable {

    private String name;

    public Item() {
    }

    public Item(String name) {
        this.name = name;
    }

    @Override
    public String toString() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
