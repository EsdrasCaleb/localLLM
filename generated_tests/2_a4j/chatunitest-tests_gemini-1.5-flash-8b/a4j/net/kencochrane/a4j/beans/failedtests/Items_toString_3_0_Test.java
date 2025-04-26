package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
java.io.Serializable;

public class Items_toString_3_0_Test {

    @Test
    public void testToString_emptyArrayList() throws NoSuchFieldException, IllegalAccessException {
        Items items = new Items();
        String result = items.toString();
        assertEquals("", result);
    }

    @Test
    public void testToString_nullArrayList() throws NoSuchFieldException, IllegalAccessException {
        Items items = new Items();
        Field itemsField = Items.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(items, null);
        String result = items.toString();
        assertEquals("", result);
    }

    @Test
    public void testToString_nonEmptyArrayList() throws NoSuchFieldException, IllegalAccessException {
        Items items = new Items();
        ArrayList<Item> itemList = new ArrayList<>();
        itemList.add(new Item("Product 1", 10.0));
        itemList.add(new Item("Product 2", 20.0));
        // Test for null element
        itemList.add(null);
        itemList.add(new Item("Product 3", 30.0));
        Field itemsField = Items.class.getDeclaredField("items");
        itemsField.setAccessible(true);
        itemsField.set(items, itemList);
        String expected = "Product 1, price: 10.0Product 2, price: 20.0Product 3, price: 30.0";
        String result = items.toString();
        assertEquals(expected, result);
    }

    // Helper class for testing
    static class Item {

        String name;

        double price;

        public Item(String name, double price) {
            this.name = name;
            this.price = price;
        }

        public Item() {
        }

        @Override
        public String toString() {
            return name + ", price: " + price;
        }
    }
}
