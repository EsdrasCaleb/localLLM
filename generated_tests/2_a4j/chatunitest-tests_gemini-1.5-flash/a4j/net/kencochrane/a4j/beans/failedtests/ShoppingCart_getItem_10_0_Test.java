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

class ShoppingCart_getItem_10_0_Test {

    @Test
    void testGetItem_itemExists() {
        ShoppingCart cart = new ShoppingCart();
        Item item1 = Mockito.mock(Item.class);
        Mockito.when(item1.getItemId()).thenReturn("123");
        Item item2 = Mockito.mock(Item.class);
        Mockito.when(item2.getItemId()).thenReturn("456");
        ArrayList<Item> itemsList = new ArrayList<>();
        itemsList.add(item1);
        itemsList.add(item2);
        Items items = Mockito.mock(Items.class);
        Mockito.when(items.getItemsArrayList()).thenReturn(itemsList);
        try {
            Field itemsField = ShoppingCart.class.getDeclaredField("items");
            itemsField.setAccessible(true);
            itemsField.set(cart, items);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set items field: " + e.getMessage());
        }
        assertEquals(item1, cart.getItem("123"));
        assertEquals(item2, cart.getItem("456"));
    }

    @Test
    void testGetItem_itemDoesNotExist() {
        ShoppingCart cart = new ShoppingCart();
        assertNull(cart.getItem("789"));
        Item item1 = Mockito.mock(Item.class);
        Mockito.when(item1.getItemId()).thenReturn("123");
        ArrayList<Item> itemsList = new ArrayList<>();
        itemsList.add(item1);
        Items items = Mockito.mock(Items.class);
        Mockito.when(items.getItemsArrayList()).thenReturn(itemsList);
        try {
            Field itemsField = ShoppingCart.class.getDeclaredField("items");
            itemsField.setAccessible(true);
            itemsField.set(cart, items);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set items field: " + e.getMessage());
        }
        assertNull(cart.getItem("789"));
    }

    @Test
    void testGetItem_emptyList() {
        ShoppingCart cart = new ShoppingCart();
        assertNull(cart.getItem("789"));
        Items items = Mockito.mock(Items.class);
        Mockito.when(items.getItemsArrayList()).thenReturn(new ArrayList<>());
        try {
            Field itemsField = ShoppingCart.class.getDeclaredField("items");
            itemsField.setAccessible(true);
            itemsField.set(cart, items);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to set items field: " + e.getMessage());
        }
        assertNull(cart.getItem("789"));
    }

    @Test
    void testGetItem_nullItems() {
        ShoppingCart cart = new ShoppingCart();
        assertNull(cart.getItem("789"));
    }

    // Dummy classes for compilation
    static class Item {

        public String getItemId() {
            return null;
        }

        public String getOurPrice() {
            return null;
        }

        public String getQuantity() {
            return null;
        }
    }

    static class Items {

        public ArrayList getItemsArrayList() {
            return null;
        }
    }

    static class a4jUtil {

        public BigDecimal getPrice(String itemPrice) {
            return new BigDecimal(0.00);
        }
    }
}
