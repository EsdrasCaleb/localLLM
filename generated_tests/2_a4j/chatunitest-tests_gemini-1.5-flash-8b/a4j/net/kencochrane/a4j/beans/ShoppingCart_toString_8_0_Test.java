package net.kencochrane.a4j.beans;

import java.util.ArrayList;
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

// Assuming these classes exist and have the appropriate methods
class ShoppingCart_toString_8_0_Test {

    @Test
    public void testToString_emptyItems() {
        Items items = Mockito.mock(Items.class);
        Mockito.when(items.getItemsArrayList()).thenReturn(new ArrayList<>());
        ShoppingCart cart = new ShoppingCart();
        cart.setCartId("123");
        cart.setHMAC("abc");
        cart.setPurchaseURL("https://example.com");
        cart.setItems(items);
        String expectedOutput = "HMAC = abc\nPurchase URL = https://example.com\nCartId = 123\nitems = []\n";
        String actualOutput = cart.toString();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    public void testToString_nonEmptyItems() {
        Items items = new Items();
        ArrayList<Item> itemList = new ArrayList<>();
        Item item1 = new Item();
        item1.setOurPrice("10.00");
        item1.setQuantity("2");
        itemList.add(item1);
        // Correctly set the items
        items.setItem(itemList.toArray(new Item[0]));
        ShoppingCart cart = new ShoppingCart();
        cart.setCartId("456");
        cart.setHMAC("def");
        cart.setPurchaseURL("https://another.com");
        cart.setItems(items);
        String expectedOutput = "HMAC = def\nPurchase URL = https://another.com\nCartId = 456\nitems = [Item{ourPrice='10.00', quantity='2'}]\n";
        String actualOutput = cart.toString();
        assertEquals(expectedOutput, actualOutput);
    }
}
