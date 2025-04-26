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

class ShoppingCart_toString_8_0_Test {

    @Test
    void testToString_emptyCart() {
        ShoppingCart cart = new ShoppingCart();
        String expected = "HMAC = null\nPurchase URL = null\nCartId = null\nitems = null\n";
        assertEquals(expected, cart.toString());
    }

    @Test
    void testToString_nullItems() {
        ShoppingCart cart = new ShoppingCart();
        cart.setHMAC("testHMAC");
        cart.setPurchaseURL("testURL");
        cart.setCartId("testCartId");
        String expected = "HMAC = testHMAC\nPurchase URL = testURL\nCartId = testCartId\nitems = null\n";
        assertEquals(expected, cart.toString());
    }

    @Test
    void testToString_nullAllFields() {
        ShoppingCart cart = new ShoppingCart();
        String expected = "HMAC = null\nPurchase URL = null\nCartId = null\nitems = null\n";
        assertEquals(expected, cart.toString());
    }

    static class Items {

        public ArrayList<Item> getItemsArrayList() {
            return null;
        }

        @Override
        public String toString() {
            return "mockedItems";
        }
    }

    static class Item {

        public String getOurPrice() {
            return null;
        }

        public String getQuantity() {
            return null;
        }
    }

    static class a4jUtil {

        public BigDecimal getPrice(String itemPrice) {
            return new BigDecimal(itemPrice);
        }
    }
}
