package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import // org.apache.log4j.Logger
net.kencochrane.a4j.util.a4jUtil;
import java.io.Serializable;

public class Item_toString_18_0_Test {

    @Test
    void testToString_allFieldsSet() {
        Item item = new Item();
        item.setAsin("B012345678");
        item.setProductName("Test Product");
        item.setQuantity("10");
        String expected = "Asin = B012345678\nName = Test Product\nquantity = 10\n";
        assertEquals(expected, item.toString());
    }

    @Test
    void testToString_nullAsin() {
        Item item = new Item();
        item.setAsin(null);
        item.setProductName("Test Product");
        item.setQuantity("10");
        String expected = "Asin = null\nName = Test Product\nquantity = 10\n";
        assertEquals(expected, item.toString());
    }

    @Test
    void testToString_nullProductName() {
        Item item = new Item();
        item.setAsin("B012345678");
        item.setProductName(null);
        item.setQuantity("10");
        String expected = "Asin = B012345678\nName = null\nquantity = 10\n";
        assertEquals(expected, item.toString());
    }

    @Test
    void testToString_nullQuantity() {
        Item item = new Item();
        item.setAsin("B012345678");
        item.setProductName("Test Product");
        item.setQuantity(null);
        String expected = "Asin = B012345678\nName = Test Product\nquantity = null\n";
        assertEquals(expected, item.toString());
    }

    @Test
    void testToString_allFieldsNull() {
        Item item = new Item();
        String expected = "Asin = null\nName = null\nquantity = null\n";
        assertEquals(expected, item.toString());
    }

    @Test
    void testToString_emptyFields() {
        Item item = new Item();
        item.setAsin("");
        item.setProductName("");
        item.setQuantity("");
        String expected = "Asin = \nName = \nquantity = \n";
        assertEquals(expected, item.toString());
    }
}
