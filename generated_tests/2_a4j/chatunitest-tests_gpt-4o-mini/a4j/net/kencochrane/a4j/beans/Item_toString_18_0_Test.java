package net.kencochrane.a4j.beans;

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

    private Item item;

    @BeforeEach
    public void setUp() {
        item = new Item();
    }

    @Test
    public void testToString_WithAllFieldsSet() {
        item.setAsin("B001234567");
        item.setProductName("Test Product");
        item.setQuantity("10");
        String expected = "Asin = B001234567\nName = Test Product\nquantity = 10\n";
        assertEquals(expected, item.toString());
    }

    @Test
    public void testToString_WithNullFields() {
        item.setAsin(null);
        item.setProductName(null);
        item.setQuantity(null);
        String expected = "Asin = null\nName = null\nquantity = null\n";
        assertEquals(expected, item.toString());
    }

    @Test
    public void testToString_WithEmptyFields() {
        item.setAsin("");
        item.setProductName("");
        item.setQuantity("");
        String expected = "Asin = \nName = \nquantity = \n";
        assertEquals(expected, item.toString());
    }
}
