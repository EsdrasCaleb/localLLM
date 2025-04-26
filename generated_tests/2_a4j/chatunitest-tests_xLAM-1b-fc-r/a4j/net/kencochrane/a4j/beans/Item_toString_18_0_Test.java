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

class Item_toString_18_0_Test {

    @Test
    void testToString() {
        Item item = new Item();
        item.setAsin("12345");
        item.setProductName("Product Name");
        item.setQuantity("10");
        item.setListPrice("123.45");
        item.setOurPrice("123.45");
        String expected = "Asin = 12345\nName = Product Name\nquantity = 10\n";
        String actual = item.toString();
        assertEquals(expected, actual);
    }
}
