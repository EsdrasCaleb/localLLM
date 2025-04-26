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

class Item_toString_18_2_Test {

    @Test
    void testToString() {
        Item item = new Item();
        item.setAsin("TestAsin");
        item.setProductName("TestProduct");
        item.setQuantity("TestQuantity");
        String expected = "Asin = TestAsin\nName = TestProduct\nquantity = TestQuantity\n";
        assertEquals(expected, item.toString());
    }
}
