package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
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
    void testToString() {
        Item item = new Item();
        item.setAsin("123456");
        item.setProductName("Test Product");
        item.setQuantity("10");
        item.setListPrice("10.00");
        item.setOurPrice("10.50");
        item.setMerchantSku("12345");
        String expected = "Asin = 123456\nName = Test Product\nquantity = 10\n";
        assertEquals(expected, item.toString());
    }
}
