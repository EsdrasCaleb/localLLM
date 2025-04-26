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

@ExtendWith(MockitoExtension.class)
public class Item_toString_18_0_Test {

    // Test class
    @Test
    public void testToString() {
        Item item = new Item();
        item.setAsin("123456789");
        item.setProductName("iPhone 14");
        item.setQuantity("1");
        item.setListPrice("1000");
        item.setOurPrice("900");
        item.setMerchantSku("1234567890");
        assertEquals("Asin = 123456789\nName = iPhone 14\nQty = 1\nPrice = 900\n", item.toString());
    }
}
