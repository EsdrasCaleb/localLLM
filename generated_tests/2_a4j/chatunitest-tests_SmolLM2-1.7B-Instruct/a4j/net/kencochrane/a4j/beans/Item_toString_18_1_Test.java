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

public class Item_toString_18_1_Test {

    @Test
    public void testToString() {
        Item item = new Item();
        item.setAsin("1234567890");
        item.setProductName("Product Name");
        item.setQuantity("10");
        item.setListPrice("10.99");
        item.setOurPrice("10.99");
        item.setMerchantSku("1234567890");
        String expected = "Asin = 1234567890\nName = Product Name\nquantity = 10\nList Price = 10.99\nOur Price = 10.99\nMerchant Sku = 1234567890";
        assertEquals(expected, item.toString());
    }
}
