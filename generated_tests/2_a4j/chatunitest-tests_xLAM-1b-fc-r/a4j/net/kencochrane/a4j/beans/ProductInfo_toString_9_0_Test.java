package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ProductInfo_toString_9_0_Test {

    @Test
    public void testToString() throws Exception {
        ProductInfo productInfo = new ProductInfo();
        // Set the fields
        Field field = ProductInfo.class.getDeclaredField("listName");
        field.setAccessible(true);
        field.set(productInfo, "testList");
        Field field2 = ProductInfo.class.getDeclaredField("totalResults");
        field2.setAccessible(true);
        field2.set(productInfo, "10");
        Field field3 = ProductInfo.class.getDeclaredField("totalPages");
        field3.setAccessible(true);
        field3.set(productInfo, "20");
        Field field4 = ProductInfo.class.getDeclaredField("products");
        field4.setAccessible(true);
        field4.set(productInfo, new ArrayList());
        // Call the method
        String result = productInfo.toString();
        // Check the result
        assertEquals("Total results = 10\nTotal pages = 20\n# of products = 0\n", result);
    }
}
