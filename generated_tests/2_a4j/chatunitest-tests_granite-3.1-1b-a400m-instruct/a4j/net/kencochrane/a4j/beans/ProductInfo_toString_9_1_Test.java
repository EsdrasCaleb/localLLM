package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class ProductInfo_toString_9_1_Test {

    @Test
    void testToString() {
        // Setup: Create a ProductInfo object
        ProductInfo productInfo = new ProductInfo();
        productInfo.setListName("Example List");
        productInfo.setDetails(new ProductDetails[] { new ProductDetails() });
        productInfo.setTotalResults("100");
        productInfo.setTotalPages("2");
        // Act: Call toString() method
        String result = productInfo.toString();
        // Assert: The result should be a string with the expected format
        assertEquals("Total results = 100\nTotal pages = 2\nproducts is null\n", result);
    }
}
