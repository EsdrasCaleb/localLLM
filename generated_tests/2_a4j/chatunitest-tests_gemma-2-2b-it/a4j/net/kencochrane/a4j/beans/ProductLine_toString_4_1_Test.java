package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_1_Test {

    @Test
    void testToString() {
        // Create a mock ProductInfo object
        ProductInfo productInfo = mock(ProductInfo.class);
        // Create a ProductLine object
        ProductLine productLine = new ProductLine();
        productLine.setMode("A");
        productLine.setProductInfo(productInfo);
        // Call the toString() method
        String result = productLine.toString();
        // Assert the expected output
        assertEquals("Mode = A\nProductInfo: [ProductInfo object]", result);
    }
}
