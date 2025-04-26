package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductLine_toString_4_2_Test {

    @Test
    public void testToString() {
        // Create an instance of ProductLine
        ProductLine productLine = mock(ProductLine.class);
        // Set the mode and productInfo fields
        when(productLine.getMode()).thenReturn("Light");
        when(productLine.getProductInfo()).thenReturn(new ProductInfo());
        // Call the toString method
        String result = productLine.toString();
        // Verify that the toString method returns the expected output
        assertEquals("Mode = Light\nProductInfo = \n", result);
    }
}
