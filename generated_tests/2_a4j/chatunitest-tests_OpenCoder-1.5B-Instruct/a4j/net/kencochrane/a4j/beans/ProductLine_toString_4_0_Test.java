package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductLine_toString_4_0_Test {

    @Test
    public void testToString() throws Exception {
        // Create a mock object of ProductLine
        ProductLine productLine = Mockito.mock(ProductLine.class);
        // Set the behavior of the mock object
        Mockito.when(productLine.getMode()).thenReturn("Mode1");
        Mockito.when(productLine.getProductInfo()).thenReturn(new ProductInfo());
        // Get the toString method
        Method toStringMethod = ProductLine.class.getDeclaredMethod("toString");
        // Invoke the toString method and get the result
        String result = (String) toStringMethod.invoke(productLine);
        // Expected output
        String expectedOutput = "Mode = Mode1\nProductInfo = null";
        // Assert the result
        assertEquals(expectedOutput, result);
    }
}
