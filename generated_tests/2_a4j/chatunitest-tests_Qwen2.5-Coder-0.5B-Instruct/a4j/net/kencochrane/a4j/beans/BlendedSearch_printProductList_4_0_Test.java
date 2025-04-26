package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class BlendedSearch_printProductList_4_0_Test {

    @Test
    public void testPrintProductList() {
        // Create a mock object of ProductLine
        ProductLine mockProductLine = Mockito.mock(ProductLine.class);
        // Set up the mock object to return a specific string
        when(mockProductLine.printProductList()).thenReturn("Product 1\nProduct 2\nProduct 3");
        // Create an instance of BlendedSearch with the mock object
        BlendedSearch testObject = new BlendedSearch();
        // Call the method to be tested
        String result = testObject.printProductList();
        // Verify that the method returned the expected string
        assertEquals("Product 1\nProduct 2\nProduct 3", result);
    }
}
