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

    private ProductInfo productInfo;

    @BeforeEach
    public void setUp() throws Exception {
        // Create an instance of ProductInfo using Mockito
        productInfo = Mockito.mock(ProductInfo.class);
        // Set up mock behavior for the toString() method
        Mockito.when(productInfo.getTotalResults()).thenReturn("100");
        Mockito.when(productInfo.getTotalPages()).thenReturn("20");
        Mockito.when(productInfo.getListName()).thenReturn("Latest Products");
        Mockito.when(productInfo.getProductsArrayList()).thenReturn(new ArrayList<>());
        // Additional setup if needed
    }

    @Test
    public void testToString() {
        // Call the toString() method on the mocked ProductInfo instance
        String result = productInfo.toString();
        // Expected output based on the mock behavior
        String expectedOutput = "Total results = 100\n" + "Total pages = 20\n" + "List name = Latest Products\n" + "# of products = 0\n";
        // Assert that the actual output matches the expected output
        assertEquals(expectedOutput, result);
    }
}
