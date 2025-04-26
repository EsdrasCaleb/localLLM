package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class ProductInfo_toString_9_1_Test {

    private ProductInfo productInfo;

    @BeforeEach
    public void setup() {
        productInfo = new ProductInfo();
    }

    @Test
    public void testToString() {
        // Create mock ProductDetails objects
        ProductDetails product1 = Mockito.mock(ProductDetails.class);
        ProductDetails product2 = Mockito.mock(ProductDetails.class);
        // Mock the toString() method to return a specific string for each product
        when(product1.toString()).thenReturn("Product 1");
        when(product2.toString()).thenReturn("Product 2");
        // Create an ArrayList of mock ProductDetails objects
        ArrayList<ProductDetails> products = new ArrayList<>();
        products.add(product1);
        products.add(product2);
        // Use reflection to invoke the private setProductsArrayList() method
        try {
            productInfo.getClass().getDeclaredMethod("setProductsArrayList", ArrayList.class).setAccessible(true);
            productInfo.getClass().getDeclaredMethod("setProductsArrayList", ArrayList.class).invoke(productInfo, products);
        } catch (Exception e) {
            e.printStackTrace();
        }
        // Set the total results and total pages in the ProductInfo object
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
        // Call the toString() method and store the result
        String result = productInfo.toString();
        // Define the expected result
        String expectedResult = "Total results = 10\n" + "Total pages = 2\n" + "Product 1\n" + "Product 2\n" + "# of products = 2\n";
        // Assert that the result is equal to the expected result
        assertEquals(expectedResult, result);
    }
}
