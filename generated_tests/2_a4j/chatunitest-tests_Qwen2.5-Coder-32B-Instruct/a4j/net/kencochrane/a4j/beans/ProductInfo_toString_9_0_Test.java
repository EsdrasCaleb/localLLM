package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class ProductInfo_toString_9_0_Test {

    @InjectMocks
    private ProductInfo productInfo;

    @Mock
    private ProductDetails productDetails;

    @BeforeEach
    void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the ArrayList and ProductDetails mock object
        ArrayList<ProductDetails> mockProducts = new ArrayList<>();
        mockProducts.add(productDetails);
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, mockProducts);
        Field detailsField = ProductInfo.class.getDeclaredField("details");
        detailsField.setAccessible(true);
        detailsField.set(productInfo, productDetails);
        // Set values for totalResults, totalPages, and listName
        Field totalResultsField = ProductInfo.class.getDeclaredField("totalResults");
        totalResultsField.setAccessible(true);
        totalResultsField.set(productInfo, "100");
        Field totalPagesField = ProductInfo.class.getDeclaredField("totalPages");
        totalPagesField.setAccessible(true);
        totalPagesField.set(productInfo, "5");
        Field listNameField = ProductInfo.class.getDeclaredField("listName");
        listNameField.setAccessible(true);
        listNameField.set(productInfo, "Best Products");
        // Mock the toString method of ProductDetails
        when(productDetails.toString()).thenReturn("ProductDetails{...}");
    }

    @Test
    void testToStringWithProducts() {
        String expectedOutput = "Total results = 100\n" + "Total pages = 5\n" + "ProductDetails{...}\n" + "# of products = 1\n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    void testToStringWithNullProducts() throws Exception {
        // Set products to null
        Field productsField = ProductInfo.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(productInfo, null);
        String expectedOutput = "Total results = 100\n" + "Total pages = 5\n" + "products is null \n";
        assertEquals(expectedOutput, productInfo.toString());
    }
}
