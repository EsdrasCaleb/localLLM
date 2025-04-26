package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductInfo_toString_9_1_Test {

    private ProductInfo productInfo;

    private ProductDetails productDetails1;

    private ProductDetails productDetails2;

    @BeforeEach
    void setUp() {
        productInfo = new ProductInfo();
        productDetails1 = Mockito.mock(ProductDetails.class);
        productDetails2 = Mockito.mock(ProductDetails.class);
        ArrayList<ProductDetails> productsList = new ArrayList<>();
        productsList.add(productDetails1);
        productsList.add(productDetails2);
        Mockito.when(productDetails1.toString()).thenReturn("Product 1 Details");
        Mockito.when(productDetails2.toString()).thenReturn("Product 2 Details");
        productInfo.setProductsArrayList(productsList);
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("5");
    }

    @Test
    void testToStringWithProducts() {
        String expectedOutput = "Total results = 10\n" + "Total pages = 5\n" + "Product 1 Details\n" + "Product 2 Details\n" + "# of products = 2\n";
        assertEquals(expectedOutput, productInfo.toString());
    }

    @Test
    void testToStringWithNullProducts() {
        ProductInfo productInfoNull = new ProductInfo();
        String expectedOutput = "Total results = null\n" + "Total pages = null\n" + "products is null \n";
        assertEquals(expectedOutput, productInfoNull.toString());
    }

    @Test
    void testToStringEmptyProducts() {
        ProductInfo productInfoEmpty = new ProductInfo();
        ArrayList<ProductDetails> emptyList = new ArrayList<>();
        productInfoEmpty.setProductsArrayList(emptyList);
        productInfoEmpty.setTotalResults("0");
        productInfoEmpty.setTotalPages("0");
        String expectedOutput = "Total results = 0\n" + "Total pages = 0\n" + "products is null \n";
        assertEquals(expectedOutput, productInfoEmpty.toString());
    }

    // Add more tests for different scenarios, like null totalResults, null totalPages, etc.
    // Helper class (replace with your actual ProductDetails class)
    static class ProductDetails {

        @Override
        public String toString() {
            return "Product Details";
        }
    }

    // Add setters and getters for ProductInfo class
    static class ProductInfo {

        ArrayList products;

        ProductDetails details;

        String totalResults, totalPages, listName;

        public ArrayList getProductsArrayList() {
            return products;
        }

        public void setProductsArrayList(ArrayList products) {
            this.products = products;
        }

        public String getTotalResults() {
            return totalResults;
        }

        public void setTotalResults(String totalResults) {
            this.totalResults = totalResults;
        }

        public String getTotalPages() {
            return totalPages;
        }

        public void setTotalPages(String totalPages) {
            this.totalPages = totalPages;
        }

        // ... other methods
        @Override
        public String toString() {
            // ... (implementation from the question)
            return super.toString();
        }
    }
}
