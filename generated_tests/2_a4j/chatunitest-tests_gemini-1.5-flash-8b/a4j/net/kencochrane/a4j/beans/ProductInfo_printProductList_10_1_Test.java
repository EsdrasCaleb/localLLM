package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class ProductInfo_printProductList_10_1_Test {

    private ProductInfo productInfo;

    private ArrayList<ProductDetails> mockProducts;

    private ProductDetails mockProduct1, mockProduct2;

    @BeforeEach
    void setUp() {
        productInfo = new ProductInfo();
        mockProducts = new ArrayList<>();
        mockProduct1 = Mockito.mock(ProductDetails.class);
        mockProduct2 = Mockito.mock(ProductDetails.class);
        Mockito.when(mockProduct1.getAsin()).thenReturn("ASIN1");
        Mockito.when(mockProduct1.getProductName()).thenReturn("Product 1");
        Mockito.when(mockProduct1.getOurPrice()).thenReturn("10.0");
        Mockito.when(mockProduct2.getAsin()).thenReturn("ASIN2");
        Mockito.when(mockProduct2.getProductName()).thenReturn("Product 2");
        Mockito.when(mockProduct2.getOurPrice()).thenReturn("20.0");
        mockProducts.add(mockProduct1);
        mockProducts.add(mockProduct2);
        try {
            java.lang.reflect.Field productsField = ProductInfo.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(productInfo, mockProducts);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        productInfo.setTotalResults("10");
        productInfo.setTotalPages("2");
    }

    @Test
    void printProductList_withProducts() {
        String expectedOutput = "Total results = 10\nTotal pages = 2\n< 0 > ASIN1 : Product 1 - 10.0\n< 1 > ASIN2 : Product 2 - 20.0\n# of products = 2\n";
        String actualOutput = productInfo.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void printProductList_withNullProducts() {
        try {
            java.lang.reflect.Field productsField = ProductInfo.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(productInfo, null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        String expectedOutput = "Total results = 10\nTotal pages = 2\nproducts is null \n";
        String actualOutput = productInfo.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }

    @Test
    void printProductList_withEmptyProducts() {
        try {
            java.lang.reflect.Field productsField = ProductInfo.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(productInfo, new ArrayList<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        String expectedOutput = "Total results = 10\nTotal pages = 2\n# of products = 0\n";
        String actualOutput = productInfo.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
