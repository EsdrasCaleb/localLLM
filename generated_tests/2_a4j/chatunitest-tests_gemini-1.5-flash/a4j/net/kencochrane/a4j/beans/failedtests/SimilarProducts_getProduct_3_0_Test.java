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

public class SimilarProducts_getProduct_3_0_Test {

    private SimilarProducts similarProducts;

    @BeforeEach
    void setUp() {
        similarProducts = new SimilarProducts();
    }

    @Test
    void testGetProductWithinBounds() {
        String[] products = { "Product A", "Product B", "Product C" };
        similarProducts.setProduct(products);
        assertEquals("Product B", similarProducts.getProduct(1));
    }

    @Test
    void testGetProductOutOfBounds() {
        String[] products = { "Product A", "Product B", "Product C" };
        similarProducts.setProduct(products);
        assertNull(similarProducts.getProduct(3));
        assertNull(similarProducts.getProduct(-1));
    }

    @Test
    void testGetProductEmptyList() {
        assertNull(similarProducts.getProduct(0));
    }

    @Test
    void testGetProductSingleElement() {
        String[] products = { "Product A" };
        similarProducts.setProduct(products);
        assertEquals("Product A", similarProducts.getProduct(0));
        assertNull(similarProducts.getProduct(1));
    }

    @Test
    void testProductArray() {
        String[] products = { "Product A", "Product B", "Product C" };
        similarProducts.setProduct(products);
        ArrayList<String> list = similarProducts.getProductsArray();
        assertEquals(3, list.size());
        assertEquals("Product A", list.get(0));
        assertEquals("Product B", list.get(1));
        assertEquals("Product C", list.get(2));
    }
}
