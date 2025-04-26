package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_getProduct_3_0_Test {

    @Mock
    private SimilarProducts similarProducts;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testGetProductWithinBounds() {
        ArrayList<String> products = new ArrayList<>();
        products.add("Product 1");
        products.add("Product 2");
        products.add("Product 3");
        when(similarProducts.getProductsArray()).thenReturn(products);
        String result = similarProducts.getProduct(1);
        assertEquals("Product 2", result);
    }

    @Test
    public void testGetProductOutsideBounds() {
        ArrayList<String> products = new ArrayList<>();
        products.add("Product 1");
        products.add("Product 2");
        products.add("Product 3");
        when(similarProducts.getProductsArray()).thenReturn(products);
        String result = similarProducts.getProduct(5);
        assertEquals(null, result);
    }
}
