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
