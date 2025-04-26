package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SimilarProducts_getProduct_3_0_Test {

    private SimilarProducts similarProducts;

    @BeforeEach
    void setUp() {
        similarProducts = new SimilarProducts();
    }

    @Test
    void testGetProduct_NegativeIndex() throws Exception {
        String[] products = { "Product1", "Product2", "Product3" };
        similarProducts.setProduct(products);
        // Invalid index
        String result = similarProducts.getProduct(-1);
        assertNull(result);
    }
}
