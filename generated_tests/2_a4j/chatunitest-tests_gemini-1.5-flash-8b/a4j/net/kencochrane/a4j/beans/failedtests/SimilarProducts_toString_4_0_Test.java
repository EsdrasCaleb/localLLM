package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SimilarProducts_toString_4_0_Test {

    private SimilarProducts similarProducts;

    @BeforeEach
    void setUp() {
        similarProducts = new SimilarProducts();
    }

    @Test
    void testToStringEmptyList() {
        assertEquals("Similar Products is null or size 0\n", similarProducts.toString());
    }

    @Test
    void testToStringNullList() {
        SimilarProducts nullProducts = new SimilarProducts();
        nullProducts.setProduct(null);
        assertEquals("Similar Products is null or size 0\n", nullProducts.toString());
    }

    @Test
    void testToStringNonEmptyList() {
        String[] products = { "Product 1", "Product 2", "Product 3" };
        similarProducts.setProduct(products);
        String expected = "# of Simular products = 3\n" + "ProductAction - Product 1\n" + "ProductAction - Product 2\n" + "ProductAction - Product 3\n";
        assertEquals(expected, similarProducts.toString());
    }

    @Test
    void testToStringWithNullProduct() {
        String[] products = { "Product 1", null, "Product 3" };
        similarProducts.setProduct(products);
        String expected = "# of Simular products = 3\n" + "ProductAction - Product 1\n" + "ProductAction - null\n" + "ProductAction - Product 3\n";
        assertEquals(expected, similarProducts.toString());
    }
}
