package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_toString_4_0_Test {

    private SimilarProducts similarProducts;

    @BeforeEach
    public void setUp() {
        similarProducts = new SimilarProducts();
    }

    @Test
    public void testToString_WithProducts() {
        String[] products = { "Product1", "Product2", "Product3" };
        similarProducts.setProduct(products);
        String expectedOutput = "# of Simular products = 3\n" + "ProductAction - Product1\n" + "ProductAction - Product2\n" + "ProductAction - Product3\n";
        assertEquals(expectedOutput, similarProducts.toString());
    }

    @Test
    public void testToString_EmptyProducts() {
        String[] products = {};
        similarProducts.setProduct(products);
        String expectedOutput = "Similar Products is null or size 0\n";
        assertEquals(expectedOutput, similarProducts.toString());
    }

    @Test
    public void testToString_NullProducts() {
        similarProducts.setProduct(null);
        String expectedOutput = "Similar Products is null or size 0\n";
        assertEquals(expectedOutput, similarProducts.toString());
    }

    @Test
    public void testToString_WithNullProduct() {
        String[] products = { "Product1", null, "Product3" };
        similarProducts.setProduct(products);
        String expectedOutput = "# of Simular products = 3\n" + "ProductAction - Product1\n" + "ProductAction - null\n" + "ProductAction - Product3\n";
        assertEquals(expectedOutput, similarProducts.toString());
    }
}
