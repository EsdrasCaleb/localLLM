package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_toString_4_0_Test {

    @Test
    void testToString_NullList() {
        SimilarProducts sp = new SimilarProducts();
        assertEquals("Similar Products is null or size 0\n", sp.toString());
    }

    @Test
    void testToString_EmptyList() {
        SimilarProducts sp = new SimilarProducts();
        try {
            Field simProductsField = SimilarProducts.class.getDeclaredField("simProducts");
            simProductsField.setAccessible(true);
            simProductsField.set(sp, new ArrayList<>());
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Failed to access simProducts field: " + e.getMessage());
        }
        assertEquals("Similar Products is null or size 0\n", sp.toString());
    }

    @Test
    void testToString_NonEmptyList() {
        SimilarProducts sp = new SimilarProducts();
        String[] products = { "Product A", "Product B", "Product C" };
        sp.setProduct(products);
        String expected = "# of Simular products = 3\n" + "ProductAction - Product A\n" + "ProductAction - Product B\n" + "ProductAction - Product C\n";
        assertEquals(expected, sp.toString());
    }

    @Test
    void testToString_ListWithEmptyProduct() {
        SimilarProducts sp = new SimilarProducts();
        String[] products = { "Product A", "", "Product C" };
        sp.setProduct(products);
        String expected = "# of Simular products = 3\n" + "ProductAction - Product A\n" + "ProductAction - \n" + "ProductAction - Product C\n";
        assertEquals(expected, sp.toString());
    }
}
