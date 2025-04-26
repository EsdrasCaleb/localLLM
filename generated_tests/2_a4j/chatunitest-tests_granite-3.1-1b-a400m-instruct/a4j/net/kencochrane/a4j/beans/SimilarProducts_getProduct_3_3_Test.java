package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SimilarProducts_getProduct_3_3_Test {

    @Test
    public void testGetProduct() {
        SimilarProducts simProducts = new SimilarProducts();
        simProducts.setProduct(new String[] { "Product A", "Product B" });
        simProducts.setProduct(new String[] { "Product C", "Product D" });
        simProducts.setProduct(new String[] { "Product E", "Product F" });
        assertEquals("Product C", simProducts.getProduct(1));
        assertEquals("Product D", simProducts.getProduct(2));
        assertEquals("Product E", simProducts.getProduct(3));
        assertEquals("Product F", simProducts.getProduct(4));
        // Out of bounds
        assertEquals(null, simProducts.getProduct(5));
    }
}
