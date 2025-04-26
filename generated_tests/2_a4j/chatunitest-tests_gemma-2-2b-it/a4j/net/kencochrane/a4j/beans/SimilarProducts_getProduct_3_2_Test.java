package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SimilarProducts_getProduct_3_2_Test {

    @Test
    void testGetProduct() {
        SimilarProducts similarProducts = new SimilarProducts();
        similarProducts.setProduct(new String[] { "Product 1", "Product 2", "Product 3" });
        String product = similarProducts.getProduct(1);
        assertEquals("Product 2", product);
    }
}
