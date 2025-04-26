package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

class SimilarProducts_getProduct_3_0_Test {

    @Test
    public void testGetProduct() {
        SimilarProducts similarProducts = new SimilarProducts();
        similarProducts.setProduct(new String[] { "Product A", "Product B", "Product C" });
        String product = similarProducts.getProduct(1);
        assertEquals("Product B", product);
    }
}
