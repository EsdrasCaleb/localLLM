package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_getProduct_3_1_Test {

    @Test
    public void testGetProduct() {
        SimilarProducts similarProducts = new SimilarProducts();
        String[] testProducts = { "product1", "product2", "product3" };
        similarProducts.setProduct(testProducts);
        assertEquals("product1", similarProducts.getProduct(0));
        assertEquals("product2", similarProducts.getProduct(1));
        assertEquals("product3", similarProducts.getProduct(2));
    }
}
