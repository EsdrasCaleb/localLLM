package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class SimilarProducts_getProduct_3_1_Test {

    // Test class
    @Test
    public void testGetProduct() {
        SimilarProducts simProducts = new SimilarProducts();
        simProducts.setProduct(new String[] { "A", "B", "C", "D" });
        assertEquals("B", simProducts.getProduct(1));
    }
}
