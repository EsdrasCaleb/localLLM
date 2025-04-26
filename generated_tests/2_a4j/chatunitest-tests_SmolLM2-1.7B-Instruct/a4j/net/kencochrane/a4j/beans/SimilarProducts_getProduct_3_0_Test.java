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
public class SimilarProducts_getProduct_3_0_Test {

    // <Buggy Line>: annotation type not applicable to this kind of declaration
    @Test
    public void testGetProduct() {
        SimilarProducts simProducts = new SimilarProducts();
        simProducts.setProduct(new String[] { "Product1", "Product2" });
        String product = simProducts.getProduct(0);
        assertEquals("Product1", product);
        product = simProducts.getProduct(1);
        assertEquals("Product2", product);
    }
}
