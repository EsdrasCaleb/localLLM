package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SimilarProducts_getProduct_3_0_Test {

    @Test
    public void testGetProduct_InValidIndex_ReturnsNull() {
        SimilarProducts similarProducts = new SimilarProducts();
        similarProducts.setProduct(new String[] { "Product1", "Product2", "Product3" });
        String result = similarProducts.getProduct(-1);
        assertNull(result);
    }

    @Test
    public void testGetProduct_ValidIndex_ReturnsProduct() {
        SimilarProducts similarProducts = new SimilarProducts();
        similarProducts.setProduct(new String[] { "Product1", "Product2", "Product3" });
        String result = similarProducts.getProduct(1);
        assertEquals("Product2", result);
    }

    @Test
    public void testGetProduct_LastIndex_ReturnsProduct() {
        SimilarProducts similarProducts = new SimilarProducts();
        similarProducts.setProduct(new String[] { "Product1", "Product2", "Product3" });
        String result = similarProducts.getProduct(similarProducts.getProductsArray().size() - 1);
        assertEquals("Product3", result);
    }
}
