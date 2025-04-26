package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class SimilarProducts_toString_4_0_Test {

    @InjectMocks
    private SimilarProducts similarProducts;

    @Test
    public void testToString_EmptyList() {
        similarProducts.setProduct(new String[0]);
        String result = similarProducts.toString();
        assertEquals("Similar Products is null or size 0", result);
    }

    @Test
    public void testToString_SingleElementList() {
        String[] products = { "Product1", "Product2", "Product3" };
        similarProducts.setProduct(products);
        String result = similarProducts.toString();
        assertEquals("ProductAction - Product1\nProductAction - Product2\nProductAction - Product3", result);
    }

    @Test
    public void testToString_MultipleElementsList() {
        String[] products = { "Product1", "Product2", "Product3", "Product4" };
        similarProducts.setProduct(products);
        String result = similarProducts.toString();
        assertEquals("ProductAction - Product1\nProductAction - Product2\nProductAction - Product3\nProductAction - Product4", result);
    }
}
