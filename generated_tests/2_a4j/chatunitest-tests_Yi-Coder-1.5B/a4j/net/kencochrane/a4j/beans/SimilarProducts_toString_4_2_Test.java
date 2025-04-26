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
public class SimilarProducts_toString_4_2_Test {

    @Test
    public void testToString() {
        SimilarProducts test = new SimilarProducts();
        test.setProduct(new String[] { "Product 1", "Product 2", "Product 3", "Product 4" });
        assertEquals("# of Simular products = 4\nProductAction - Product 1\nProductAction - Product 2\nProductAction - Product 3\nProductAction - Product 4\n", test.toString());
    }
}
