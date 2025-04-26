package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class SimilarProducts_toString_4_0_Test {

    @Test
    public void testToString() {
        // Given
        SimilarProducts simProduct = new SimilarProducts();
        simProduct.setProduct(new String[] { "Product1", "Product2", "Product3" });
        simProduct.setProduct(new String[] { "Product4", "Product5", "Product6" });
        simProduct.setProduct(new String[] { "Product7", "Product8", "Product9" });
        // When
        String output = simProduct.toString();
        // Then
        assertEquals("# of Simular products = 3\nProductAction - Product1\nProductAction - Product2\nProductAction - Product3\nProductAction - Product4\nProductAction - Product5\nProductAction - Product6\nProductAction - Product7\nProductAction - Product8\nProductAction - Product9", output);
    }
}
