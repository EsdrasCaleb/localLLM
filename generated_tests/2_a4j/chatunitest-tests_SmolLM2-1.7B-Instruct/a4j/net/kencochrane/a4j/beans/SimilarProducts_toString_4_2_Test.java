// Test method
package net.kencochrane.a4j.beans;

import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import java.io.Serializable;
import java.util.ArrayList;

@ExtendWith(MockitoExtension.class)
public class SimilarProducts_toString_4_2_Test {

    @Mock
    private Serializable similarProducts;

    @InjectMocks
    private SimilarProducts similarProductsUnderTest;

    @BeforeEach
    public void setUp() {
        similarProductsUnderTest.setProduct(new String[] { "ProductA", "ProductB", "ProductC" });
    }

    @Test
    public void testToString_SimilarProductsNotNull() {
        Assertions.assertEquals("# of Simular products = 3\nProductAction - ProductA\nProductAction - ProductB\nProductAction - ProductC", similarProductsUnderTest.toString());
    }

    @Test
    public void testToString_SimilarProductsNull() {
        Assertions.assertEquals("Similar Products is null or size 0", similarProductsUnderTest.toString());
    }

    @Test
    public void testToString_SimilarProductsEmpty() {
        similarProductsUnderTest.setProduct(new String[0]);
        Assertions.assertEquals("Similar Products is null or size 0", similarProductsUnderTest.toString());
    }
}
