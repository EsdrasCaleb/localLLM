package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_toString_4_0_Test {

    @InjectMocks
    private SimilarProducts similarProducts;

    @Mock
    private ArrayList<String> simProducts;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToStringWithProducts() {
        when(simProducts.size()).thenReturn(2);
        when(simProducts.get(0)).thenReturn("Product1");
        when(simProducts.get(1)).thenReturn("Product2");
        similarProducts.setProduct(new String[] { "Product1", "Product2" });
        String expected = "# of Simular products = 2\n" + "ProductAction - Product1\n" + "ProductAction - Product2\n";
        assertEquals(expected, similarProducts.toString());
    }

    @Test
    public void testToStringWithNoProducts() {
        when(simProducts.size()).thenReturn(0);
        similarProducts.setProduct(new String[] {});
        String expected = "Similar Products is null or size 0\n";
        assertEquals(expected, similarProducts.toString());
    }

    @Test
    public void testToStringWithNullProducts() {
        similarProducts.setProduct(null);
        String expected = "Similar Products is null or size 0\n";
        assertEquals(expected, similarProducts.toString());
    }
}
