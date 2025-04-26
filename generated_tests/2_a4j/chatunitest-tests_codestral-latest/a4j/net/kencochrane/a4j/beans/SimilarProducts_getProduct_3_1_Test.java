package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class SimilarProducts_getProduct_3_1_Test {

    @InjectMocks
    private SimilarProducts similarProducts;

    @Mock
    private ArrayList<String> simProducts;

    @BeforeEach
    public void setUp() {
        similarProducts = new SimilarProducts();
        similarProducts.simProducts = simProducts;
    }

    @Test
    public void testGetProduct_IndexWithinBounds() {
        when(simProducts.size()).thenReturn(3);
        when(simProducts.get(1)).thenReturn("Product2");
        String result = similarProducts.getProduct(1);
        assertEquals("Product2", result);
    }

    @Test
    public void testGetProduct_IndexOutOfBounds() {
        when(simProducts.size()).thenReturn(3);
        String result = similarProducts.getProduct(3);
        assertNull(result);
    }

    @Test
    public void testGetProduct_EmptyList() {
        when(simProducts.size()).thenReturn(0);
        String result = similarProducts.getProduct(0);
        assertNull(result);
    }
}
