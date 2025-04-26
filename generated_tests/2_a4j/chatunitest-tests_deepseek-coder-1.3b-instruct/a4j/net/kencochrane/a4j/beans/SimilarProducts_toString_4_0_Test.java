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

    SimilarProducts similarProducts;

    ArrayList<String> mockList;

    @BeforeEach
    public void setUp() {
        mockList = Mockito.mock(ArrayList.class);
        similarProducts = new SimilarProducts();
        similarProducts.simProducts = mockList;
    }

    @Test
    public void testToString() {
        String expected = "Similar Products is null or size 0";
        Mockito.when(mockList.size()).thenReturn(0);
        Mockito.when(mockList.get(0)).thenReturn(null);
        assertEquals(expected, similarProducts.toString());
        String product = "Product";
        Mockito.when(mockList.size()).thenReturn(1);
        Mockito.when(mockList.get(0)).thenReturn(product);
        expected = "# of Simular products = 1\nProductAction - Product";
        assertEquals(expected, similarProducts.toString());
    }
}
