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

    @Mock
    private ArrayList<String> simProducts;

    @InjectMocks
    private SimilarProducts similarProducts;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testToString() {
        when(simProducts.size()).thenReturn(2);
        when(simProducts.get(0)).thenReturn("Product 1");
        when(simProducts.get(1)).thenReturn("Product 2");
        String expected = "# of Simular products = 2\n" + "ProductAction - Product 1\n" + "ProductAction - Product 2\n";
        String actual = similarProducts.toString();
        assertEquals(expected, actual);
    }
}
