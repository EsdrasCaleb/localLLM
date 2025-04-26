package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_getProduct_3_1_Test {

    SimilarProducts similarProducts;

    ArrayList<String> mockedProducts;

    @BeforeEach
    public void setUp() {
        similarProducts = new SimilarProducts();
        mockedProducts = Mockito.mock(ArrayList.class);
        similarProducts.simProducts = mockedProducts;
    }

    @Test
    public void testGetProduct() {
        String product1 = "Product1";
        String product2 = "Product2";
        Mockito.when(mockedProducts.get(0)).thenReturn(product1);
        Mockito.when(mockedProducts.get(1)).thenReturn(product2);
        assertEquals(product1, similarProducts.getProduct(0));
        assertEquals(product2, similarProducts.getProduct(1));
    }
}
