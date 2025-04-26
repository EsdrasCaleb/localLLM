package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

class SimilarProducts_getProduct_3_1_Test {

    @ExtendWith(MockitoExtension.class)
    public class SimilarProductsTests {

        @Mock
        private SimilarProducts similarProducts;

        @Test
        public void testGetProduct() {
            int index = 2;
            // <Buggy Line>: cannot find symbol  symbol:   variable similarProducts  location: class net.kencochrane.a4j.beans.SimilarProducts_getProduct_3_1_Test
            when(similarProducts.getProduct(index)).thenReturn("Product 3");
            String result = similarProducts.getProduct(index);
            assertEquals("Product 3", result);
        }
    }
}
