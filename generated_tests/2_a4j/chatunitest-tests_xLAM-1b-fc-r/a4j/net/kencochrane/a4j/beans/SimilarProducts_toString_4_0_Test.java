package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class SimilarProducts_toString_4_0_Test {

    SimilarProducts similarProducts;

    @Test
    void toStringTest() throws Exception {
        similarProducts = new SimilarProducts();
        String[] product = { "product1", "product2", "product3" };
        similarProducts.setProduct(product);
        String expectedOutput = "# of Simular products = 3\n" + "ProductAction - product1\n" + "ProductAction - product2\n" + "ProductAction - product3\n";
        Field field = SimilarProducts.class.getDeclaredField("simProducts");
        field.setAccessible(true);
        String result = similarProducts.toString();
        assertEquals(expectedOutput, result);
    }
}
