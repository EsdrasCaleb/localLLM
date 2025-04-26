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

public class SimilarProducts_getProduct_3_0_Test {

    SimilarProducts similarProducts;

    @BeforeEach
    public void setup() {
        similarProducts = new SimilarProducts();
        similarProducts.setProduct(new String[] { "product1", "product2", "product3" });
    }

    @Test
    public void testGetProduct() throws NoSuchFieldException, IllegalAccessException {
        Field field = SimilarProducts.class.getDeclaredField("simProducts");
        field.setAccessible(true);
        ArrayList simProducts = (ArrayList) field.get(similarProducts);
        assertEquals("product1", similarProducts.getProduct(0));
        assertEquals("product2", similarProducts.getProduct(1));
        assertEquals("product3", similarProducts.getProduct(2));
    }
}
