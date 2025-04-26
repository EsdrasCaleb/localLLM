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

public class RecentlyViewed_addProduct_0_0_Test {

    private RecentlyViewed rv;

    private MiniProduct miniProd;

    @BeforeEach
    public void setUp() {
        rv = new RecentlyViewed();
        miniProd = Mockito.mock(MiniProduct.class);
    }

    @Test
    public void testAddProduct() throws Exception {
        Field field = RecentlyViewed.class.getDeclaredField("products");
        field.setAccessible(true);
        ArrayList products = (ArrayList) field.get(rv);
        // Mocking the behavior of the miniProd.getAsin() method
        Mockito.when(miniProd.getAsin()).thenReturn("testAsin");
        rv.addProduct(miniProd);
        // Checking if the product is added to the list
        assertEquals(1, products.size());
        assertEquals(miniProd, products.get(0));
    }
}
