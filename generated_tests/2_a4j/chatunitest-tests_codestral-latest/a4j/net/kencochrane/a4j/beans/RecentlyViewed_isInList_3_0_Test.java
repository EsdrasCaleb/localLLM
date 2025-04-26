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

class RecentlyViewed_isInList_3_0_Test {

    private RecentlyViewed recentlyViewed;

    private ArrayList<MiniProduct> products;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        recentlyViewed = new RecentlyViewed();
        products = new ArrayList<>();
        // Use reflection to set the private products field
        Field field = RecentlyViewed.class.getDeclaredField("products");
        field.setAccessible(true);
        field.set(recentlyViewed, products);
    }

    @Test
    void testIsInList_ProductInList() {
        MiniProduct product1 = mock(MiniProduct.class);
        when(product1.getAsin()).thenReturn("12345");
        products.add(product1);
        assertTrue(recentlyViewed.isInList("12345"));
    }

    @Test
    void testIsInList_ProductNotInList() {
        MiniProduct product1 = mock(MiniProduct.class);
        when(product1.getAsin()).thenReturn("12345");
        products.add(product1);
        assertFalse(recentlyViewed.isInList("67890"));
    }

    @Test
    void testIsInList_EmptyList() {
        assertFalse(recentlyViewed.isInList("12345"));
    }

    @Test
    void testIsInList_NullAsin() {
        MiniProduct product1 = mock(MiniProduct.class);
        when(product1.getAsin()).thenReturn(null);
        products.add(product1);
        assertFalse(recentlyViewed.isInList("12345"));
    }

    @Test
    void testIsInList_NullProduct() {
        products.add(null);
        assertFalse(recentlyViewed.isInList("12345"));
    }

    @Test
    void testIsInList_ProductsNull() throws NoSuchFieldException, IllegalAccessException {
        // Set products to null using reflection
        Field field = RecentlyViewed.class.getDeclaredField("products");
        field.setAccessible(true);
        field.set(recentlyViewed, null);
        assertFalse(recentlyViewed.isInList("12345"));
    }
}
