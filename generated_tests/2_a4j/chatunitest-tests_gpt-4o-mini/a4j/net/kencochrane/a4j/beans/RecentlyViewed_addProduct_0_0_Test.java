package net.kencochrane.a4j.beans;

import java.lang.reflect.Method;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class RecentlyViewed_addProduct_0_0_Test {

    private RecentlyViewed recentlyViewed;

    private MiniProduct miniProduct;

    @BeforeEach
    void setUp() {
        recentlyViewed = new RecentlyViewed();
        miniProduct = mock(MiniProduct.class);
    }

    @Test
    void testAddProduct_NullProduct() {
        recentlyViewed.addProduct(null);
        assertEquals(0, recentlyViewed.getNumProducts());
    }

    @Test
    void testAddProduct_ProductAlreadyInList() throws Exception {
        when(miniProduct.getAsin()).thenReturn("12345");
        // Add the product first
        invokeAddProduct(miniProduct);
        // Attempt to add the same product again
        recentlyViewed.addProduct(miniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    @Test
    void testAddProduct_ValidProduct() throws Exception {
        when(miniProduct.getAsin()).thenReturn("67890");
        recentlyViewed.addProduct(miniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    @Test
    void testAddProduct_ValidProduct_NotInList() throws Exception {
        when(miniProduct.getAsin()).thenReturn("11111");
        recentlyViewed.addProduct(miniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    private void invokeAddProduct(MiniProduct miniProd) throws Exception {
        Method method = RecentlyViewed.class.getDeclaredMethod("addProduct", MiniProduct.class);
        method.setAccessible(true);
        method.invoke(recentlyViewed, miniProd);
    }

    private boolean isInList(String asin) throws Exception {
        Method method = RecentlyViewed.class.getDeclaredMethod("isInList", String.class);
        method.setAccessible(true);
        return (boolean) method.invoke(recentlyViewed, asin);
    }
}
