package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class RecentlyViewed_addProduct_0_1_Test {

    private RecentlyViewed recentlyViewed;

    private MiniProduct mockMiniProduct;

    @BeforeEach
    public void setUp() throws Exception {
        recentlyViewed = new RecentlyViewed();
        mockMiniProduct = mock(MiniProduct.class);
    }

    @Test
    public void testAddProductWithNewProduct() {
        when(mockMiniProduct.getAsin()).thenReturn("asin1");
        recentlyViewed.addProduct(mockMiniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
        assertTrue(recentlyViewed.getProducts().contains(mockMiniProduct));
    }

    @Test
    public void testAddProductWithExistingProduct() {
        when(mockMiniProduct.getAsin()).thenReturn("asin2");
        recentlyViewed.addProduct(mockMiniProduct);
        // Add the same product again
        recentlyViewed.addProduct(mockMiniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
        assertTrue(recentlyViewed.getProducts().contains(mockMiniProduct));
    }

    @Test
    public void testAddProductWithNullProduct() {
        recentlyViewed.addProduct(null);
        assertEquals(0, recentlyViewed.getNumProducts());
    }

    @Test
    public void testAddProductWithEmptyAsin() {
        when(mockMiniProduct.getAsin()).thenReturn("");
        recentlyViewed.addProduct(mockMiniProduct);
        assertEquals(0, recentlyViewed.getNumProducts());
    }
}
