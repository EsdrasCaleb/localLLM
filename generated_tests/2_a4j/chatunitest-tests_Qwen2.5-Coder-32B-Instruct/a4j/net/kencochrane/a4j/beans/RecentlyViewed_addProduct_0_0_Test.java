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

class RecentlyViewed_addProduct_0_0_Test {

    @Mock
    private MiniProduct mockMiniProduct;

    private RecentlyViewed recentlyViewed;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
        recentlyViewed = new RecentlyViewed();
    }

    @Test
    void testAddProduct_NullProduct() {
        recentlyViewed.addProduct(null);
        assertTrue(recentlyViewed.getProducts().isEmpty());
    }

    @Test
    void testAddProduct_ProductNotInList() {
        when(mockMiniProduct.getAsin()).thenReturn("123");
        recentlyViewed.addProduct(mockMiniProduct);
        assertEquals(1, recentlyViewed.getProducts().size());
        assertSame(mockMiniProduct, recentlyViewed.getProducts().get(0));
    }

    @Test
    void testAddProduct_ProductAlreadyInList() throws Exception {
        when(mockMiniProduct.getAsin()).thenReturn("123");
        recentlyViewed.addProduct(mockMiniProduct);
        MiniProduct anotherMockMiniProduct = mock(MiniProduct.class);
        when(anotherMockMiniProduct.getAsin()).thenReturn("123");
        recentlyViewed.addProduct(anotherMockMiniProduct);
        assertEquals(1, recentlyViewed.getProducts().size());
        assertSame(mockMiniProduct, recentlyViewed.getProducts().get(0));
    }

    private boolean isInList(String asin) throws Exception {
        Field productsField = RecentlyViewed.class.getDeclaredField("products");
        productsField.setAccessible(true);
        ArrayList<MiniProduct> products = (ArrayList<MiniProduct>) productsField.get(recentlyViewed);
        for (MiniProduct product : products) {
            if (product.getAsin().equals(asin)) {
                return true;
            }
        }
        return false;
    }
}

class MiniProduct {

    private String asin;

    public String getAsin() {
        return asin;
    }

    public void setAsin(String asin) {
        this.asin = asin;
    }
}
