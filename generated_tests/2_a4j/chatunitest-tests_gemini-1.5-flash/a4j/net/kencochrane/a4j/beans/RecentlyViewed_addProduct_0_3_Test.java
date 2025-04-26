package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RecentlyViewed_addProduct_0_3_Test {

    private RecentlyViewed recentlyViewed;

    @BeforeEach
    void setUp() {
        recentlyViewed = new RecentlyViewed();
    }

    @Test
    void addProduct_nullProduct_doesNotAdd() {
        recentlyViewed.addProduct(null);
        assertEquals(0, recentlyViewed.getNumProducts());
    }

    @Test
    void addProduct_existingProduct_doesNotAdd() {
        MiniProduct product1 = new MiniProduct();
        product1.setAsin("asin1");
        // Corrected: Use setName instead of setTitle
        product1.setName("name1");
        recentlyViewed.addProduct(product1);
        recentlyViewed.addProduct(product1);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    @Test
    void addProduct_newProduct_addsProduct() {
        MiniProduct product1 = new MiniProduct();
        product1.setAsin("asin1");
        // Corrected: Use setName instead of setTitle
        product1.setName("name1");
        recentlyViewed.addProduct(product1);
        assertEquals(1, recentlyViewed.getNumProducts());
        assertEquals(product1, recentlyViewed.getProducts().get(0));
    }

    @Test
    void addProduct_multipleNewProducts_addsAllProducts() {
        MiniProduct product1 = new MiniProduct();
        product1.setAsin("asin1");
        // Corrected: Use setName instead of setTitle
        product1.setName("name1");
        MiniProduct product2 = new MiniProduct();
        product2.setAsin("asin2");
        // Corrected: Use setName instead of setTitle
        product2.setName("name2");
        recentlyViewed.addProduct(product1);
        recentlyViewed.addProduct(product2);
        assertEquals(2, recentlyViewed.getNumProducts());
        assertTrue(recentlyViewed.getProducts().contains(product1));
        assertTrue(recentlyViewed.getProducts().contains(product2));
    }
}
