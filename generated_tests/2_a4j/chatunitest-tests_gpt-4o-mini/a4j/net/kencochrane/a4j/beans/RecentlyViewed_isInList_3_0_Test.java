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

    @BeforeEach
    void setUp() {
        recentlyViewed = new RecentlyViewed();
    }

    @Test
    void testIsInList_NoProducts() {
        assertFalse(recentlyViewed.isInList("12345"));
    }

    @Test
    void testIsInList_ProductIsNull() {
        addNullProductToRecentlyViewed();
        assertFalse(recentlyViewed.isInList("12345"));
    }

    private void addProductToRecentlyViewed(MiniProduct product) {
        try {
            Field field = RecentlyViewed.class.getDeclaredField("products");
            field.setAccessible(true);
            ArrayList<MiniProduct> products = (ArrayList<MiniProduct>) field.get(recentlyViewed);
            products.add(product);
        } catch (Exception e) {
            fail("Failed to add product to RecentlyViewed");
        }
    }

    private void addNullProductToRecentlyViewed() {
        try {
            Field field = RecentlyViewed.class.getDeclaredField("products");
            field.setAccessible(true);
            ArrayList<MiniProduct> products = (ArrayList<MiniProduct>) field.get(recentlyViewed);
            products.add(null);
        } catch (Exception e) {
            fail("Failed to add null product to RecentlyViewed");
        }
    }
}

class MiniProduct {

    private String asin;

    public MiniProduct(String asin) {
        this.asin = asin;
    }

    public String getAsin() {
        return asin;
    }
}
