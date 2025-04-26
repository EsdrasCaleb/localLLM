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
    void testIsInList_emptyProductsList() {
        assertFalse(recentlyViewed.isInList("B07XYZ123"));
    }

    @Test
    void testIsInList_nullProductsList() {
        // Simulate null list using reflection (though ideally, the class would handle this gracefully)
        try {
            Field productsField = RecentlyViewed.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(recentlyViewed, null);
            assertFalse(recentlyViewed.isInList("B07XYZ123"));
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }

    @Test
    void testIsInList_productNotFound() {
        recentlyViewed.getProducts().add(new MiniProduct("B01ABC123", "Product 1"));
        assertFalse(recentlyViewed.isInList("B07XYZ123"));
    }

    @Test
    void testIsInList_productFound_exactMatch() {
        recentlyViewed.getProducts().add(new MiniProduct("B01ABC123", "Product 1"));
        assertTrue(recentlyViewed.isInList("B01ABC123"));
    }

    @Test
    void testIsInList_productFound_caseInsensitiveMatch() {
        recentlyViewed.getProducts().add(new MiniProduct("b01abc123", "Product 1"));
        assertTrue(recentlyViewed.isInList("B01ABC123"));
    }

    @Test
    void testIsInList_productFound_withWhitespace() {
        recentlyViewed.getProducts().add(new MiniProduct(" B01ABC123 ", "Product 1"));
        assertTrue(recentlyViewed.isInList("B01ABC123"));
    }

    @Test
    void testIsInList_nullAsin() {
        recentlyViewed.getProducts().add(new MiniProduct("B01ABC123", "Product 1"));
        assertFalse(recentlyViewed.isInList(null));
    }

    @Test
    void testIsInList_emptyAsin() {
        recentlyViewed.getProducts().add(new MiniProduct("B01ABC123", "Product 1"));
        assertFalse(recentlyViewed.isInList(""));
    }

    @Test
    void testIsInList_nullAsinInList() {
        recentlyViewed.getProducts().add(new MiniProduct(null, "Product 1"));
        assertFalse(recentlyViewed.isInList("B01ABC123"));
    }

    // Inner class for MiniProduct (replace with your actual class if different)
    class MiniProduct {

        private String asin;

        private String name;

        MiniProduct(String asin, String name) {
            this.asin = asin;
            this.name = name;
        }

        public String getAsin() {
            return asin;
        }
    }
}
