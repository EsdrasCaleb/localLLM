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

public class RecentlyViewed_isInList_3_0_Test {

    private RecentlyViewed recentlyViewed;

    @BeforeEach
    void setUp() {
        recentlyViewed = new RecentlyViewed();
    }

    @Test
    void testIsInList_ProductListIsNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field productsField = RecentlyViewed.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(recentlyViewed, null);
        // Act & Assert
        assertFalse(recentlyViewed.isInList("testAsin"));
    }

    @Test
    void testIsInList_ProductListIsEmpty() {
        // Act & Assert
        assertFalse(recentlyViewed.isInList("testAsin"));
    }

    static class MiniProduct {

        private String asin;

        public String getAsin() {
            return asin;
        }
    }
}
