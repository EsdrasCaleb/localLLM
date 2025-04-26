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

    @Test
    void testIsInList_ProductListContainsMatchingAsin() {
        // Arrange
        ArrayList<MiniProduct> products = new ArrayList<>();
        MiniProduct mockMiniProduct = mock(MiniProduct.class);
        when(mockMiniProduct.getAsin()).thenReturn("testAsin");
        products.add(mockMiniProduct);
        Field productsField;
        try {
            productsField = RecentlyViewed.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(recentlyViewed, products);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
        // Act & Assert
        assertTrue(recentlyViewed.isInList("testAsin"));
        verify(mockMiniProduct).getAsin();
    }

    @Test
    void testIsInList_ProductListContainsNonMatchingAsin() {
        // Arrange
        ArrayList<MiniProduct> products = new ArrayList<>();
        MiniProduct mockMiniProduct = mock(MiniProduct.class);
        when(mockMiniProduct.getAsin()).thenReturn("anotherAsin");
        products.add(mockMiniProduct);
        Field productsField;
        try {
            productsField = RecentlyViewed.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(recentlyViewed, products);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
        // Act & Assert
        assertFalse(recentlyViewed.isInList("testAsin"));
        verify(mockMiniProduct).getAsin();
    }

    @Test
    void testIsInList_ProductListContainsNullMiniProduct() {
        // Arrange
        ArrayList<MiniProduct> products = new ArrayList<>();
        products.add(null);
        products.add(mock(MiniProduct.class));
        Field productsField;
        try {
            productsField = RecentlyViewed.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(recentlyViewed, products);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
        // Act & Assert
        assertFalse(recentlyViewed.isInList("testAsin"));
    }

    @Test
    void testIsInList_ProductListContainsMiniProductWithNullAsin() {
        // Arrange
        ArrayList<MiniProduct> products = new ArrayList<>();
        MiniProduct mockMiniProduct = mock(MiniProduct.class);
        when(mockMiniProduct.getAsin()).thenReturn(null);
        products.add(mockMiniProduct);
        Field productsField;
        try {
            productsField = RecentlyViewed.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(recentlyViewed, products);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
        // Act & Assert
        assertFalse(recentlyViewed.isInList("testAsin"));
        verify(mockMiniProduct).getAsin();
    }

    @Test
    void testIsInList_ProductListContainsMiniProductWithWhitespaceAsin() {
        // Arrange
        ArrayList<MiniProduct> products = new ArrayList<>();
        MiniProduct mockMiniProduct = mock(MiniProduct.class);
        when(mockMiniProduct.getAsin()).thenReturn(" testAsin ");
        products.add(mockMiniProduct);
        Field productsField;
        try {
            productsField = RecentlyViewed.class.getDeclaredField("products");
            productsField.setAccessible(true);
            productsField.set(recentlyViewed, products);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            fail("Reflection failed: " + e.getMessage());
        }
        // Act & Assert
        assertTrue(recentlyViewed.isInList("testAsin"));
        verify(mockMiniProduct).getAsin();
    }

    static class MiniProduct {

        private String asin;

        public String getAsin() {
            return asin;
        }
    }
}
