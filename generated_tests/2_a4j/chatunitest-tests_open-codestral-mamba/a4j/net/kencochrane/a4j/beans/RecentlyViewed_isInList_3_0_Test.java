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

    @Mock
    private ArrayList<MiniProduct> products;

    @InjectMocks
    private RecentlyViewed recentlyViewed;

    @BeforeEach
    void setUp() throws NoSuchFieldException, IllegalAccessException {
        MockitoAnnotations.initMocks(this);
        Field productsField = RecentlyViewed.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(recentlyViewed, products);
    }

    @Test
    void isInList_WithEmptyList_ReturnsFalse() {
        when(products.size()).thenReturn(0);
        assertFalse(recentlyViewed.isInList("12345"));
    }

    @Test
    void isInList_WithNullList_ReturnsFalse() throws NoSuchFieldException, IllegalAccessException {
        Field productsField = RecentlyViewed.class.getDeclaredField("products");
        productsField.setAccessible(true);
        productsField.set(recentlyViewed, null);
        assertFalse(recentlyViewed.isInList("12345"));
    }
}
