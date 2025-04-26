package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class RecentlyViewed_addProduct_0_0_Test {

    @Test
    void addProduct_validProduct_success() {
        // Arrange
        RecentlyViewed rv = new RecentlyViewed();
        MiniProduct miniProd = Mockito.mock(MiniProduct.class);
        Mockito.when(miniProd.getAsin()).thenReturn("12345");
        // Act
        rv.addProduct(miniProd);
        // Assert
        assertEquals(1, rv.getNumProducts());
    }

    @Test
    void addProduct_nullProduct_failure() {
        // Arrange
        RecentlyViewed rv = new RecentlyViewed();
        // Act & Assert
        assertThrows(NullPointerException.class, () -> rv.addProduct(null));
    }

    @Test
    void addProduct_duplicateProduct_failure() {
        // Arrange
        RecentlyViewed rv = new RecentlyViewed();
        MiniProduct miniProd = Mockito.mock(MiniProduct.class);
        Mockito.when(miniProd.getAsin()).thenReturn("12345");
        rv.addProduct(miniProd);
        // Act
        rv.addProduct(miniProd);
        // Assert
        assertEquals(1, rv.getNumProducts());
    }
}
