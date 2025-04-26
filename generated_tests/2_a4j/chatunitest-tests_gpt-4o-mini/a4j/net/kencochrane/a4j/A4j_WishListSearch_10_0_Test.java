package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

class A4j_WishListSearch_10_0_Test {

    private A4j a4j;

    private Search searchMock;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        searchMock = mock(Search.class);
    }

    @Test
    void testWishListSearch_ValidId() {
        // Arrange
        String wishListId = "validId";
        // Assume this is a valid ProductInfo object
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.WishListSearch(wishListId)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.WishListSearch(wishListId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void testWishListSearch_EmptyId() {
        // Arrange
        String wishListId = "";
        // Assume it returns null for empty ID
        when(searchMock.WishListSearch(wishListId)).thenReturn(null);
        // Act
        ProductInfo actualProductInfo = a4j.WishListSearch(wishListId);
        // Assert
        assertNull(actualProductInfo);
    }

    @Test
    void testWishListSearch_NullId() {
        // Arrange
        String wishListId = null;
        when(searchMock.WishListSearch(wishListId)).thenThrow(new IllegalArgumentException("ID cannot be null"));
        // Act & Assert
        Exception exception = assertThrows(IllegalArgumentException.class, () -> {
            a4j.WishListSearch(wishListId);
        });
        assertEquals("ID cannot be null", exception.getMessage());
    }
}
