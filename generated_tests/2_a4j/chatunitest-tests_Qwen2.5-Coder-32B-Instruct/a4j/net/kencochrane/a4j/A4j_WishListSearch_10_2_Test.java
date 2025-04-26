package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Search;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_WishListSearch_10_2_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testWishListSearch_withValidId() {
        // Arrange
        String wishListId = "validWishListId";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.WishListSearch(wishListId)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.WishListSearch(wishListId);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).WishListSearch(wishListId);
    }

    @Test
    public void testWishListSearch_withNullId() {
        // Arrange
        String wishListId = null;
        // Act
        ProductInfo result = a4j.WishListSearch(wishListId);
        // Assert
        assertNull(result);
        verify(search, never()).WishListSearch(wishListId);
    }

    @Test
    public void testWishListSearch_withEmptyId() {
        // Arrange
        String wishListId = "";
        // Act
        ProductInfo result = a4j.WishListSearch(wishListId);
        // Assert
        assertNull(result);
        verify(search, never()).WishListSearch(wishListId);
    }
}
