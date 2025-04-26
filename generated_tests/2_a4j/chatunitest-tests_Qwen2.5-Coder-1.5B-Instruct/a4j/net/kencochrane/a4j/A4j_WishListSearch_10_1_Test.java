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

class A4j_WishListSearch_10_1_Test {

    private A4j a4j;

    private Search mockSearch;

    @BeforeEach
    public void setUp() {
        a4j = new A4j();
        mockSearch = mock(Search.class);
        when(mockSearch.WishListSearch(anyString())).thenReturn(new ProductInfo());
    }

    @Test
    public void testWishListSearch() {
        // Given
        String wishListId = "123";
        // When
        ProductInfo result = a4j.WishListSearch(wishListId);
        // Then
        assertNotNull(result);
        verify(mockSearch).WishListSearch(wishListId);
    }
}
