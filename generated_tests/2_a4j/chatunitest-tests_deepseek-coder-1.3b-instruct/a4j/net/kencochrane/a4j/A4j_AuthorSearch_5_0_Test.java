package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_AuthorSearch_5_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testAuthorSearch() {
        // Given
        String authorName = "Test Author";
        String page = "Test Page";
        ProductInfo productInfo = new ProductInfo();
        when(search.AuthorSearch(authorName, page)).thenReturn(productInfo);
        // When
        ProductInfo result = a4j.AuthorSearch(authorName, page);
        // Then
        assertNotNull(result);
        verify(search, times(1)).AuthorSearch(authorName, page);
    }
}
