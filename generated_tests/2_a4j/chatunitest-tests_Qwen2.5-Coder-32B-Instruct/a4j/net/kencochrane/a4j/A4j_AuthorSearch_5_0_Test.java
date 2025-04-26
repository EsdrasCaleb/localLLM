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

public class A4j_AuthorSearch_5_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAuthorSearch_Success() {
        // Arrange
        String authorName = "John Doe";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.AuthorSearch(authorName, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).AuthorSearch(authorName, page);
    }

    @Test
    public void testAuthorSearch_AuthorNameNull() {
        // Arrange
        String authorName = null;
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.AuthorSearch(authorName, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).AuthorSearch(authorName, page);
    }

    @Test
    public void testAuthorSearch_PageNull() {
        // Arrange
        String authorName = "John Doe";
        String page = null;
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.AuthorSearch(authorName, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).AuthorSearch(authorName, page);
    }

    @Test
    public void testAuthorSearch_BothNull() {
        // Arrange
        String authorName = null;
        String page = null;
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.AuthorSearch(authorName, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).AuthorSearch(authorName, page);
    }
}
