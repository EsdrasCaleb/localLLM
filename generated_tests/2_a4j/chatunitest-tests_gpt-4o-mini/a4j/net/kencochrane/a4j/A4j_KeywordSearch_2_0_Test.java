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

class A4j_KeywordSearch_2_0_Test {

    private A4j a4j;

    private Search searchMock;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        searchMock = mock(Search.class);
    }

    @Test
    void testKeywordSearch_ValidInputs() {
        // Arrange
        String searchTerm = "laptop";
        String productLine = "electronics";
        String type = "new";
        String page = "1";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        // Mocking the Keyword method
        when(searchMock.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).Keyword(searchTerm, productLine, type, page);
    }

    @Test
    void testKeywordSearch_EmptySearchTerm() {
        // Arrange
        String searchTerm = "";
        String productLine = "electronics";
        String type = "new";
        String page = "1";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        // Mocking the Keyword method
        when(searchMock.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).Keyword(searchTerm, productLine, type, page);
    }

    @Test
    void testKeywordSearch_NullProductLine() {
        // Arrange
        String searchTerm = "laptop";
        String productLine = null;
        String type = "new";
        String page = "1";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        // Mocking the Keyword method
        when(searchMock.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).Keyword(searchTerm, productLine, type, page);
    }

    @Test
    void testKeywordSearch_InvalidType() {
        // Arrange
        String searchTerm = "laptop";
        String productLine = "electronics";
        String type = "invalidType";
        String page = "1";
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        // Mocking the Keyword method
        when(searchMock.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).Keyword(searchTerm, productLine, type, page);
    }

    @Test
    void testKeywordSearch_NullPage() {
        // Arrange
        String searchTerm = "laptop";
        String productLine = "electronics";
        String type = "new";
        String page = null;
        // Assume this is a valid object
        ProductInfo expectedProductInfo = new ProductInfo();
        // Mocking the Keyword method
        when(searchMock.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(searchMock).Keyword(searchTerm, productLine, type, page);
    }
}
