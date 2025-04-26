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

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testKeywordSearch() {
        // Arrange
        String searchTerm = "term";
        String productLine = "line";
        String type = "type";
        String page = "page";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).Keyword(searchTerm, productLine, type, page);
    }
}
