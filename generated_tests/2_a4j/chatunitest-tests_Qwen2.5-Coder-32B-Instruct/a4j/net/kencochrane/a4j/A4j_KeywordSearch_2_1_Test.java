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

public class A4j_KeywordSearch_2_1_Test {

    @Mock
    private Search searchMock;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testKeywordSearch() {
        // Arrange
        String searchTerm = "laptop";
        String productLine = "electronics";
        String type = "new";
        String page = "1";
        // Assuming ProductInfo has a no-arg constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.Keyword(searchTerm, productLine, type, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.KeywordSearch(searchTerm, productLine, type, page);
        // Assert
        assertNotNull(actualProductInfo);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(searchMock, times(1)).Keyword(searchTerm, productLine, type, page);
    }
}
