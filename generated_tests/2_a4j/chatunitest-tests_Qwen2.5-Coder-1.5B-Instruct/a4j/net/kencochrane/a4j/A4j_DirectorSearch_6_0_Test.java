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
public class A4j_DirectorSearch_6_0_Test {

    @Mock
    private Search mockSearch;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testDirectorSearch() throws Exception {
        // Arrange
        String directorName = "John Doe";
        String mode = "all";
        String page = "1";
        // Create a ProductInfo object to be returned by mockSearch.DirectorSearch()
        ProductInfo productInfo = new ProductInfo();
        when(mockSearch.DirectorSearch(directorName, mode, page)).thenReturn(productInfo);
        // Act
        ProductInfo result = a4j.DirectorSearch(directorName, mode, page);
        // Assert
        assertNotNull(result);
        assertEquals(productInfo, result);
    }
}
