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

public class A4j_DirectorSearch_6_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testDirectorSearch() {
        // Arrange
        String directorName = "Steven Spielberg";
        String mode = "advanced";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.DirectorSearch(directorName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.DirectorSearch(directorName, mode, page);
        // Assert
        assertNotNull(actualProductInfo);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).DirectorSearch(directorName, mode, page);
    }
}
