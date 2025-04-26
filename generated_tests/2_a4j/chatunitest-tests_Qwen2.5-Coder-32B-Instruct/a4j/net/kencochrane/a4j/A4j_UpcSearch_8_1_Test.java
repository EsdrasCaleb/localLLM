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

public class A4j_UpcSearch_8_1_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testUpcSearch_Success() {
        // Arrange
        String upc = "123456789012";
        String mode = "basic";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).UpcSearch(upc, mode, page);
    }

    @Test
    public void testUpcSearch_NullUpc() {
        // Arrange
        String upc = null;
        String mode = "basic";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).UpcSearch(upc, mode, page);
    }

    @Test
    public void testUpcSearch_EmptyUpc() {
        // Arrange
        String upc = "";
        String mode = "basic";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).UpcSearch(upc, mode, page);
    }

    @Test
    public void testUpcSearch_NullMode() {
        // Arrange
        String upc = "123456789012";
        String mode = null;
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).UpcSearch(upc, mode, page);
    }

    @Test
    public void testUpcSearch_EmptyMode() {
        // Arrange
        String upc = "123456789012";
        String mode = "";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).UpcSearch(upc, mode, page);
    }

    @Test
    public void testUpcSearch_NullPage() {
        // Arrange
        String upc = "123456789012";
        String mode = "basic";
        String page = null;
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).UpcSearch(upc, mode, page);
    }

    @Test
    public void testUpcSearch_EmptyPage() {
        // Arrange
        String upc = "123456789012";
        String mode = "basic";
        String page = "";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.UpcSearch(upc, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
