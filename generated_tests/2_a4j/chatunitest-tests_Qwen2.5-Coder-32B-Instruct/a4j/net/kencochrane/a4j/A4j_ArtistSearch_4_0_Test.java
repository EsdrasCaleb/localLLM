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

public class A4j_ArtistSearch_4_0_Test {

    @Mock
    private Search searchMock;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testArtistSearch_Success() {
        // Arrange
        String artistName = "John Doe";
        String mode = "album";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(searchMock, times(1)).ArtistSearch(artistName, mode, page);
    }

    @Test
    public void testArtistSearch_NullArtistName() {
        // Arrange
        String artistName = null;
        String mode = "album";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(searchMock, times(1)).ArtistSearch(artistName, mode, page);
    }

    @Test
    public void testArtistSearch_NullMode() {
        // Arrange
        String artistName = "John Doe";
        String mode = null;
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(searchMock, times(1)).ArtistSearch(artistName, mode, page);
    }

    @Test
    public void testArtistSearch_NullPage() {
        // Arrange
        String artistName = "John Doe";
        String mode = "album";
        String page = null;
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(searchMock, times(1)).ArtistSearch(artistName, mode, page);
    }

    @Test
    public void testArtistSearch_AllNull() {
        // Arrange
        String artistName = null;
        String mode = null;
        String page = null;
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(searchMock, times(1)).ArtistSearch(artistName, mode, page);
    }
}
