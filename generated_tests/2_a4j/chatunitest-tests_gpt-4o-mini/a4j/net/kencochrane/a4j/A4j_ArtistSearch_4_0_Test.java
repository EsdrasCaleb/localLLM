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

class A4j_ArtistSearch_4_0_Test {

    private A4j a4j;

    private Search searchMock;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        searchMock = mock(Search.class);
    }

    @Test
    void testArtistSearch_ValidInputs() {
        // Arrange
        String artistName = "Artist Name";
        String mode = "mode";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void testArtistSearch_EmptyArtistName() {
        // Arrange
        String artistName = "";
        String mode = "mode";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void testArtistSearch_NullArtistName() {
        // Arrange
        String artistName = null;
        String mode = "mode";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void testArtistSearch_InvalidMode() {
        // Arrange
        String artistName = "Artist Name";
        String mode = "invalidMode";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void testArtistSearch_NegativePage() {
        // Arrange
        String artistName = "Artist Name";
        String mode = "mode";
        String page = "-1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo actualProductInfo = a4j.ArtistSearch(artistName, mode, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
