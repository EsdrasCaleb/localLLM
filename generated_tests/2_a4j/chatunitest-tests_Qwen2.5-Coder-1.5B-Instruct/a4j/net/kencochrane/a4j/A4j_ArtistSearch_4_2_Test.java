package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

class A4j_ArtistSearch_4_2_Test {

    @Mock
    private Search searchMock;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testArtistSearch() {
        // Arrange
        String artistName = "John Doe";
        String mode = "Albums";
        String page = "1";
        // Stub the ArtistSearch method to return a predefined ProductInfo object
        ProductInfo expectedResult = new ProductInfo();
        when(searchMock.ArtistSearch(artistName, mode, page)).thenReturn(expectedResult);
        // Call the ArtistSearch method on the A4j class instance
        ProductInfo result = new A4j().ArtistSearch(artistName, mode, page);
        // Assert
        assertEquals(expectedResult, result);
    }
}
