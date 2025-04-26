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

    @Test
    public void testArtistSearch() {
        // Given
        String artistName = "testArtist";
        String mode = "testMode";
        String page = "testPage";
        ProductInfo expectedProductInfo = new ProductInfo();
        // Mock the Search class
        Search search = Mockito.mock(Search.class);
        Mockito.when(search.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        // Create an instance of A4j
        A4j a4j = new A4j();
        // When
        ProductInfo actualProductInfo = a4j.ArtistSearch(artistName, mode, page);
        // Then
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
