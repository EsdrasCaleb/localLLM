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
public class A4j_ArtistSearch_4_1_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    public void artistSearchTest() {
        // Given
        String artistName = "testArtist";
        String mode = "testMode";
        String page = "testPage";
        ProductInfo productInfo = new ProductInfo();
        when(search.ArtistSearch(artistName, mode, page)).thenReturn(productInfo);
        // When
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        // Then
        assertEquals(productInfo, result);
    }
}
