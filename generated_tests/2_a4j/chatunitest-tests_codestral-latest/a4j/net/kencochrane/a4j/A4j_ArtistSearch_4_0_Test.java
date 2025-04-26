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
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testArtistSearch() {
        String artistName = "Artist";
        String mode = "exact";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ArtistSearch(artistName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo result = a4j.ArtistSearch(artistName, mode, page);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
        verify(search, times(1)).ArtistSearch(artistName, mode, page);
    }
}
