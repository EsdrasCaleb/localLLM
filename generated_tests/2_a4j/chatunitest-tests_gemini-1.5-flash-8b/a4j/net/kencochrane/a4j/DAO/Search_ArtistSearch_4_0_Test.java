package net.kencochrane.a4j.DAO;

import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.ProductInfo;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

class Search_ArtistSearch_4_0_Test {

    @Test
    void artistSearch_validInput_returnsProductInfo() {
        Search search = Mockito.mock(Search.class);
        // Create a real ProductInfo object
        ProductInfo mockProductInfo = new ProductInfo();
        Mockito.when(search.Generic("ArtistSearch", "artistName", "mode", "lite", "page1", "all")).thenReturn(mockProductInfo);
        // Use a real Search object
        Search realSearch = new Search();
        ProductInfo result = realSearch.ArtistSearch("artistName", "mode", "page1");
        assertNotNull(result);
        Mockito.verify(search).Generic("ArtistSearch", "artistName", "mode", "lite", "page1", "all");
    }

    @Test
    void artistSearch_nullArtistName_returnsNull() {
        Search search = new Search();
        ProductInfo result = search.ArtistSearch(null, "mode", "page1");
        assertNull(result);
    }

    @Test
    void artistSearch_emptyArtistName_returnsNull() {
        Search search = new Search();
        ProductInfo result = search.ArtistSearch("", "mode", "page1");
        assertNull(result);
    }

    @Test
    void artistSearch_nullMode_returnsNull() {
        Search search = new Search();
        ProductInfo result = search.ArtistSearch("artistName", null, "page1");
        assertNull(result);
    }

    @Test
    void artistSearch_nullPage_returnsNull() {
        Search search = new Search();
        ProductInfo result = search.ArtistSearch("artistName", "mode", null);
        assertNull(result);
    }
}
