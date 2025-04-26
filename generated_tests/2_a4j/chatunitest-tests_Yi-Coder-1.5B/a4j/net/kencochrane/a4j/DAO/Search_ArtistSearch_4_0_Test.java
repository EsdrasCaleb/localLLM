package net.kencochrane.a4j.DAO;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

@ExtendWith(MockitoExtension.class)
public class Search_ArtistSearch_4_0_Test {

    // Test class
    @Test
    public void ArtistSearchTest() {
        // Arrange
        Search search = new Search();
        String artistName = "";
        String mode = "";
        String page = "";
        // Act
        ProductInfo result = search.ArtistSearch(artistName, mode, page);
        // Assert
        assertNotNull(result);
    }
}
