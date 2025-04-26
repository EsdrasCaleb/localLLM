package net.kencochrane.a4j.DAO;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
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

public class Search_SimilaritesSearch_12_0_Test {

    @Test
    public void testSimilaritesSearch() throws IOException {
        // Arrange
        Search search = new Search();
        String asin = "1234567890";
        String page = "1";
        FileInputStream fileIn = Mockito.mock(FileInputStream.class);
        when(fileIn.read()).thenReturn(-1);
        when(fileIn.available()).thenReturn(0);
        // Act
        ProductInfo productInfo = search.SimilaritesSearch(asin, page);
        // Assert
        assertEquals(null, productInfo);
    }
}
