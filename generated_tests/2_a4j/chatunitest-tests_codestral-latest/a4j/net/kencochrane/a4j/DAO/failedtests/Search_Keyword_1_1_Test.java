package net.kencochrane.a4j.DAO;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
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

public class Search_Keyword_1_1_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Search search;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testKeyword_FileFound() throws Exception {
        String searchTerm = "example";
        String productLine = "line";
        String type = "type";
        String page = "1";
        FileInputStream mockFileInputStream = mock(FileInputStream.class);
        when(fileUtil.fetchKeywordSearchFile(searchTerm, productLine, type, page)).thenReturn(mockFileInputStream);
        ProductInfo expectedProductInfo = new ProductInfo();
        JOXBeanInputStream joxIn = mock(JOXBeanInputStream.class);
        when(joxIn.readObject(ProductInfo.class)).thenReturn(expectedProductInfo);
        ProductInfo result = search.Keyword(searchTerm, productLine, type, page);
        assertNotNull(result);
        assertEquals(expectedProductInfo, result);
    }

    @Test
    public void testKeyword_FileNotFound() throws Exception {
        String searchTerm = "example";
        String productLine = "line";
        String type = "type";
        String page = "1";
        when(fileUtil.fetchKeywordSearchFile(searchTerm, productLine, type, page)).thenReturn(null);
        ProductInfo result = search.Keyword(searchTerm, productLine, type, page);
        assertNull(result);
    }

    @Test
    public void testKeyword_ExceptionThrown() throws Exception {
        String searchTerm = "example";
        String productLine = "line";
        String type = "type";
        String page = "1";
        when(fileUtil.fetchKeywordSearchFile(searchTerm, productLine, type, page)).thenThrow(new FileNotFoundException());
        ProductInfo result = search.Keyword(searchTerm, productLine, type, page);
        assertNull(result);
    }
}
