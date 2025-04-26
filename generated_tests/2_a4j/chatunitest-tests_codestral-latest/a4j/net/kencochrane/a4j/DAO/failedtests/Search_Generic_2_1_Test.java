package net.kencochrane.a4j.DAO;

import java.io.FileInputStream;
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

class Search_Generic_2_1_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Search search;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testGeneric_Success() throws IOException, ClassNotFoundException {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        FileInputStream mockFileInputStream = mock(FileInputStream.class);
        JOXBeanInputStream mockJOXBeanInputStream = mock(JOXBeanInputStream.class);
        ProductInfo mockProductInfo = mock(ProductInfo.class);
        when(fileUtil.fetchGenericSearchFile(searchType, searchTerm, mode, type, page, offer)).thenReturn(mockFileInputStream);
        when(mockJOXBeanInputStream.readObject(ProductInfo.class)).thenReturn(mockProductInfo);
        ProductInfo result = search.Generic(searchType, searchTerm, mode, type, page, offer);
        assertNotNull(result);
        assertEquals(mockProductInfo, result);
    }

    @Test
    void testGeneric_FileInputNull() throws IOException {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        when(fileUtil.fetchGenericSearchFile(searchType, searchTerm, mode, type, page, offer)).thenReturn(null);
        ProductInfo result = search.Generic(searchType, searchTerm, mode, type, page, offer);
        assertNull(result);
    }

    @Test
    void testGeneric_Exception() throws IOException {
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "type";
        String page = "page";
        String offer = "offer";
        when(fileUtil.fetchGenericSearchFile(searchType, searchTerm, mode, type, page, offer)).thenThrow(new IOException());
        ProductInfo result = search.Generic(searchType, searchTerm, mode, type, page, offer);
        assertNull(result);
    }
}
