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

public class Search_ThirdParty_11_1_Test {

    @Mock
    private FileUtil fileUtil;

    @InjectMocks
    private Search search;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testThirdParty_Success() throws Exception {
        String sellerId = "123";
        String type = "type1";
        String page = "1";
        String status = "active";
        FileInputStream mockFileInputStream = mock(FileInputStream.class);
        when(fileUtil.fetchThirdPartySearchFile(sellerId, type, page, status)).thenReturn(mockFileInputStream);
        SellerSearch sellerSearch = new SellerSearch();
        JOXBeanInputStream joxIn = mock(JOXBeanInputStream.class);
        when(joxIn.readObject(SellerSearch.class)).thenReturn(sellerSearch);
        SellerSearch result = search.ThirdParty(sellerId, type, page, status);
        assertNotNull(result);
        assertEquals(sellerSearch, result);
    }

    @Test
    public void testThirdParty_FileNotFound() throws Exception {
        String sellerId = "123";
        String type = "type1";
        String page = "1";
        String status = "active";
        when(fileUtil.fetchThirdPartySearchFile(sellerId, type, page, status)).thenReturn(null);
        SellerSearch result = search.ThirdParty(sellerId, type, page, status);
        assertNull(result);
    }

    @Test
    public void testThirdParty_Exception() throws Exception {
        String sellerId = "123";
        String type = "type1";
        String page = "1";
        String status = "active";
        when(fileUtil.fetchThirdPartySearchFile(sellerId, type, page, status)).thenThrow(new FileNotFoundException());
        SellerSearch result = search.ThirdParty(sellerId, type, page, status);
        assertNull(result);
    }
}
