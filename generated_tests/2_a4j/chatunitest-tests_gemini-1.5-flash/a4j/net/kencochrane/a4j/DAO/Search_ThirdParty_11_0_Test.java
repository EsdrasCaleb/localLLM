package net.kencochrane.a4j.DAO;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileInputStream;
import java.io.InputStream;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;

@ExtendWith(MockitoExtension.class)
class Search_ThirdParty_11_0_Test {

    @Mock
    FileUtil fileUtil;

    @InjectMocks
    Search search;

    @Test
    void ThirdParty_fileDoesNotExist_returnsNull() throws Exception {
        when(fileUtil.fetchThirdPartySearchFile(anyString(), anyString(), anyString(), anyString())).thenReturn(null);
        SellerSearch result = search.ThirdParty("123", "type", "1", "active");
        assertNull(result);
        verify(fileUtil).fetchThirdPartySearchFile("123", "type", "1", "active");
    }

    static class FileUtil {

        FileInputStream fetchThirdPartySearchFile(String sellerId, String type, String page, String status) {
            return null;
        }
    }

    static class Search {

        private FileUtil fileUtil;

        public Search(FileUtil fileUtil) {
            this.fileUtil = fileUtil;
        }

        public Search() {
        }

        public SellerSearch ThirdParty(String sellerId, String type, String page, String status) {
            FileInputStream fis = fileUtil.fetchThirdPartySearchFile(sellerId, type, page, status);
            if (fis == null)
                return null;
            try (JOXBeanInputStream joxIn = new JOXBeanInputStream(fis)) {
                return (SellerSearch) joxIn.readObject(SellerSearch.class);
            } catch (Exception e) {
                return null;
            }
        }
    }

    static class JOXBeanInputStream implements AutoCloseable {

        InputStream in;

        public JOXBeanInputStream(InputStream in) {
            this.in = in;
        }

        Object readObject(Class<?> cls) throws Exception {
            throw new Exception();
        }

        @Override
        public void close() throws Exception {
            in.close();
        }
    }

    static class SellerSearch {
    }
}
