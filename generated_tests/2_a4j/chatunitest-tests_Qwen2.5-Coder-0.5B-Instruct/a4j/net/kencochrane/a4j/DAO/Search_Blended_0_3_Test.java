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

class Search_Blended_0_3_Test {

    @Test
    public void testBlended() throws Exception {
        Search search = new Search();
        String searchTerm = "example";
        String type = "type";
        BlendedSearch expectedResult = new BlendedSearch();
        // Arrange
        FileUtil fileUtil = new FileUtil();
        JOXBeanInputStream joxIn = null;
        BlendedSearch testBean = new BlendedSearch();
        try {
            FileInputStream fileIn = fileUtil.fetchBlendedSearchFile(searchTerm, type);
            if (fileIn != null) {
                joxIn = new JOXBeanInputStream(fileIn);
                testBean = (BlendedSearch) joxIn.readObject(BlendedSearch.class);
            } else {
                // log.debug("Error no fileInput");
                testBean = null;
            }
        } catch (Exception exc) {
            exc.printStackTrace();
        }
        // Act
        BlendedSearch result = search.Blended(searchTerm, type);
        // Assert
        assertEquals(expectedResult, result);
    }
}
