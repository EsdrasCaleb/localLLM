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
public class Search_Generic_2_1_Test {

    @Test
    public void testGeneric() {
        // Arrange
        FileUtil fileUtil = new FileUtil();
        JOXBeanInputStream joxIn = null;
        ProductInfo productInfo = new ProductInfo();
        // Act
        try {
            FileInputStream fileIn = fileUtil.fetchGenericSearchFile("searchType", "searchTerm", "mode", "type", "page", "offer");
            if (fileIn != null) {
                joxIn = new JOXBeanInputStream(fileIn);
                productInfo = (ProductInfo) joxIn.readObject(ProductInfo.class);
            } else {
                // log.debug("Error no fileInput");
                productInfo = null;
            }
        } catch (Exception exc) {
            exc.printStackTrace();
        }
        // Assert
        assertNotNull(productInfo);
    }
}
