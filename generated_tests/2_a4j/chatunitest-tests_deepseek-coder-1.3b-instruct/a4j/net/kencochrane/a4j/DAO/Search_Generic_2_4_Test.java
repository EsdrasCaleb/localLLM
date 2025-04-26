package net.kencochrane.a4j.DAO;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.FileInputStream;
import java.lang.reflect.Field;
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
public class Search_Generic_2_4_Test {

    @Mock
    private FileUtil fileUtil;

    @Mock
    private JOXBeanInputStream joxIn;

    @InjectMocks
    private Search search;

    @Test
    public void testGeneric() throws Exception {
        // Arrange
        String searchType = "type";
        String searchTerm = "term";
        String mode = "mode";
        String type = "product";
        String page = "1";
        String offer = "offer";
        FileInputStream fileIn = mock(FileInputStream.class);
        ProductInfo productInfo = new ProductInfo();
        when(fileUtil.fetchGenericSearchFile(searchType, searchTerm, mode, type, page, offer)).thenReturn(fileIn);
        when(joxIn.readObject(ProductInfo.class)).thenReturn(productInfo);
        // Act
        ProductInfo result = search.Generic(searchType, searchTerm, mode, type, page, offer);
        // Assert
        assertEquals(productInfo, result);
    }
}
