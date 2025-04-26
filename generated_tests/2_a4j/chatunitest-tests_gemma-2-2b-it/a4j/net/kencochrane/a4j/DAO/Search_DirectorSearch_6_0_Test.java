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

public class Search_DirectorSearch_6_0_Test {

    @Test
    void DirectorSearch_ValidInput() {
        // Arrange
        Search search = new Search();
        String directorName = "Christopher Nolan";
        String mode = "all";
        String page = "1";
        // Act
        ProductInfo result = search.DirectorSearch(directorName, mode, page);
        // Assert
        assertNotNull(result);
    }
}
