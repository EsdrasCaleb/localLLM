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
public class Search_Keyword_1_1_Test {

    // JUnit class
    @Test
    public void testKeyword() {
        Search search = new Search();
        ProductInfo productInfo = search.Keyword("test", "Samsung", "Phone", "1");
        assertNotNull(productInfo);
    }
}
