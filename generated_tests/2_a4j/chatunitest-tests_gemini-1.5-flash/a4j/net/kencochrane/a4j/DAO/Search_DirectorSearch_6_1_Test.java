package net.kencochrane.a4j.DAO;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import com.wutka.jox.JOXBeanInputStream;
import net.kencochrane.a4j.beans.BlendedSearch;
import net.kencochrane.a4j.beans.ProductInfo;
import net.kencochrane.a4j.beans.SellerSearch;
import net.kencochrane.a4j.file.FileUtil;
import java.io.FileInputStream;

class Search_DirectorSearch_6_1_Test {

    @Mock
    private Search search;

    @InjectMocks
    private Search searchUnderTest = new Search();

    @Test
    void DirectorSearch_EmptyMode_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String directorName = "Christopher Nolan";
        String mode = "";
        String page = "1";
        ProductInfo expectedProductInfo = new ProductInfo();
    }
}
