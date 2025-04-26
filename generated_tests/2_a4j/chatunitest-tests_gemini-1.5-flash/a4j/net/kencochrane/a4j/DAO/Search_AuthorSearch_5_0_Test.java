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

@ExtendWith(MockitoExtension.class)
class Search_AuthorSearch_5_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private Search searchUnderTest;

    @Test
    void AuthorSearch_ValidInput_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String authorName = "Stephen King";
        String page = "1";
        // Corrected: Use the correct import
        ProductInfo expectedProductInfo = new ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        // Corrected: Cast the return value and use 'search' instead of 'searchUnderTest'
        when((ProductInfo) genericMethod.invoke(search, "AuthorSearch", authorName, "books", "lite", page, "all")).thenReturn(expectedProductInfo);
        // Act
        // Corrected: Use the correct import
        ProductInfo actualProductInfo = searchUnderTest.AuthorSearch(authorName, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void AuthorSearch_NullAuthorName_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String authorName = null;
        String page = "1";
        // Corrected: Use the correct import
        ProductInfo expectedProductInfo = new ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        // Corrected: Cast the return value and use 'search' instead of 'searchUnderTest'
        when((ProductInfo) genericMethod.invoke(search, "AuthorSearch", authorName, "books", "lite", page, "all")).thenReturn(expectedProductInfo);
        // Act
        // Corrected: Use the correct import
        ProductInfo actualProductInfo = searchUnderTest.AuthorSearch(authorName, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void AuthorSearch_EmptyAuthorName_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String authorName = "";
        String page = "1";
        // Corrected: Use the correct import
        ProductInfo expectedProductInfo = new ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        // Corrected: Cast the return value and use 'search' instead of 'searchUnderTest'
        when((ProductInfo) genericMethod.invoke(search, "AuthorSearch", authorName, "books", "lite", page, "all")).thenReturn(expectedProductInfo);
        // Act
        // Corrected: Use the correct import
        ProductInfo actualProductInfo = searchUnderTest.AuthorSearch(authorName, page);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
