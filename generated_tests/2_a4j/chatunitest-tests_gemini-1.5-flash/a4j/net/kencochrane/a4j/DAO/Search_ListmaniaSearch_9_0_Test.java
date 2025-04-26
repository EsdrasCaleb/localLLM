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
class Search_ListmaniaSearch_9_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private Search searchUnderTest = new Search();

    @Test
    void ListmaniaSearch_ValidListId_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String listId = "12345";
        // Replace with your actual ProductInfo object creation
        net.kencochrane.a4j.beans.ProductInfo expectedProductInfo = new net.kencochrane.a4j.beans.ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        when(genericMethod.invoke(search, "ListManiaSearch", listId, "mode", "lite", "1", "all")).thenReturn(expectedProductInfo);
        // Act
        net.kencochrane.a4j.beans.ProductInfo actualProductInfo = searchUnderTest.ListmaniaSearch(listId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void ListmaniaSearch_NullListId_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String listId = null;
        // Replace with your actual ProductInfo object creation, potentially a default or error state.
        net.kencochrane.a4j.beans.ProductInfo expectedProductInfo = new net.kencochrane.a4j.beans.ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        when(genericMethod.invoke(search, "ListManiaSearch", listId, "mode", "lite", "1", "all")).thenReturn(expectedProductInfo);
        // Act
        net.kencochrane.a4j.beans.ProductInfo actualProductInfo = searchUnderTest.ListmaniaSearch(listId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void ListmaniaSearch_EmptyListId_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String listId = "";
        // Replace with your actual ProductInfo object creation, potentially a default or error state.
        net.kencochrane.a4j.beans.ProductInfo expectedProductInfo = new net.kencochrane.a4j.beans.ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        when(genericMethod.invoke(search, "ListManiaSearch", listId, "mode", "lite", "1", "all")).thenReturn(expectedProductInfo);
        // Act
        net.kencochrane.a4j.beans.ProductInfo actualProductInfo = searchUnderTest.ListmaniaSearch(listId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
