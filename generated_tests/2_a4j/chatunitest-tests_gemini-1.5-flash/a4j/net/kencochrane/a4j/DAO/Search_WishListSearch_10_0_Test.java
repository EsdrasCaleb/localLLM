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
class Search_WishListSearch_10_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private Search searchUnderTest;

    @Test
    void WishListSearch_ValidWishListId_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String wishListId = "12345";
        // Replace with your actual ProductInfo object creation
        net.kencochrane.a4j.beans.ProductInfo expectedProductInfo = new net.kencochrane.a4j.beans.ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        when(genericMethod.invoke(search, "WishlistSearch", wishListId, "mode", "lite", "1", "all")).thenReturn(expectedProductInfo);
        // Act
        net.kencochrane.a4j.beans.ProductInfo actualProductInfo = searchUnderTest.WishListSearch(wishListId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void WishListSearch_NullWishListId_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String wishListId = null;
        // Replace with your actual ProductInfo object creation, potentially a default or empty object
        net.kencochrane.a4j.beans.ProductInfo expectedProductInfo = new net.kencochrane.a4j.beans.ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        when(genericMethod.invoke(search, "WishlistSearch", wishListId, "mode", "lite", "1", "all")).thenReturn(expectedProductInfo);
        // Act
        net.kencochrane.a4j.beans.ProductInfo actualProductInfo = searchUnderTest.WishListSearch(wishListId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void WishListSearch_EmptyWishListId_ReturnsProductInfo() throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        // Arrange
        String wishListId = "";
        // Replace with your actual ProductInfo object creation, potentially a default or empty object
        net.kencochrane.a4j.beans.ProductInfo expectedProductInfo = new net.kencochrane.a4j.beans.ProductInfo();
        Method genericMethod = Search.class.getDeclaredMethod("Generic", String.class, String.class, String.class, String.class, String.class, String.class);
        genericMethod.setAccessible(true);
        when(genericMethod.invoke(search, "WishlistSearch", wishListId, "mode", "lite", "1", "all")).thenReturn(expectedProductInfo);
        // Act
        net.kencochrane.a4j.beans.ProductInfo actualProductInfo = searchUnderTest.WishListSearch(wishListId);
        // Assert
        assertEquals(expectedProductInfo, actualProductInfo);
    }
}
