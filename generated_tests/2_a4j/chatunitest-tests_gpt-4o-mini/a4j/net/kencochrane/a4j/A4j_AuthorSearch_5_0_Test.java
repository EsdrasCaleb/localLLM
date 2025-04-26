package net.kencochrane.a4j;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

class A4j_AuthorSearch_5_0_Test {

    private A4j a4j;

    private Search searchMock;

    @BeforeEach
    void setUp() {
        a4j = new A4j();
        searchMock = mock(Search.class);
    }

    @Test
    void testAuthorSearch_ValidInput() {
        String authorName = "John Doe";
        String page = "1";
        // Assume this is properly initialized
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Use reflection to set the mock Search object into the A4j instance
        try {
            java.lang.reflect.Field field = A4j.class.getDeclaredField("search");
            field.setAccessible(true);
            field.set(a4j, searchMock);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        ProductInfo actualProductInfo = a4j.AuthorSearch(authorName, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(searchMock).AuthorSearch(authorName, page);
    }

    @Test
    void testAuthorSearch_NullAuthorName() {
        String authorName = null;
        String page = "1";
        // Assume this is properly initialized
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Use reflection to set the mock Search object into the A4j instance
        try {
            java.lang.reflect.Field field = A4j.class.getDeclaredField("search");
            field.setAccessible(true);
            field.set(a4j, searchMock);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        ProductInfo actualProductInfo = a4j.AuthorSearch(authorName, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(searchMock).AuthorSearch(authorName, page);
    }

    @Test
    void testAuthorSearch_EmptyPage() {
        String authorName = "Jane Doe";
        String page = "";
        // Assume this is properly initialized
        ProductInfo expectedProductInfo = new ProductInfo();
        when(searchMock.AuthorSearch(authorName, page)).thenReturn(expectedProductInfo);
        // Use reflection to set the mock Search object into the A4j instance
        try {
            java.lang.reflect.Field field = A4j.class.getDeclaredField("search");
            field.setAccessible(true);
            field.set(a4j, searchMock);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
        ProductInfo actualProductInfo = a4j.AuthorSearch(authorName, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(searchMock).AuthorSearch(authorName, page);
    }
}
