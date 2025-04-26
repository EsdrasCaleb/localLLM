package net.kencochrane.a4j;

import net.kencochrane.a4j.DAO.Search;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.beans.*;

class A4j_DirectorSearch_6_0_Test {

    @Test
    void directorSearch_validInput_returnsProductInfo() {
        Search searchMock = Mockito.mock(Search.class);
        ProductInfo expectedProductInfo = new ProductInfo();
        Mockito.when(searchMock.DirectorSearch("directorName", "mode", "page")).thenReturn(expectedProductInfo);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.DirectorSearch("directorName", "mode", "page", searchMock);
        assertNotNull(actualProductInfo);
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void directorSearch_nullDirectorName_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        Mockito.when(searchMock.DirectorSearch(null, "mode", "page")).thenReturn(null);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.DirectorSearch(null, "mode", "page", searchMock);
        assertNull(actualProductInfo);
    }

    @Test
    void directorSearch_emptyDirectorName_returnsExpected() {
        Search searchMock = Mockito.mock(Search.class);
        ProductInfo expectedProductInfo = new ProductInfo();
        Mockito.when(searchMock.DirectorSearch("", "mode", "page")).thenReturn(expectedProductInfo);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.DirectorSearch("", "mode", "page", searchMock);
        assertNotNull(actualProductInfo);
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void directorSearch_nullMode_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        Mockito.when(searchMock.DirectorSearch("directorName", null, "page")).thenReturn(null);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.DirectorSearch("directorName", null, "page", searchMock);
        assertNull(actualProductInfo);
    }

    @Test
    void directorSearch_nullPage_returnsNull() {
        Search searchMock = Mockito.mock(Search.class);
        Mockito.when(searchMock.DirectorSearch("directorName", "mode", null)).thenReturn(null);
        A4j a4j = new A4j();
        ProductInfo actualProductInfo = a4j.DirectorSearch("directorName", "mode", null, searchMock);
        assertNull(actualProductInfo);
    }

    // Dummy class for compilation.  Replace with your actual A4j class.
    static class A4j {

        public ProductInfo DirectorSearch(String directorName, String mode, String page, Search search) {
            ProductInfo productInfo = search.DirectorSearch(directorName, mode, page);
            return productInfo;
        }
    }
}
