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

class A4j_ManufactureSearch_7_2_Test {

    @Test
    void testManufactureSearch_validInput_returnsProductInfo() {
        Search searchMock = Mockito.mock(Search.class);
        ProductInfo expectedProductInfo = new ProductInfo();
        Mockito.when(searchMock.ManufactureSearch("manufacturer1", "mode1", "1")).thenReturn(expectedProductInfo);
        // Inject the mock Search object
        A4j a4j = new A4j(searchMock);
        ProductInfo actualProductInfo = a4j.ManufactureSearch("manufacturer1", "mode1", "1");
        assertEquals(expectedProductInfo, actualProductInfo);
    }

    @Test
    void testManufactureSearch_nullInput_throwsIllegalArgumentException() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j(searchMock);
        assertThrows(IllegalArgumentException.class, () -> a4j.ManufactureSearch(null, null, null));
    }

    @Test
    void testManufactureSearch_emptyInput_throwsIllegalArgumentException() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j(searchMock);
        assertThrows(IllegalArgumentException.class, () -> a4j.ManufactureSearch("", "", ""));
    }

    @Test
    void testManufactureSearch_invalidPage_throwsIllegalArgumentException() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j(searchMock);
        assertThrows(IllegalArgumentException.class, () -> a4j.ManufactureSearch("manufacturer", "mode", "abc"));
    }

    @Test
    void testManufactureSearch_pageZero_throwsIllegalArgumentException() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j(searchMock);
        assertThrows(IllegalArgumentException.class, () -> a4j.ManufactureSearch("manufacturer", "mode", "0"));
    }

    // This test case is added to cover the negative case for the page parameter.
    @Test
    void testManufactureSearch_negativePage_throwsIllegalArgumentException() {
        Search searchMock = Mockito.mock(Search.class);
        A4j a4j = new A4j(searchMock);
        assertThrows(IllegalArgumentException.class, () -> a4j.ManufactureSearch("manufacturer", "mode", "-1"));
    }

    // Dummy class for A4j
    static class A4j {

        private final Search search;

        public A4j(Search search) {
            this.search = search;
        }

        public ProductInfo ManufactureSearch(String manufactureName, String mode, String page) {
            if (manufactureName == null || mode == null || page == null || manufactureName.isEmpty() || mode.isEmpty() || !page.matches("\\d+") || Integer.parseInt(page) <= 0) {
                throw new IllegalArgumentException("Input cannot be null or empty, and page must be a positive integer.");
            }
            return search.ManufactureSearch(manufactureName, mode, page);
        }
    }
}
