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

class A4j_ManufactureSearch_7_1_Test {

    @InjectMocks
    private A4j a4j;

    @Mock
    private Search search;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testManufactureSearch_ValidInput() {
        String manufactureName = "TestManufacturer";
        String mode = "normal";
        String page = "1";
        // Assuming a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.ManufactureSearch(manufactureName, mode, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search).ManufactureSearch(manufactureName, mode, page);
    }

    @Test
    void testManufactureSearch_EmptyManufactureName() {
        String manufactureName = "";
        String mode = "normal";
        String page = "1";
        // Assuming a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.ManufactureSearch(manufactureName, mode, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search).ManufactureSearch(manufactureName, mode, page);
    }

    @Test
    void testManufactureSearch_NullManufactureName() {
        String manufactureName = null;
        String mode = "normal";
        String page = "1";
        // Assuming a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.ManufactureSearch(manufactureName, mode, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search).ManufactureSearch(manufactureName, mode, page);
    }

    @Test
    void testManufactureSearch_InvalidMode() {
        String manufactureName = "TestManufacturer";
        String mode = "invalidMode";
        String page = "1";
        // Assuming a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.ManufactureSearch(manufactureName, mode, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search).ManufactureSearch(manufactureName, mode, page);
    }

    @Test
    void testManufactureSearch_NullPage() {
        String manufactureName = "TestManufacturer";
        String mode = "normal";
        String page = null;
        // Assuming a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.ManufactureSearch(manufactureName, mode, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search).ManufactureSearch(manufactureName, mode, page);
    }
}
