package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_ManufactureSearch_7_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testManufactureSearch() {
        String manufactureName = "TestManufacture";
        String mode = "TestMode";
        String page = "TestPage";
        // Initialize with expected values
        ProductInfo expected = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expected);
        ProductInfo actual = a4j.ManufactureSearch(manufactureName, mode, page);
        assertEquals(expected, actual);
    }
}
