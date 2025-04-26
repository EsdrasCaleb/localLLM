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

    private String manufactureName;

    private String mode;

    private String page;

    private ProductInfo productInfo;

    @BeforeEach
    public void setUp() {
        manufactureName = "TestManufacturer";
        mode = "TestMode";
        page = "1";
        productInfo = new ProductInfo();
    }

    @Test
    public void testManufactureSearch() {
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(productInfo);
        ProductInfo result = a4j.ManufactureSearch(manufactureName, mode, page);
        assertNotNull(result);
        assertEquals(productInfo, result);
        verify(search, times(1)).ManufactureSearch(manufactureName, mode, page);
    }
}
