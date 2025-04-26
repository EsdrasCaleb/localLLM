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
public class A4j_DirectorSearch_6_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    private String directorName;

    private String mode;

    private String page;

    @BeforeEach
    public void setUp() {
        directorName = "Christopher Nolan";
        mode = "exact";
        page = "1";
    }

    @Test
    public void testDirectorSearch() {
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.DirectorSearch(directorName, mode, page)).thenReturn(expectedProductInfo);
        ProductInfo actualProductInfo = a4j.DirectorSearch(directorName, mode, page);
        assertEquals(expectedProductInfo, actualProductInfo);
        verify(search, times(1)).DirectorSearch(directorName, mode, page);
    }
}
