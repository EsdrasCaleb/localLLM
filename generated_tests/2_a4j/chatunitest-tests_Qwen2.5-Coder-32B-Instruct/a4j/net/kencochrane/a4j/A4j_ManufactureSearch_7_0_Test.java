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

public class A4j_ManufactureSearch_7_0_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testManufactureSearch() {
        // Arrange
        String manufactureName = "ExampleManufacturer";
        String mode = "ExampleMode";
        String page = "1";
        // Assuming ProductInfo has a default constructor
        ProductInfo expectedProductInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(expectedProductInfo);
        // Act
        ProductInfo result = a4j.ManufactureSearch(manufactureName, mode, page);
        // Assert
        assertNotNull(result, "The result should not be null");
        assertEquals(expectedProductInfo, result, "The result should match the expected ProductInfo object");
        verify(search, times(1)).ManufactureSearch(manufactureName, mode, page);
    }
}
