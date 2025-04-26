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

public class A4j_ManufactureSearch_7_2_Test {

    @Test
    public void testManufactureSearch() {
        // Arrange
        A4j a4j = new A4j();
        String manufactureName = "manufactureName";
        String mode = "mode";
        String page = "page";
        Search search = mock(Search.class);
        ProductInfo productInfo = new ProductInfo();
        when(search.ManufactureSearch(manufactureName, mode, page)).thenReturn(productInfo);
        // Act
        ProductInfo result = a4j.ManufactureSearch(manufactureName, mode, page);
        // Assert
        assertEquals(productInfo, result);
    }
}
