package net.kencochrane.a4j;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Product;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

@ExtendWith(MockitoExtension.class)
public class A4j_UpcSearch_8_4_Test {

    @Mock
    private Search search;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testUpcSearch() {
        // Given
        String upc = "1234567890";
        String mode = "mode";
        String page = "page";
        ProductInfo productInfoMock = new ProductInfo();
        when(search.UpcSearch(upc, mode, page)).thenReturn(productInfoMock);
        // When
        ProductInfo result = a4j.UpcSearch(upc, mode, page);
        // Then
        assertEquals(productInfoMock, result);
    }
}
