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

class A4j_UpcSearch_8_1_Test {

    private A4j a4j;

    private Search mockSearch;

    @BeforeEach
    public void setUp() {
        a4j = new A4j();
        mockSearch = mock(Search.class);
        when(mockSearch.UpcSearch(anyString(), anyString(), anyString())).thenReturn(new ProductInfo());
    }

    @Test
    public void testUpcSearch() {
        // Call the focal method
        ProductInfo result = a4j.UpcSearch("1234567890", "all", "1");
        // Verify the result
        assertEquals(result, mockSearch.UpcSearch("1234567890", "all", "1"));
    }
}
