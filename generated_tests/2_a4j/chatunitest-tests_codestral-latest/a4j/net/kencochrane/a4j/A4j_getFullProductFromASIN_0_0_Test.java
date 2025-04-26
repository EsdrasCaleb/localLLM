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

public class A4j_getFullProductFromASIN_0_0_Test {

    @Mock
    private Product product;

    @InjectMocks
    private A4j a4j;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testGetFullProductFromASIN() {
        // Given
        String asin = "B001234567";
        String offer = "offer1";
        String page = "page1";
        FullProduct expectedFullProduct = new FullProduct();
        when(product.getProduct(asin, offer, page)).thenReturn(expectedFullProduct);
        // When
        FullProduct result = a4j.getFullProductFromASIN(asin, offer, page);
        // Then
        assertNotNull(result);
        assertEquals(expectedFullProduct, result);
        verify(product, times(1)).getProduct(asin, offer, page);
    }
}
