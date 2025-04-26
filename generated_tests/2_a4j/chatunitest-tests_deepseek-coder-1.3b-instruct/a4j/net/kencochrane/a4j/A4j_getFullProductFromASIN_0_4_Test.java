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
public class A4j_getFullProductFromASIN_0_4_Test {

    @Mock
    private Product product;

    @InjectMocks
    private A4j a4j;

    @Test
    public void testGetFullProductFromASIN() {
        // Given
        String asin = "1234567890";
        String offer = "offer";
        String page = "page";
        FullProduct expectedProduct = new FullProduct();
        // When
        when(product.getProduct(asin, offer, page)).thenReturn(expectedProduct);
        // Then
        assertEquals(expectedProduct, a4j.getFullProductFromASIN(asin, offer, page));
    }
}
