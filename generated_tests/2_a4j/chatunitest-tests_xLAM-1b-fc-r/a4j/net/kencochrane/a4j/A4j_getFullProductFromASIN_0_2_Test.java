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

public class A4j_getFullProductFromASIN_0_2_Test {

    @Test
    public void testGetFullProductFromASIN() {
        // Given
        A4j a4j = new A4j();
        String asin = "1234567890";
        String offer = "offer1";
        String page = "page1";
        Product mockProduct = Mockito.mock(Product.class);
        FullProduct expectedFullProduct = new FullProduct();
        // When
        Mockito.when(mockProduct.getProduct(asin, offer, page)).thenReturn(expectedFullProduct);
        // Then
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertEquals(expectedFullProduct, actualFullProduct);
    }
}
