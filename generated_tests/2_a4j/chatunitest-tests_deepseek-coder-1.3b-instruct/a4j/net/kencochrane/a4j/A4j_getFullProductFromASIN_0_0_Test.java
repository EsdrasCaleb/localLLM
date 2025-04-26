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
public class A4j_getFullProductFromASIN_0_0_Test {

    @Mock
    private Product product;

    @InjectMocks
    private A4j a4j;

    @Test
    public void getFullProductFromASINTest() {
        // given
        String asin = "testAsin";
        String offer = "testOffer";
        String page = "testPage";
        FullProduct expected = new FullProduct();
        when(product.getProduct(asin, offer, page)).thenReturn(expected);
        // when
        FullProduct result = a4j.getFullProductFromASIN(asin, offer, page);
        // then
        assertEquals(expected, result);
    }
}
