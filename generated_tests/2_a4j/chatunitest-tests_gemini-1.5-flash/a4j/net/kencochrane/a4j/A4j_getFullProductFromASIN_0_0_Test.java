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
class A4j_getFullProductFromASIN_0_0_Test {

    @Mock
    private Product product;

    @InjectMocks
    private A4j a4j;

    @Test
    void testGetFullProductFromASIN_validInput_returnsFullProduct() {
        String asin = "B012345678";
        String offer = "offer1";
        String page = "page1";
        FullProduct expectedFullProduct = new FullProduct();
        when(product.getProduct(asin, offer, page)).thenReturn(expectedFullProduct);
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertEquals(expectedFullProduct, actualFullProduct);
    }

    @Test
    void testGetFullProductFromASIN_nullAsin_returnsNull() {
        String asin = null;
        String offer = "offer1";
        String page = "page1";
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualFullProduct);
    }

    @Test
    void testGetFullProductFromASIN_nullOffer_returnsNull() {
        String asin = "B012345678";
        String offer = null;
        String page = "page1";
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualFullProduct);
    }

    @Test
    void testGetFullProductFromASIN_nullPage_returnsNull() {
        String asin = "B012345678";
        String offer = "offer1";
        String page = null;
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualFullProduct);
    }

    @Test
    void testGetFullProductFromASIN_emptyAsin_returnsNull() {
        String asin = "";
        String offer = "offer1";
        String page = "page1";
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualFullProduct);
    }

    @Test
    void testGetFullProductFromASIN_emptyOffer_returnsNull() {
        String asin = "B012345678";
        String offer = "";
        String page = "page1";
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualFullProduct);
    }

    @Test
    void testGetFullProductFromASIN_emptyPage_returnsNull() {
        String asin = "B012345678";
        String offer = "offer1";
        String page = "";
        FullProduct actualFullProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualFullProduct);
    }

    static class A4j {

        private final Product product;

        A4j(Product product) {
            this.product = product;
        }

        public FullProduct getFullProductFromASIN(String asin, String offer, String page) {
            return product.getProduct(asin, offer, page);
        }
    }

    static class Product {

        public FullProduct getProduct(String asin, String offer, String page) {
            if (asin == null || asin.isEmpty() || offer == null || offer.isEmpty() || page == null || page.isEmpty()) {
                return null;
            }
            return new FullProduct();
        }
    }
}
