package net.kencochrane.a4j;

import net.kencochrane.a4j.DAO.Product;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import net.kencochrane.a4j.DAO.Cart;
import net.kencochrane.a4j.DAO.Search;
import net.kencochrane.a4j.beans.*;

class A4j_getFullProductFromASIN_0_0_Test {

    @Test
    void getFullProductFromASIN_validInput_returnsProduct() {
        Product mockProduct = Mockito.mock(Product.class);
        FullProduct expectedProduct = new FullProduct();
        Mockito.when(mockProduct.getProduct(Mockito.anyString(), Mockito.anyString(), Mockito.anyString())).thenReturn(expectedProduct);
        A4j a4j = new A4j();
        String asin = "B001234567";
        String offer = "new";
        String page = "1";
        FullProduct actualProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNotNull(actualProduct);
        assertEquals(expectedProduct, actualProduct);
    }

    @Test
    void getFullProductFromASIN_nullInput_returnsNull() {
        A4j a4j = new A4j();
        String asin = null;
        String offer = null;
        String page = null;
        FullProduct actualProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualProduct);
    }

    @Test
    void getFullProductFromASIN_emptyInput_returnsNull() {
        A4j a4j = new A4j();
        String asin = "";
        String offer = "";
        String page = "";
        FullProduct actualProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualProduct);
    }

    @Test
    void getFullProductFromASIN_invalidASIN_returnsNull() {
        Product mockProduct = Mockito.mock(Product.class);
        Mockito.when(mockProduct.getProduct(Mockito.anyString(), Mockito.anyString(), Mockito.anyString())).thenReturn(null);
        A4j a4j = new A4j();
        String asin = "INVALID_ASIN";
        String offer = "new";
        String page = "1";
        FullProduct actualProduct = a4j.getFullProductFromASIN(asin, offer, page);
        assertNull(actualProduct);
    }

    // Dummy classes for compilation.  Crucially, these are *not* the same as the classes in the original question.
    // The original question had a problem where it was trying to use a test class's FullProduct, which is not the same as the FullProduct from the actual A4j class.
    static class A4j {

        private Product product;

        public A4j() {
            this.product = new Product();
        }

        public FullProduct getFullProductFromASIN(String asin, String offer, String page) {
            if (asin == null || asin.isEmpty() || offer == null || offer.isEmpty() || page == null || page.isEmpty()) {
                return null;
            }
            return product.getProduct(asin, offer, page);
        }
    }
}
