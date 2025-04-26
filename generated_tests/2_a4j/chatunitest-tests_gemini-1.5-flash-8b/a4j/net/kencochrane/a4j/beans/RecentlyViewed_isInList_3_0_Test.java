package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class RecentlyViewed_isInList_3_0_Test {

    @Test
    void isInList_productExists() {
        List<RecentlyViewed.MiniProduct> products = new ArrayList<>();
        products.add(new RecentlyViewed.MiniProduct("B001234567"));
        RecentlyViewed rv = new RecentlyViewed();
        rv.products = products;
        boolean result = rv.isInList("B001234567");
        assertTrue(result);
    }

    @Test
    void isInList_productDoesNotExist() {
        List<RecentlyViewed.MiniProduct> products = new ArrayList<>();
        products.add(new RecentlyViewed.MiniProduct("B001234567"));
        RecentlyViewed rv = new RecentlyViewed();
        rv.products = products;
        boolean result = rv.isInList("B009876543");
        assertFalse(result);
    }

    @Test
    void isInList_emptyProduct() {
        RecentlyViewed rv = new RecentlyViewed();
        boolean result = rv.isInList("B009876543");
        assertFalse(result);
    }

    @Test
    void isInList_nullProduct() {
        RecentlyViewed rv = new RecentlyViewed();
        rv.products = null;
        boolean result = rv.isInList("B009876543");
        assertFalse(result);
    }

    @Test
    void isInList_nullAsin() {
        List<RecentlyViewed.MiniProduct> products = new ArrayList<>();
        products.add(new RecentlyViewed.MiniProduct(null));
        RecentlyViewed rv = new RecentlyViewed();
        rv.products = products;
        boolean result = rv.isInList("B009876543");
        assertFalse(result);
    }

    @Test
    void isInList_caseInsensitive() {
        List<RecentlyViewed.MiniProduct> products = new ArrayList<>();
        products.add(new RecentlyViewed.MiniProduct("b001234567"));
        RecentlyViewed rv = new RecentlyViewed();
        rv.products = products;
        boolean result = rv.isInList("B001234567");
        assertTrue(result);
    }

    @Test
    void isInList_trim() {
        List<RecentlyViewed.MiniProduct> products = new ArrayList<>();
        products.add(new RecentlyViewed.MiniProduct("  B001234567  "));
        RecentlyViewed rv = new RecentlyViewed();
        rv.products = products;
        boolean result = rv.isInList("B001234567");
        assertTrue(result);
    }

    static class RecentlyViewed {

        List<MiniProduct> products;

        public boolean isInList(String asin) {
            boolean isIn = false;
            if (products != null && products.size() > 0) {
                for (MiniProduct mp : products) {
                    if (mp != null && mp.getAsin() != null && mp.getAsin().trim().equalsIgnoreCase(asin.trim())) {
                        isIn = true;
                        // Crucial: Exit the loop once a match is found
                        break;
                    }
                }
            }
            return isIn;
        }

        static class MiniProduct {

            private String asin;

            public MiniProduct(String asin) {
                this.asin = asin;
            }

            public String getAsin() {
                return asin;
            }
        }
    }
}
