package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class RecentlyViewed_addProduct_0_1_Test {

    @Test
    void addProduct_validProduct_addedToList() {
        RecentlyViewed recentlyViewed = new RecentlyViewed();
        MiniProduct miniProd = new MiniProduct("asin123");
        recentlyViewed.addProduct(miniProd);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    @Test
    void addProduct_nullProduct_notAdded() {
        RecentlyViewed recentlyViewed = new RecentlyViewed();
        MiniProduct miniProd = null;
        recentlyViewed.addProduct(miniProd);
        assertEquals(0, recentlyViewed.getNumProducts());
    }

    @Test
    void addProduct_duplicateProduct_notAdded() {
        RecentlyViewed recentlyViewed = new RecentlyViewed();
        MiniProduct miniProd = new MiniProduct("asin123");
        recentlyViewed.addProduct(miniProd);
        recentlyViewed.addProduct(miniProd);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    @Test
    void addProduct_validProduct_alreadyInList_notAdded() {
        RecentlyViewed recentlyViewed = new RecentlyViewed();
        MiniProduct miniProd = new MiniProduct("asin123");
        recentlyViewed.addProduct(miniProd);
        recentlyViewed.addProduct(miniProd);
        assertEquals(1, recentlyViewed.getNumProducts());
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

    static class RecentlyViewed {

        private List<MiniProduct> products = new ArrayList<>();

        public void addProduct(MiniProduct product) {
            if (product != null && !products.contains(product)) {
                products.add(product);
            }
        }

        public int getNumProducts() {
            return products.size();
        }
    }
}
