package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class RecentlyViewed_addProduct_0_0_Test {

    private RecentlyViewed recentlyViewed;

    @BeforeEach
    public void setUp() {
        recentlyViewed = new RecentlyViewed();
    }

    @Test
    public void testAddProduct() {
        MiniProduct miniProd = Mockito.mock(MiniProduct.class);
        recentlyViewed.addProduct(miniProd);
        // Verify that the product was added to the ArrayList
        ArrayList<MiniProduct> products = recentlyViewed.getProducts();
        assertNotNull(products);
        assertEquals(1, products.size());
        assertTrue(products.contains(miniProd));
        // Verify that the product was not added if it's not in the list
        MiniProduct notInList = Mockito.mock(MiniProduct.class);
        recentlyViewed.addProduct(notInList);
        // Verify that the product was not added
        products = recentlyViewed.getProducts();
        assertNull(products);
    }
}
