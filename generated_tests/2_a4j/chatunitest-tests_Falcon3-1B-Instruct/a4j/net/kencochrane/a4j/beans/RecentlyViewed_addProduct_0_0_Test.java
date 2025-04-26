package net.kencochrane.a4j.beans;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class RecentlyViewed_addProduct_0_0_Test {

    @Test
    public void testAddProduct() {
        // Create a new MiniProduct
        MiniProduct newMiniProduct = new MiniProduct();
        // Invoke the addProduct method
        RecentlyViewed recentlyViewed = new RecentlyViewed();
        recentlyViewed.addProduct(newMiniProduct);
        // Verify the method was called
        // Assuming 1 added product initially
        assertEquals(1, recentlyViewed.getNumProducts());
        // Add a product that should not be in the list
        recentlyViewed.addProduct(newMiniProduct);
        // Verify the method was called again
        // 2 products should be present now
        assertEquals(2, recentlyViewed.getNumProducts());
        // Add a product that already exists
        recentlyViewed.addProduct(newMiniProduct);
        // Verify the method was called again
        // 1 product should be present now
        assertEquals(1, recentlyViewed.getNumProducts());
        // Add a product that is not in the list
        recentlyViewed.addProduct(newMiniProduct);
        // Verify the method was called again
        // 0 products should be present now
        assertEquals(0, recentlyViewed.getNumProducts());
    }
}
