package net.kencochrane.a4j.beans;

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
        RecentlyViewed rv = new RecentlyViewed();
        MiniProduct mp = new MiniProduct();
        mp.setAsin("12345");
        rv.addProduct(mp);
        assertTrue(rv.getProducts().contains(mp));
    }
}
