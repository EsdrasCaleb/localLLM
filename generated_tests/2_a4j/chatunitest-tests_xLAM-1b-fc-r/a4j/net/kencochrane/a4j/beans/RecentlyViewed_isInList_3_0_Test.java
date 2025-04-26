package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class RecentlyViewed_isInList_3_0_Test {

    @Test
    public void isInListTest() {
        RecentlyViewed recentlyViewed = new RecentlyViewed();
        MiniProduct mp = new MiniProduct();
        mp.setAsin("12345");
        recentlyViewed.getProducts().add(mp);
        assertTrue(recentlyViewed.isInList("12345"));
        assertFalse(recentlyViewed.isInList("67890"));
    }
}
