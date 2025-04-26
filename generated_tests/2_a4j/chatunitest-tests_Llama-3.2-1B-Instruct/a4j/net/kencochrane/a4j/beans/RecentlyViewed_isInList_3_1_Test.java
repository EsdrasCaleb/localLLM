package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import java.util.ArrayList;

public class RecentlyViewed_isInList_3_1_Test {

    @Test
    public void testIsInList() {
        RecentlyViewed focal = new RecentlyViewed();
        String asin = "1234567890";
        assertTrue(focal.isInList(asin));
        assertFalse(focal.isInList("1234567891"));
    }
}
