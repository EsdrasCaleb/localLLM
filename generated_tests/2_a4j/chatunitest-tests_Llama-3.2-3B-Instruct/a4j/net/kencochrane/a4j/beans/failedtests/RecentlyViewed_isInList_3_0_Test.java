package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import java.io.Serializable;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
@MockitoSettings
public class RecentlyViewed_isInList_3_0_Test {

    @InjectMocks
    private RecentlyViewed recentlyViewed;

    @Mock
    private ArrayList<MiniProduct> products;

    @Test
    public void testIsInList_EmptyList() {
        assertTrue(recentlyViewed.isInList("1234567890"));
    }

    @Test
    public void testIsInList_NotFound() {
        recentlyViewed.getProducts().add(new MiniProduct("1234567890"));
        assertFalse(recentlyViewed.isInList("0987654321"));
    }

    @Test
    public void testIsInList_Found() {
        recentlyViewed.getProducts().add(new MiniProduct("1234567890"));
        assertTrue(recentlyViewed.isInList("1234567890"));
    }

    @Test
    public void testIsInList_NullList() {
        assertFalse(recentlyViewed.isInList("1234567890"));
    }

    @Test
    public void testIsInList_NullAsin() {
        recentlyViewed.getProducts().add(new MiniProduct("1234567890"));
        assertFalse(recentlyViewed.isInList(null));
    }

    @Test
    public void testIsInList_EmptyString() {
        recentlyViewed.getProducts().add(new MiniProduct("1234567890"));
        assertFalse(recentlyViewed.isInList(""));
    }

    @Test
    public void testIsInList_CaseInsensitive() {
        recentlyViewed.getProducts().add(new MiniProduct("1234567890"));
        assertTrue(recentlyViewed.isInList("1234567890"));
        assertTrue(recentlyViewed.isInList("1234567890"));
        assertTrue(recentlyViewed.isInList("1 2 3 4 5 6 7 8 9 0"));
    }
}

class MiniProduct implements Serializable {

    private String id;

    public MiniProduct(String id) {
        this.id = id;
    }

    public String getId() {
        return id;
    }
}
