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
    public void testIsInList_NullList() {
        assertFalse(recentlyViewed.isInList("1234567890"));
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
