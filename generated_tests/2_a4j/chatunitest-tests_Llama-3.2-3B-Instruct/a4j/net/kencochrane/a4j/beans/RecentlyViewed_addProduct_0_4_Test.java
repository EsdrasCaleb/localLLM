// RecentlyViewed_addProduct_0_4_Test.java
package net.kencochrane.a4j.beans;

import java.io.Serializable;
import java.util.ArrayList;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class RecentlyViewed_addProduct_0_4_Test {

    @Mock
    private RecentlyViewed recentlyViewed;

    @BeforeEach
    void setup() {
        recentlyViewed = new RecentlyViewed();
    }

    @Test
    public void testAddProduct_NullProduct() {
        assertDoesNotThrow(() -> recentlyViewed.addProduct(null));
    }
}

class MiniProduct {

    private String id;

    public MiniProduct(String id) {
        this.id = id;
    }

    // Add a no-arg constructor
    public MiniProduct() {
    }
}
