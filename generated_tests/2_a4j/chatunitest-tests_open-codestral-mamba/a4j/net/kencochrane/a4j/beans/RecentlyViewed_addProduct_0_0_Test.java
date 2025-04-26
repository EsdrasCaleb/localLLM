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

    @Mock
    private MiniProduct miniProduct;

    @InjectMocks
    private RecentlyViewed recentlyViewed;

    @Test
    public void testAddProduct() {
        MockitoAnnotations.initMocks(this);
        when(miniProduct.getAsin()).thenReturn("123456");
        recentlyViewed.addProduct(miniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
    }
}
