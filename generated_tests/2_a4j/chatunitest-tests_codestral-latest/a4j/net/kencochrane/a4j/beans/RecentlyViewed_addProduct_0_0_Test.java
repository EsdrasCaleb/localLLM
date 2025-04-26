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

    @InjectMocks
    private RecentlyViewed recentlyViewed;

    @Mock
    private MiniProduct miniProduct;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAddProduct_NullProduct() {
        recentlyViewed.addProduct(null);
        assertEquals(0, recentlyViewed.getNumProducts());
    }

    @Test
    public void testAddProduct_ProductAlreadyInList() {
        when(miniProduct.getAsin()).thenReturn("123");
        recentlyViewed.addProduct(miniProduct);
        recentlyViewed.addProduct(miniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
    }

    @Test
    public void testAddProduct_NewProduct() {
        when(miniProduct.getAsin()).thenReturn("123");
        recentlyViewed.addProduct(miniProduct);
        assertEquals(1, recentlyViewed.getNumProducts());
    }
}
