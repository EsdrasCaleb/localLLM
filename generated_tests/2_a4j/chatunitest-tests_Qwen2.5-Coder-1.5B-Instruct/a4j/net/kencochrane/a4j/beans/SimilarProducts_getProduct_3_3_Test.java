package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import static org.mockito.MockitoAnnotations.openMocks;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class SimilarProducts_getProduct_3_3_Test {

    @Mock
    private SimilarProducts similarProducts;

    @BeforeEach
    public void setUp() {
        openMocks(this);
    }

    @Test
    public void testGetProductValidIndex() throws Exception {
        // Arrange
        when(similarProducts.getProduct(0)).thenReturn("Product1");
        // Act
        String result = similarProducts.getProduct(0);
        // Assert
        assertEquals("Product1", result);
    }

    @Test
    public void testGetProductInvalidIndex() throws Exception {
        // Arrange
        when(similarProducts.getProduct(5)).thenReturn(null);
        // Act
        String result = similarProducts.getProduct(5);
        // Assert
        assertNull(result);
    }
}
