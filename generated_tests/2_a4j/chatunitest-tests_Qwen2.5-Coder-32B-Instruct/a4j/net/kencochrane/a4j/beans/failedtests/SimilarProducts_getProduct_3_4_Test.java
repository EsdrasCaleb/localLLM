package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class SimilarProducts_getProduct_3_4_Test {

    @InjectMocks
    private SimilarProducts similarProducts;

    @Mock
    private ArrayList<String> simProductsMock;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize the simProducts field using reflection
        Field field = SimilarProducts.class.getDeclaredField("simProducts");
        field.setAccessible(true);
        field.set(similarProducts, new ArrayList<>());
    }

    @Test
    public void testGetProduct_IndexWithinBounds() {
        // Arrange
        ArrayList<String> simProducts = (ArrayList<String>) similarProducts.getProductsArray();
        simProducts.add("Product1");
        simProducts.add("Product2");
        // Act
        String result = similarProducts.getProduct(1);
        // Assert
        assertEquals("Product2", result);
    }

    @Test
    public void testGetProduct_IndexOutOfBounds() {
        // Arrange
        ArrayList<String> simProducts = (ArrayList<String>) similarProducts.getProductsArray();
        simProducts.add("Product1");
        // Act
        String result = similarProducts.getProduct(2);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetProduct_EmptyList() {
        // Arrange
        ArrayList<String> simProducts = (ArrayList<String>) similarProducts.getProductsArray();
        simProducts.clear();
        // Act
        String result = similarProducts.getProduct(0);
        // Assert
        assertNull(result);
    }

    @Test
    public void testGetProduct_NegativeIndex() {
        // Arrange
        ArrayList<String> simProducts = (ArrayList<String>) similarProducts.getProductsArray();
        simProducts.add("Product1");
        // Act
        String result = similarProducts.getProduct(-1);
        // Assert
        assertNull(result);
    }
}
