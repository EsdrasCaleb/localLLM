// Test method
package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
public class BlendedSearch_toString_3_0_Test {

    @Mock
    private ProductLine productLine1;

    @Mock
    private ProductLine productLine2;

    private BlendedSearch blendedSearch;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        blendedSearch = new BlendedSearch();
    }

    @Test
    public void testToString_ProductLinesNotNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        ArrayList<ProductLine> productLines = new ArrayList<>();
        when(productLine1.toString()).thenReturn("ProductLine1");
        when(productLine2.toString()).thenReturn("ProductLine2");
        productLines.add(productLine1);
        productLines.add(productLine2);
        Field field = blendedSearch.getClass().getDeclaredField("productLines");
        field.setAccessible(true);
        field.set(blendedSearch, productLines);
        // Act
        String result = blendedSearch.toString();
        // Assert
        assertEquals("ProductLine1\nProductLine2\n# of productLines = 2\n", result);
    }

    @Test
    public void testToString_ProductLinesNull() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        Field field = blendedSearch.getClass().getDeclaredField("productLines");
        field.setAccessible(true);
        field.set(blendedSearch, null);
        // Act
        String result = blendedSearch.toString();
        // Assert
        assertEquals("productLines is null \n", result);
    }

    @Test
    public void testToString_ProductLinesEmpty() throws NoSuchFieldException, IllegalAccessException {
        // Arrange
        ArrayList<ProductLine> productLines = new ArrayList<>();
        Field field = blendedSearch.getClass().getDeclaredField("productLines");
        field.setAccessible(true);
        field.set(blendedSearch, productLines);
        // Act
        String result = blendedSearch.toString();
        // Assert
        assertEquals("# of productLines = 0\n", result);
    }

    // Mock ProductLine class for testing purposes
    static class ProductLine {

        @Override
        public String toString() {
            return super.toString();
        }
    }
}
