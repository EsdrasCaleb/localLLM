package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_toString_3_0_Test {

    private BlendedSearch blendedSearch;

    @BeforeEach
    public void setUp() {
        blendedSearch = new BlendedSearch();
    }

    @Test
    public void testToString_WithProductLines() {
        // Arrange
        ProductLine productLine1 = mock(ProductLine.class);
        ProductLine productLine2 = mock(ProductLine.class);
        when(productLine1.toString()).thenReturn("ProductLine1");
        when(productLine2.toString()).thenReturn("ProductLine2");
        blendedSearch.setProductLine(new ProductLine[] { productLine1, productLine2 });
        // Act
        String result = blendedSearch.toString();
        // Assert
        String expected = "ProductLine1\nProductLine2\n# of productLines = 2\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_EmptyProductLines() {
        // Arrange
        blendedSearch.setProductLine(new ProductLine[] {});
        // Act
        String result = blendedSearch.toString();
        // Assert
        String expected = "# of productLines = 0\n";
        assertEquals(expected, result);
    }

    @Test
    public void testToString_NullProductLines() {
        // Arrange
        blendedSearch.setProductLine(null);
        // Act
        String result = blendedSearch.toString();
        // Assert
        String expected = "productLines is null \n";
        assertEquals(expected, result);
    }
}
