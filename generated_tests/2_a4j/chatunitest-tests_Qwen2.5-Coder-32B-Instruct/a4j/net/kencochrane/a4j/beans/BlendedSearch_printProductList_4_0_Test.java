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

public class BlendedSearch_printProductList_4_0_Test {

    @InjectMocks
    private BlendedSearch blendedSearch;

    @Mock
    private ProductLine mockProductLine1;

    @Mock
    private ProductLine mockProductLine2;

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        // Initialize productLines field using reflection
        Field productLinesField = BlendedSearch.class.getDeclaredField("productLines");
        productLinesField.setAccessible(true);
        productLinesField.set(blendedSearch, new ArrayList<>());
    }

    @Test
    public void testPrintProductListWithNonNullProductLines() {
        // Arrange
        when(mockProductLine1.printProductList()).thenReturn("Product 1 Details");
        when(mockProductLine2.printProductList()).thenReturn("Product 2 Details");
        ArrayList<ProductLine> productLines = blendedSearch.getProductLinesArrayList();
        productLines.add(mockProductLine1);
        productLines.add(mockProductLine2);
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        String expectedOutput = "Product 1 Details\nProduct 2 Details\n# of productLines = 2\n";
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testPrintProductListWithEmptyProductLines() {
        // Arrange
        ArrayList<ProductLine> productLines = blendedSearch.getProductLinesArrayList();
        productLines.clear();
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        String expectedOutput = "# of productLines = 0\n";
        assertEquals(expectedOutput, result);
    }

    @Test
    public void testPrintProductListWithNullProductLines() throws Exception {
        // Arrange
        Field productLinesField = BlendedSearch.class.getDeclaredField("productLines");
        productLinesField.setAccessible(true);
        productLinesField.set(blendedSearch, null);
        // Act
        String result = blendedSearch.printProductList();
        // Assert
        String expectedOutput = "productLines is null \n";
        assertEquals(expectedOutput, result);
    }
}
