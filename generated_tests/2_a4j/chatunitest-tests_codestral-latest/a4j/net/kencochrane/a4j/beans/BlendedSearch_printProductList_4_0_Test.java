package net.kencochrane.a4j.beans;

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
    private ArrayList<ProductLine> productLines;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testPrintProductList_ProductLinesNotNull() {
        ProductLine productLine1 = mock(ProductLine.class);
        ProductLine productLine2 = mock(ProductLine.class);
        when(productLine1.printProductList()).thenReturn("Product 1");
        when(productLine2.printProductList()).thenReturn("Product 2");
        when(productLines.size()).thenReturn(2);
        when(productLines.get(0)).thenReturn(productLine1);
        when(productLines.get(1)).thenReturn(productLine2);
        blendedSearch.setProductLine(new ProductLine[] { productLine1, productLine2 });
        String expectedOutput = "Product 1\nProduct 2\n# of productLines = 2\n";
        String actualOutput = blendedSearch.printProductList();
        assertEquals(expectedOutput, actualOutput);
    }
}
