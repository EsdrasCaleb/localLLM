package net.kencochrane.a4j.beans;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

public class BlendedSearch_printProductList_4_3_Test {

    @Mock
    BlendedSearch blendedSearch;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        List<ProductLine> productLines = new ArrayList<>(Arrays.asList(new ProductLine(), new ProductLine()));
        Mockito.when(blendedSearch.getProductLine()).thenReturn(productLines.toArray(new ProductLine[0]));
    }

    @Test
    public void testPrintProductList() {
        // Test with null productLines
        Mockito.when(blendedSearch.getProductLine()).thenReturn(null);
        Assertions.assertEquals("productLines is null \n", blendedSearch.printProductList());
        // Test with non-null productLines
        Mockito.when(blendedSearch.getProductLine()).thenReturn(new ProductLine[] { new ProductLine(), new ProductLine() });
        String expectedOutput = "ProductLine 1:\nProduct Line 2:\n# of productLines = 2\n";
        Assertions.assertEquals(expectedOutput, blendedSearch.printProductList());
    }
}
