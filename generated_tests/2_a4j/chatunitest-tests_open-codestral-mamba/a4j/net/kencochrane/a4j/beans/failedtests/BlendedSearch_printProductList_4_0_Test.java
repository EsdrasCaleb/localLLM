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

    @Mock
    private BlendedSearch blendedSearch;

    @InjectMocks
    private BlendedSearch blendedSearchUnderTest;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testPrintProductList() {
        ArrayList<ProductLine> productLines = new ArrayList<>();
        ProductLine productLine1 = new ProductLine();
        ProductLine productLine2 = new ProductLine();
        productLines.add(productLine1);
        productLines.add(productLine2);
        when(blendedSearch.getProductLinesArrayList()).thenReturn(productLines);
        when(productLine1.printProductList()).thenReturn("Product Line 1");
        when(productLine2.printProductList()).thenReturn("Product Line 2");
        String result = blendedSearchUnderTest.printProductList();
        assertEquals("Product Line 1\nProduct Line 2\n# of productLines = 2\n", result);
    }

    @Test
    public void testPrintProductListNull() {
        when(blendedSearch.getProductLinesArrayList()).thenReturn(null);
        String result = blendedSearchUnderTest.printProductList();
        assertEquals("productLines is null \n", result);
    }
}
