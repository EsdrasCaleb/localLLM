package net.kencochrane.a4j.beans;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.Serializable;

@ExtendWith(MockitoExtension.class)
class BlendedSearch_printProductList_4_0_Test {

    @Test
    void testPrintProductList_nullProductLines() {
        BlendedSearch search = new BlendedSearch();
        assertEquals("productLines is null \n", search.printProductList());
    }

    @Test
    void testPrintProductList_emptyProductLines() {
        BlendedSearch search = new BlendedSearch();
        search.setProductLine(new ProductLine[0]);
        assertEquals("# of productLines = 0\n", search.printProductList());
    }

    @Test
    void testPrintProductList_multipleProductLines() throws NoSuchFieldException, IllegalAccessException {
        ProductLine productLine1 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine1.printProductList()).thenReturn("Product Line 1");
        ProductLine productLine2 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine2.printProductList()).thenReturn("Product Line 2");
        BlendedSearch search = new BlendedSearch();
        Field productLinesField = BlendedSearch.class.getDeclaredField("productLines");
        productLinesField.setAccessible(true);
        List<ProductLine> productLinesList = new ArrayList<>(Arrays.asList(productLine1, productLine2));
        productLinesField.set(search, productLinesList);
        String expectedOutput = "Product Line 1\n" + "Product Line 2\n" + "# of productLines = 2\n";
        assertEquals(expectedOutput, search.printProductList());
    }

    @Test
    void testPrintProductList_singleProductLine() throws NoSuchFieldException, IllegalAccessException {
        ProductLine productLine1 = Mockito.mock(ProductLine.class);
        Mockito.when(productLine1.printProductList()).thenReturn("Product Line 1");
        BlendedSearch search = new BlendedSearch();
        Field productLinesField = BlendedSearch.class.getDeclaredField("productLines");
        productLinesField.setAccessible(true);
        List<ProductLine> productLinesList = new ArrayList<>(Arrays.asList(productLine1));
        productLinesField.set(search, productLinesList);
        String expectedOutput = "Product Line 1\n" + "# of productLines = 1\n";
        assertEquals(expectedOutput, search.printProductList());
    }

    static class ProductLine {

        public String printProductList() {
            return "";
        }
    }

    static class BlendedSearch {

        private List<ProductLine> productLines;

        public void setProductLine(ProductLine[] productLines) {
            this.productLines = Arrays.asList(productLines);
        }

        public String printProductList() {
            if (productLines == null) {
                return "productLines is null \n";
            } else if (productLines.isEmpty()) {
                return "# of productLines = 0\n";
            } else {
                StringBuilder sb = new StringBuilder();
                for (ProductLine line : productLines) {
                    sb.append(line.printProductList()).append("\n");
                }
                sb.append("# of productLines = ").append(productLines.size()).append("\n");
                return sb.toString();
            }
        }
    }
}
