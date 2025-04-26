package net.kencochrane.a4j.beans;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;

class ProductLine_printProductList_5_1_Test {

    @Test
    void testPrintProductList_normalCase() {
        ProductInfo mockProductInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(mockProductInfo.printProductList()).thenReturn("Product A, Product B");
        ProductLine productLine = new ProductLine();
        productLine.setMode("Mode X");
        productLine.setProductInfo(mockProductInfo);
        String expectedOutput = "Mode = Mode X\nProduct A, Product B\n";
        assertEquals(expectedOutput, productLine.printProductList());
    }

    @Test
    void testPrintProductList_nullMode() {
        ProductInfo mockProductInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(mockProductInfo.printProductList()).thenReturn("Product C");
        ProductLine productLine = new ProductLine();
        productLine.setMode(null);
        productLine.setProductInfo(mockProductInfo);
        String expectedOutput = "Mode = null\nProduct C\n";
        assertEquals(expectedOutput, productLine.printProductList());
    }

    @Test
    void testPrintProductList_nullProductInfo() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("Mode Y");
        productLine.setProductInfo(null);
        String expectedOutput = "Mode = Mode Y\nnull\n";
        assertEquals(expectedOutput, productLine.printProductList());
    }

    @Test
    void testPrintProductList_emptyProductInfo() {
        ProductInfo mockProductInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(mockProductInfo.printProductList()).thenReturn("");
        ProductLine productLine = new ProductLine();
        productLine.setMode("Mode Z");
        productLine.setProductInfo(mockProductInfo);
        String expectedOutput = "Mode = Mode Z\n\n";
        assertEquals(expectedOutput, productLine.printProductList());
    }

    static class ProductInfo {

        public String printProductList() {
            return "";
        }
    }

    static class ProductLine {

        private String mode;

        private ProductInfo productInfo;

        public void setMode(String mode) {
            this.mode = mode;
        }

        public void setProductInfo(ProductInfo productInfo) {
            this.productInfo = productInfo;
        }

        public String printProductList() {
            String modeString = (mode != null) ? "Mode = " + mode : "Mode = null";
            String productListString = (productInfo != null) ? productInfo.printProductList() : "null";
            return modeString + "\n" + productListString + "\n";
        }
    }
}
