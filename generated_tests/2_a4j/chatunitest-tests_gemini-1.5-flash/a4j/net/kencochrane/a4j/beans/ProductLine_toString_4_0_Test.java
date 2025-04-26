package net.kencochrane.a4j.beans;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.io.Serializable;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ProductLine_toString_4_0_Test {

    @Test
    void testToString_nullModeAndProductInfo() {
        ProductLine productLine = new ProductLine();
        String expected = "Mode = null\nnull\n";
        assertEquals(expected, productLine.toString());
    }

    @Test
    void testToString_modeOnly() {
        ProductLine productLine = new ProductLine();
        productLine.setMode("Online");
        String expected = "Mode = Online\nnull\n";
        assertEquals(expected, productLine.toString());
    }

    @Test
    void testToString_productInfoOnly() {
        ProductInfo productInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(productInfo.toString()).thenReturn("Product Info Details");
        ProductLine productLine = new ProductLine();
        productLine.setProductInfo(productInfo);
        String expected = "Mode = null\nProduct Info Details\n";
        assertEquals(expected, productLine.toString());
    }

    @Test
    void testToString_modeAndProductInfo() {
        ProductInfo productInfo = Mockito.mock(ProductInfo.class);
        Mockito.when(productInfo.toString()).thenReturn("Product Info Details");
        ProductLine productLine = new ProductLine();
        productLine.setMode("Offline");
        productLine.setProductInfo(productInfo);
        String expected = "Mode = Offline\nProduct Info Details\n";
        assertEquals(expected, productLine.toString());
    }

    // This is needed for the test to compile,  replace with your actual ProductInfo class if different.
    static class ProductInfo implements Serializable {

        @Override
        public String toString() {
            return "Product Info Details";
        }
    }

    // This is needed for the test to compile, replace with your actual ProductLine class if different.
    static class ProductLine implements Serializable {

        private String mode;

        private ProductInfo productInfo;

        public void setMode(String mode) {
            this.mode = mode;
        }

        public void setProductInfo(ProductInfo productInfo) {
            this.productInfo = productInfo;
        }

        @Override
        public String toString() {
            return "Mode = " + mode + "\n" + (productInfo != null ? productInfo.toString() : "null") + "\n";
        }
    }
}
